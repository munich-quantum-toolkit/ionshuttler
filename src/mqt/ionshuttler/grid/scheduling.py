# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Scheduling strategies for Grid compilation."""

from __future__ import annotations

import operator
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.grid.actions import Cycle, JunctionMove, Rearrangement
from mqt.ionshuttler.grid.model import SegmentOccupancy
from mqt.ionshuttler.grid.planning_trace import IdleReason, MoveStatus, PlannedGate, PlanningSnapshot, PrioritizedIon
from mqt.ionshuttler.grid.result import GridDiagnostics, GridStopReason
from mqt.ionshuttler.grid.routing import (
    clear_blocked_path,
    junction_moves,
    next_segments_toward,
    route_gate_bfs,
    route_junction_crossing_greedy_cycle,
    route_to_placement,
    routing_distance,
    select_greedy_target,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.circuit import Circuit
    from mqt.ionshuttler.grid.architecture import GridArchitecture
    from mqt.ionshuttler.grid.config import GridCompilerConfig
    from mqt.ionshuttler.grid.state import GridMachineState


@dataclass(frozen=True)
class _GreedyGatePlan:
    """Collect the gate choices for one greedy scheduling decision."""

    ready_gates: tuple[GateAction, ...]
    selected_ready_gates: tuple[GateAction, ...]
    lookahead_gates: tuple[GateAction, ...]


def schedule_breadth_first(
    architecture: GridArchitecture,
    circuit: Circuit,
    initial_state: GridMachineState,
    config: GridCompilerConfig,
    gate_to_segment_assignment: Mapping[int, str],
    target_state: GridMachineState | None = None,
    partial_target_by_ion: Mapping[int, str] | None = None,
) -> tuple[Schedule[Action, GridMachineState], GridDiagnostics]:
    """Schedule gates with a bounded shortest-route search.

    Returns:
        The resulting schedule prefix and diagnostic counters.
    """
    state = initial_state
    scheduled_actions: list[ScheduledAction[Action]] = []
    completed: set[int] = set()
    gate_to_segment_lock = dict(gate_to_segment_assignment)
    iterations = explored_states = 0
    planning_trace: list[PlanningSnapshot] | None = [] if config.record_planning_trace else None
    stalled = False

    while len(completed) < len(circuit.gates) and iterations < config.max_iterations:
        iterations += 1
        ready = _ready_gates(circuit, completed)
        gate_layer = _startable_gate_layer(
            architecture,
            state,
            ready,
            len(scheduled_actions),
            gate_to_segment_lock=gate_to_segment_lock,
        )
        if gate_layer:
            if planning_trace is not None:
                started_gates = _layer_gates(gate_layer)
                started_segments = {
                    gate_id: architecture.processing_zone_segment(item.processing_zone_id).segment_id
                    for item in gate_layer
                    if isinstance(item.action, GateAction)
                    and (gate_id := item.action.gate_id) is not None
                    and item.processing_zone_id is not None
                }
                planning_trace.append(
                    _planning_snapshot(
                        iterations,
                        state.time,
                        ready,
                        started_gates,
                        gate_to_target_segment=gate_to_segment_lock | started_segments,
                        started_gates=started_gates,
                    )
                )
            state = architecture.apply_layer(state, gate_layer)
            scheduled_actions.extend(gate_layer)
            completed.update(_action_layer_gate_ids(gate_layer))
            state = state.at_time(max(item.end_time for item in gate_layer))
            continue
        if not ready:
            if planning_trace is not None:
                planning_trace.append(
                    _planning_snapshot(
                        iterations,
                        state.time,
                        ready,
                        (),
                        gate_to_target_segment=gate_to_segment_lock,
                        idle_reason=IdleReason.NO_LEGAL_ACTION,
                    )
                )
            stalled = True
            break
        gate = ready[0]
        gate_id = gate.gate_id
        target_segments = (
            (gate_to_segment_assignment[gate_id],)
            if gate_id is not None and gate_id in gate_to_segment_assignment
            else _candidate_target_segments(architecture, gate)
        )
        found_route, explored = route_gate_bfs(
            architecture,
            state,
            gate,
            frozenset(target_segments),
            max_states=config.max_routing_states,
            allowed_crossings=config.allowed_junction_crossings,
        )
        explored_states += explored
        route = found_route or ()
        decision_time = state.time
        for action in route:
            timed_action = ScheduledAction(
                len(scheduled_actions),
                action,
                state.time,
                architecture.action_duration(action),
            )
            state = architecture.apply_layer(state, (timed_action,)).at_time(timed_action.end_time)
            scheduled_actions.append(timed_action)
        if planning_trace is not None:
            # breadth-first search may choose any compatible processing-zone segment; record the one it reached.
            routed_target = {} if not route or gate_id is None else {gate_id: state.ion_segment(gate.ions[0])}
            planning_trace.append(
                _planning_snapshot(
                    iterations,
                    decision_time,
                    ready,
                    (gate,),
                    gate_to_target_segment=gate_to_segment_lock | routed_target,
                    moves=route,
                    idle_reason=None if route else IdleReason.NO_ROUTE,
                )
            )
        if not route:
            stalled = True
            break

    gates_complete = len(completed) == len(circuit.gates)
    if (
        (target_state is not None or partial_target_by_ion is not None)
        and gates_complete
        and not _placement_reached(state, target_state, partial_target_by_ion)
        and iterations < config.max_iterations
    ):
        iterations += 1
        decision_time = state.time
        state, target_layer, explored = _route_to_target(
            architecture,
            state,
            config,
            target_state,
            partial_target_by_ion,
            len(scheduled_actions),
        )
        explored_states += explored
        scheduled_actions.extend(target_layer)
        if planning_trace is not None:
            planning_trace.append(
                _planning_snapshot(
                    iterations,
                    decision_time,
                    (),
                    (),
                    gate_to_target_segment={},
                    moves=tuple(item.action for item in target_layer),
                    idle_reason=None if target_layer else IdleReason.NO_ROUTE,
                )
            )
        if not target_layer:
            stalled = True

    completed_objective = gates_complete and _placement_reached(state, target_state, partial_target_by_ion)

    junction_move_count, cycle_count, rearrangement_count = _count_transport(scheduled_actions)
    return (
        Schedule(tuple(scheduled_actions), state.time, initial_state),
        GridDiagnostics(
            iterations=iterations,
            explored_routing_states=explored_states,
            junction_moves=junction_move_count,
            cycles=cycle_count,
            rearrangements=rearrangement_count,
            gates=len(completed),
            stop_reason=_stop_reason(completed=completed_objective, stalled=stalled),
            planning_trace=None if planning_trace is None else tuple(planning_trace),
        ),
    )


def schedule_greedy(
    architecture: GridArchitecture,
    circuit: Circuit,
    initial_state: GridMachineState,
    config: GridCompilerConfig,
    gate_to_segment_assignment: Mapping[int, str],
    target_state: GridMachineState | None = None,
    partial_target_by_ion: Mapping[int, str] | None = None,
) -> tuple[Schedule[Action, GridMachineState], GridDiagnostics]:
    """Schedule gates with distance-ranked lookahead and prioritized transport.

    Returns:
        The resulting schedule prefix and diagnostic counters.
    """
    state = initial_state
    scheduled_actions: list[ScheduledAction[Action]] = []
    completed: set[int] = set()
    started: dict[int, int] = {}
    gate_to_segment_lock: dict[int, str] = dict(gate_to_segment_assignment)
    iterations = explored_states = 0
    planning_trace: list[PlanningSnapshot] | None = [] if config.record_planning_trace else None
    stalled = False
    stall_clear_attempted_since_gate_completion = False
    # Transport decisions since the last gate start, keyed by placement and gate progress.
    placements_without_gate: set[object] = set()
    gates_by_ion: defaultdict[int, list[int]] = defaultdict(list)
    for gate in circuit.gates:
        if gate.gate_id is not None:
            for ion in gate.ions:
                gates_by_ion[ion].append(gate.gate_id)
    swap_guard = _SwapGuard(
        {ion: tuple(gate_ids) for ion, gate_ids in gates_by_ion.items()},
        completed,
        config.allowed_junction_crossings,
    )

    while len(completed) < len(circuit.gates) and iterations < config.max_iterations:
        iterations += 1
        completed_before_update = len(completed)
        _complete_started_gates(state.time, started, completed, gate_to_segment_lock)
        if len(completed) > completed_before_update:
            stall_clear_attempted_since_gate_completion = False
        if len(completed) == len(circuit.gates):
            break
        plan = _greedy_gate_plan(
            architecture,
            circuit,
            state,
            completed,
            started,
            gate_to_segment_lock,
            config.allowed_junction_crossings,
            gate_to_segment_assignment,
            lookahead_rounds=config.greedy_lookahead_rounds,
        )
        # A selected gate may follow a running gate on other ions, for example across a barrier.
        gate_layer = _startable_gate_layer(
            architecture,
            state,
            tuple(
                gate
                for gate in plan.selected_ready_gates
                if gate.gate_id is not None and circuit.predecessors[gate.gate_id] <= completed
            ),
            len(scheduled_actions),
            gate_to_segment_lock=gate_to_segment_lock,
        )
        for item in gate_layer:
            gate = item.action
            assert isinstance(gate, GateAction)
            assert gate.gate_id is not None
            started[gate.gate_id] = item.end_time
        priority = _greedy_ion_priority(
            architecture,
            state,
            plan.lookahead_gates,
            gate_to_segment_lock,
            config.allowed_junction_crossings,
            gate_to_segment_assignment,
        )
        retained_ions = _retained_pz_ions(
            architecture,
            state,
            plan.selected_ready_gates,
            gate_to_segment_lock,
            gate_layer,
        )
        planning_requests: list[PrioritizedIon] | None = None if planning_trace is None else []
        selected_gate_requests = frozenset(
            (ion, gate_to_segment_lock[gate.gate_id])
            for gate in plan.selected_ready_gates
            if not architecture.is_virtual_gate(gate) and gate.gate_id is not None
            for ion in gate.ions
        )
        movement_layer, explored, blocked_targets = _priority_movement_layer(
            architecture,
            state,
            len(scheduled_actions) + len(gate_layer),
            priority,
            gate_layer,
            retained_ions,
            max_states=config.max_routing_states,
            allowed_crossings=config.allowed_junction_crossings,
            selected_gate_requests=selected_gate_requests,
            planning_requests=planning_requests,
            swap_guard=swap_guard,
        )
        explored_states += explored
        if blocked_targets:
            path_clearing_layer, clearing_paths_explored = _build_clearing_layer_for_gate(
                architecture,
                state,
                len(scheduled_actions) + len(gate_layer) + len(movement_layer),
                priority,
                plan.selected_ready_gates,
                gate_to_segment_lock,
                blocked_targets=blocked_targets,
                fixed_actions=(*gate_layer, *movement_layer),
                max_states=config.max_routing_states,
                allowed_crossings=config.allowed_junction_crossings,
            )
            explored_states += clearing_paths_explored
            movement_layer = (*movement_layer, *path_clearing_layer)
        action_layer = (*gate_layer, *movement_layer)
        next_time = None if action_layer else _next_resource_release(state)
        if not action_layer and next_time is None:
            # Let a blocked incoming operand temporarily displace an operand already in its zone.
            incoming_ions_first = tuple(
                sorted(priority, key=lambda request: state.ion_segment(request[0]) == request[1])
            )
            movement_layer, additional_moves_explored, _blocked_targets = _priority_movement_layer(
                architecture,
                state,
                len(scheduled_actions),
                incoming_ions_first,
                (),
                frozenset(),
                max_states=config.max_routing_states,
                allowed_crossings=config.allowed_junction_crossings,
                selected_gate_requests=selected_gate_requests,
                swap_guard=swap_guard,
            )
            explored_states += additional_moves_explored
            action_layer = movement_layer
        if not action_layer and next_time is None and not stall_clear_attempted_since_gate_completion:
            stall_clear_attempted_since_gate_completion = True
            path_clearing_layer, clearing_paths_explored = _build_clearing_layer_for_gate(
                architecture,
                state,
                len(scheduled_actions),
                priority,
                plan.selected_ready_gates,
                gate_to_segment_lock,
                blocked_targets=frozenset(target for _ion, target in selected_gate_requests),
                fixed_actions=(),
                max_states=config.max_routing_states,
                allowed_crossings=config.allowed_junction_crossings,
            )
            explored_states += clearing_paths_explored
            movement_layer = path_clearing_layer
            action_layer = path_clearing_layer
        progress_key = (state.occupancy, frozenset(started), len(completed))
        if gate_layer:
            placements_without_gate.clear()
        elif movement_layer and progress_key in placements_without_gate:
            # The movement layers returned to an earlier placement without starting a gate.
            # Route the ions of one selected gate into its segment and start it.
            decision_time = state.time
            routed = _route_and_start_gate(
                architecture,
                state,
                config,
                plan.selected_ready_gates,
                scheduled_actions,
                started,
                completed,
                gate_to_segment_lock,
            )
            if routed is None:
                stalled = True
                break
            state, routed_layer, explored = routed
            explored_states += explored
            if planning_trace is not None:
                planning_trace.append(
                    _planning_snapshot(
                        iterations,
                        decision_time,
                        plan.ready_gates,
                        plan.selected_ready_gates,
                        gate_to_target_segment=gate_to_segment_lock,
                        lookahead_gates=plan.lookahead_gates,
                        started_gates=_layer_gates(routed_layer),
                        moves=tuple(item.action for item in routed_layer if not isinstance(item.action, GateAction)),
                    )
                )
            scheduled_actions.extend(routed_layer)
            placements_without_gate.clear()
            continue
        elif movement_layer:
            placements_without_gate.add(progress_key)
        if planning_trace is not None:
            planning_trace.append(
                _planning_snapshot(
                    iterations,
                    state.time,
                    plan.ready_gates,
                    plan.selected_ready_gates,
                    gate_to_target_segment=gate_to_segment_lock,
                    lookahead_gates=plan.lookahead_gates,
                    started_gates=_layer_gates(gate_layer),
                    prioritized_ions=planning_requests or (),
                    retained_ions=retained_ions,
                    moves=tuple(item.action for item in movement_layer),
                    idle_reason=_idle_reason(action_layer, next_time),
                )
            )
        if not action_layer:
            if next_time is None:
                stalled = True
                break
            state = state.at_time(next_time)
            continue
        targets = dict(priority)
        for item in action_layer:
            if isinstance(item.action, Rearrangement):
                swap_guard.record(item.action, swap_guard.progress(architecture, state, item.action.ions, targets))
        state = architecture.apply_layer(state, action_layer)
        scheduled_actions.extend(action_layer)
        if any(item.end_time > state.time for item in action_layer):
            state = state.at_time(state.time + 1)

    if scheduled_actions:
        state = state.at_time(max(state.time, *(action.end_time for action in scheduled_actions)))
        _complete_started_gates(state.time, started, completed, gate_to_segment_lock)

    gates_complete = len(completed) == len(circuit.gates)
    if (
        (target_state is not None or partial_target_by_ion is not None)
        and gates_complete
        and not _placement_reached(state, target_state, partial_target_by_ion)
        and iterations < config.max_iterations
    ):
        iterations += 1
        decision_time = state.time
        state, target_layer, explored = _route_to_target(
            architecture,
            state,
            config,
            target_state,
            partial_target_by_ion,
            len(scheduled_actions),
        )
        explored_states += explored
        scheduled_actions.extend(target_layer)
        if planning_trace is not None:
            planning_trace.append(
                _planning_snapshot(
                    iterations,
                    decision_time,
                    (),
                    (),
                    gate_to_target_segment={},
                    moves=tuple(item.action for item in target_layer),
                    idle_reason=None if target_layer else IdleReason.NO_ROUTE,
                )
            )
        if not target_layer:
            stalled = True

    completed_objective = gates_complete and _placement_reached(state, target_state, partial_target_by_ion)

    junction_move_count, cycle_count, rearrangement_count = _count_transport(scheduled_actions)
    return (
        Schedule(tuple(scheduled_actions), state.time, initial_state),
        GridDiagnostics(
            iterations=iterations,
            explored_routing_states=explored_states,
            junction_moves=junction_move_count,
            cycles=cycle_count,
            rearrangements=rearrangement_count,
            gates=len(completed),
            stop_reason=_stop_reason(completed=completed_objective, stalled=stalled),
            planning_trace=None if planning_trace is None else tuple(planning_trace),
        ),
    )


def _placement_reached(
    state: GridMachineState,
    target_state: GridMachineState | None,
    partial_target_by_ion: Mapping[int, str] | None,
) -> bool:
    """Return whether the optional exact or partial target is satisfied."""
    if target_state is not None:
        return state.occupancy == target_state.occupancy
    if partial_target_by_ion is not None:
        return all(state.ion_segment(ion) == segment_id for ion, segment_id in partial_target_by_ion.items())
    return True


def _route_to_target(
    architecture: GridArchitecture,
    state: GridMachineState,
    config: GridCompilerConfig,
    target_state: GridMachineState | None,
    partial_target_by_ion: Mapping[int, str] | None,
    first_action_id: int,
) -> tuple[GridMachineState, tuple[ScheduledAction[Action], ...], int]:
    """Move the ions of an idle state into the target placement.

    The placement search finds a sequential movement order. Each movement then
    starts as soon as its segments, junctions, and ions are free; movements on
    disjoint segments commute, so the outcome stays the same.

    Returns:
        The state after the movement, the timed movements (empty if the search
        finds none), and the number of explored states.
    """
    route, explored = route_to_placement(
        architecture,
        state,
        max_states=config.max_routing_states,
        target_state=target_state,
        partial_target_by_ion=partial_target_by_ion,
        allowed_crossings=config.allowed_junction_crossings,
    )
    free_at: dict[tuple[str, str | int], int] = {}
    timed: list[tuple[int, Action, int]] = []
    for action in route or ():
        resources: set[tuple[str, str | int]] = set()
        if isinstance(action, Rearrangement):
            resources.add(("segment", action.segment_id))
            resources.update(("ion", ion) for ion in action.ions)
            moves: tuple[Action, ...] = ()
        else:
            moves = action.moves if isinstance(action, Cycle) else (action,)
        for move in moves:
            assert isinstance(move, JunctionMove)
            resources.update({("segment", move.source.segment_id), ("segment", move.destination.segment_id)})
            resources.add(("junction", architecture.junction_for(move.source).junction_id))
            resources.update(("ion", ion) for ion in move.ions)
        start = max((free_at.get(resource, state.time) for resource in resources), default=state.time)
        duration = architecture.action_duration(action)
        timed.append((start, action, duration))
        for resource in resources:
            free_at[resource] = start + duration
    timed.sort(key=operator.itemgetter(0))
    layer = tuple(
        ScheduledAction(first_action_id + offset, action, start, duration)
        for offset, (start, action, duration) in enumerate(timed)
    )
    for _start_time, actions in groupby(layer, key=lambda item: item.start_time):
        state = architecture.apply_layer(state, tuple(actions))
    if layer:
        state = state.at_time(max(item.end_time for item in layer))
    return state, layer, explored


def _route_and_start_gate(
    architecture: GridArchitecture,
    state: GridMachineState,
    config: GridCompilerConfig,
    candidate_gates: Sequence[GateAction],
    scheduled_actions: Sequence[ScheduledAction[Action]],
    started: dict[int, int],
    completed: set[int],
    gate_to_segment_lock: dict[int, str],
) -> tuple[GridMachineState, tuple[ScheduledAction[Action], ...], int] | None:
    """Wait until the machine is idle, move the ions of one candidate gate into its segment, and start the gate.

    The placement search plans the movement for the gate's ions alone, so the
    ions reach the segment even when the greedy movement layers repeat a cycle.

    Returns:
        The state after the gate starts, the movements followed by the gate,
        and the number of explored states, or ``None`` if no candidate can
        start.
    """
    idle_time = max((item.end_time for item in scheduled_actions), default=state.time)
    state = state.at_time(max(state.time, idle_time))
    _complete_started_gates(state.time, started, completed, gate_to_segment_lock)
    explored_states = 0
    for gate in candidate_gates:
        if architecture.is_virtual_gate(gate) or gate.gate_id not in gate_to_segment_lock:
            continue
        segment_id = gate_to_segment_lock[gate.gate_id]
        routed_state, movements, explored = _route_to_target(
            architecture,
            state,
            config,
            None,
            dict.fromkeys(gate.ions, segment_id),
            len(scheduled_actions),
        )
        explored_states += explored
        gate_layer = _startable_gate_layer(
            architecture,
            routed_state,
            (gate,),
            len(scheduled_actions) + len(movements),
            gate_to_segment_lock=gate_to_segment_lock,
        )
        if not gate_layer:
            continue
        assert gate.gate_id is not None
        started[gate.gate_id] = gate_layer[0].end_time
        routed_state = architecture.apply_layer(routed_state, gate_layer)
        return routed_state.at_time(routed_state.time + 1), (*movements, *gate_layer), explored_states
    return None


def _stop_reason(*, completed: bool, stalled: bool) -> GridStopReason:
    """Return why scheduling stopped."""
    if completed:
        return GridStopReason.COMPLETED
    return GridStopReason.NO_PROGRESS if stalled else GridStopReason.ITERATION_LIMIT


def _idle_reason(action_layer: Sequence[ScheduledAction[Action]], next_release: int | None) -> IdleReason | None:
    """Return why a greedy decision selected no action, or ``None`` if it selected one."""
    if action_layer:
        return None
    return IdleReason.NO_LEGAL_ACTION if next_release is None else IdleReason.WAITING_FOR_RESOURCE


def _planning_snapshot(
    iteration: int,
    time: int,
    ready_gates: Sequence[GateAction],
    selected_gates: Sequence[GateAction],
    *,
    gate_to_target_segment: Mapping[int, str],
    lookahead_gates: Sequence[GateAction] = (),
    started_gates: Sequence[GateAction] = (),
    prioritized_ions: Sequence[PrioritizedIon] = (),
    retained_ions: frozenset[int] = frozenset(),
    moves: Sequence[Action] = (),
    idle_reason: IdleReason | None = None,
) -> PlanningSnapshot:
    """Build a trace snapshot from the gates and targets considered in one scheduling decision.

    Returns:
        The snapshot.
    """
    named = {
        gate.gate_id: gate for gate in (*selected_gates, *lookahead_gates, *started_gates) if gate.gate_id is not None
    }
    return PlanningSnapshot(
        iteration=iteration,
        time=time,
        ready_gates=_gate_ids(ready_gates),
        selected_gates=_gate_ids(selected_gates),
        lookahead_gates=_gate_ids(lookahead_gates),
        started_gates=_gate_ids(started_gates),
        gates=tuple(PlannedGate(named[gate_id], gate_to_target_segment.get(gate_id)) for gate_id in sorted(named)),
        prioritized_ions=tuple(prioritized_ions),
        retained_ions=tuple(sorted(retained_ions)),
        moves=tuple(moves),
        idle_reason=idle_reason,
    )


def _gate_ids(gates: Sequence[GateAction]) -> tuple[int, ...]:
    """Return the gate IDs of circuit gates."""
    return tuple(gate.gate_id for gate in gates if gate.gate_id is not None)


def _layer_gates(layer: Sequence[ScheduledAction[Action]]) -> tuple[GateAction, ...]:
    """Return the gates in one action layer."""
    return tuple(item.action for item in layer if isinstance(item.action, GateAction))


def _ready_gates(circuit: Circuit, completed: set[int]) -> tuple[GateAction, ...]:
    """Return gates whose dependencies are complete."""
    return tuple(
        gate
        for gate, predecessors in zip(circuit.gates, circuit.predecessors, strict=True)
        if gate.gate_id not in completed and predecessors <= completed
    )


def _action_layer_gate_ids(layer: Sequence[ScheduledAction[Action]]) -> set[int]:
    """Return the gate identifiers in one action layer."""
    return {
        item.action.gate_id for item in layer if isinstance(item.action, GateAction) and item.action.gate_id is not None
    }


def _count_transport(scheduled_actions: Sequence[ScheduledAction[Action]]) -> tuple[int, int, int]:
    """Count the transport of a schedule.

    Returns:
        The number of junction moves (including the moves inside cycles),
        cycles, and rearrangements.
    """
    moves = cycles = rearrangements = 0
    for item in scheduled_actions:
        if isinstance(item.action, Cycle):
            moves += len(item.action.moves)
            cycles += 1
        elif isinstance(item.action, JunctionMove):
            moves += 1
        elif isinstance(item.action, Rearrangement):
            rearrangements += 1
    return moves, cycles, rearrangements


def _complete_started_gates(
    time: int,
    started: dict[int, int],
    completed: set[int],
    gate_to_segment_lock: dict[int, str],
) -> None:
    """Mark elapsed gates as complete and release their segment locks."""
    for gate_id in sorted(gate_id for gate_id, end_time in started.items() if end_time <= time):
        completed.add(gate_id)
        started.pop(gate_id)
        gate_to_segment_lock.pop(gate_id, None)


def _greedy_gate_plan(
    architecture: GridArchitecture,
    circuit: Circuit,
    state: GridMachineState,
    completed: set[int],
    started: Mapping[int, int],
    gate_to_segment_lock: dict[int, str],
    allowed_crossings: frozenset[tuple[str, str]] | None,
    gate_to_segment_assignment: Mapping[int, str],
    *,
    lookahead_rounds: int,
) -> _GreedyGatePlan:
    """Choose at most one nearby gate per processing-zone segment over a short dependency horizon.

    Returns:
        The ready gates, the selected ready gates, and the distance-ranked lookahead order.
    """
    started_ids = set(started)
    virtually_completed = set(completed) | started_ids
    ready_gates: tuple[GateAction, ...] = ()
    lookahead_gates: list[GateAction] = []
    selected_ready_gates: list[GateAction] = []
    target_order = {
        architecture.processing_zone_segment(zone.zone_id).segment_id: index
        for index, zone in enumerate(architecture.processing_zones)
    }
    for round_index in range(lookahead_rounds):
        ready = tuple(gate for gate in _ready_gates(circuit, virtually_completed) if gate.gate_id not in started_ids)
        if not ready:
            break
        # A virtual gate claims no ion, so it may start only after its predecessors finish.
        virtual = tuple(
            gate
            for gate in ready
            if architecture.is_virtual_gate(gate)
            and (round_index > 0 or (gate.gate_id is not None and circuit.predecessors[gate.gate_id] <= completed))
        )
        gates_by_segment: dict[str, list[GateAction]] = defaultdict(list)
        for gate in ready:
            if architecture.is_virtual_gate(gate):
                continue
            try:
                target_segment_id = _select_gate_segment(
                    architecture,
                    state,
                    gate,
                    gate_to_segment_lock,
                    allowed_crossings,
                    gate_to_segment_assignment,
                )
            except ValueError:
                continue
            gates_by_segment[target_segment_id].append(gate)
        chosen = list(virtual)
        ordered_segment_ids = sorted(
            gates_by_segment,
            key=lambda item: (target_order.get(item, len(target_order)), item),
        )
        chosen.extend(
            min(
                gates_by_segment[target_segment_id],
                key=lambda gate: (
                    *_gate_distance_rank(
                        architecture,
                        state,
                        gate,
                        target_segment_id,
                        allowed_crossings,
                    ),
                    gate.gate_id,
                ),
            )
            for target_segment_id in ordered_segment_ids
        )
        chosen.sort(key=lambda gate: gate.gate_id if gate.gate_id is not None else -1)
        if round_index == 0:
            ready_gates = ready
            selected_ready_gates.extend(chosen)
        lookahead_gates.extend(chosen)
        virtually_completed.update(gate.gate_id for gate in chosen if gate.gate_id is not None)
    return _GreedyGatePlan(ready_gates, tuple(selected_ready_gates), tuple(lookahead_gates))


def _select_gate_segment(
    architecture: GridArchitecture,
    state: GridMachineState,
    gate: GateAction,
    gate_to_segment_lock: dict[int, str],
    allowed_crossings: frozenset[tuple[str, str]] | None,
    gate_to_segment_assignment: Mapping[int, str],
) -> str:
    """Select a processing-zone segment and lock it for each pending two-ion gate.

    Returns:
        The selected processing-zone segment ID.
    """
    gate_id = gate.gate_id
    assert gate_id is not None
    if gate_id in gate_to_segment_assignment:
        return gate_to_segment_assignment[gate_id]
    if len(gate.ions) == 1 or gate_id not in gate_to_segment_lock:
        gate_to_segment_lock[gate_id] = select_greedy_target(
            architecture,
            state,
            gate,
            _candidate_target_segments(architecture, gate),
            allowed_crossings,
        )
    return gate_to_segment_lock[gate_id]


def _gate_distance_rank(
    architecture: GridArchitecture,
    state: GridMachineState,
    gate: GateAction,
    target_segment_id: str,
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> tuple[int, int]:
    """Return maximum and total operand distance for ready-gate selection."""
    distances = [
        routing_distance(architecture, state.ion_segment(ion), target_segment_id, allowed_crossings)
        for ion in gate.ions
    ]
    if any(distance is None for distance in distances):
        return (10**9, 10**9)
    reachable_distances = tuple(distance for distance in distances if distance is not None)
    return max(reachable_distances), sum(reachable_distances)


def _candidate_target_segments(
    architecture: GridArchitecture,
    gate: GateAction,
) -> tuple[str, ...]:
    """Return processing-zone segments that support the gate."""
    zones = tuple(
        zone
        for zone in architecture.processing_zones
        if zone.supports(gate) and architecture.processing_zone_segment(zone.zone_id).capacity >= len(gate.ions)
    )
    return tuple(dict.fromkeys(architecture.processing_zone_segment(zone.zone_id).segment_id for zone in zones))


def _greedy_ion_priority(
    architecture: GridArchitecture,
    state: GridMachineState,
    ordered_gates: Sequence[GateAction],
    gate_to_segment_lock: dict[int, str],
    allowed_crossings: frozenset[tuple[str, str]] | None,
    gate_to_segment_assignment: Mapping[int, str],
) -> tuple[tuple[int, str], ...]:
    """Prioritize the first target of each ion in the selected lookahead gates.

    Returns:
        Prioritized ion and target-segment pairs selected for movement.
    """
    planned_ions: set[tuple[int, str]] = set()
    priority: dict[int, str] = {}
    for gate in ordered_gates:
        if architecture.is_virtual_gate(gate):
            continue
        target_segment_id = _select_gate_segment(
            architecture,
            state,
            gate,
            gate_to_segment_lock,
            allowed_crossings,
            gate_to_segment_assignment,
        )
        for ion in gate.ions:
            priority.setdefault(ion, target_segment_id)
            planned_ions.add((ion, target_segment_id))

    by_target: dict[str, list[int]] = defaultdict(list)
    for ion, target_segment_id in priority.items():
        by_target[target_segment_id].append(ion)
    selected: set[int] = set()
    for target_segment_id, ions in by_target.items():
        selected_distances: list[int] = []
        for index, ion in enumerate(ions):
            distance = routing_distance(
                architecture,
                state.ion_segment(ion),
                target_segment_id,
                allowed_crossings,
            )
            if distance is None:
                continue
            if (
                (distance == 0 and (ion, target_segment_id) in planned_ions)
                or index == 0
                or not any(selected_distances)
                or all(distance >= value for value in selected_distances)
            ):
                selected.add(ion)
                selected_distances.append(distance)
    return tuple((ion, target) for ion, target in priority.items() if ion in selected)


def _retained_pz_ions(
    architecture: GridArchitecture,
    state: GridMachineState,
    selected_gates: Sequence[GateAction],
    gate_to_segment_lock: Mapping[int, str],
    gate_layer: Sequence[ScheduledAction[Action]],
) -> frozenset[int]:
    """Return PZ ions needed by a selected, running, or starting gate."""
    retained: set[int] = set()
    for gate in selected_gates:
        if architecture.is_virtual_gate(gate) or gate.gate_id is None:
            continue
        target_segment_id = gate_to_segment_lock[gate.gate_id]
        retained.update(ion for ion in gate.ions if state.ion_segment(ion) == target_segment_id)
    busy_ions = {ion for ion, busy_until in state.ions_busy_until if busy_until > state.time}
    for zone_id, busy_until in state.pzs_busy_until:
        if busy_until > state.time:
            retained.update(
                busy_ions.intersection(state.occupants(architecture.processing_zone_segment(zone_id).segment_id))
            )
    for item in gate_layer:
        if isinstance(item.action, GateAction):
            retained.update(item.action.ions)
    return frozenset(retained)


def _priority_movement_layer(
    architecture: GridArchitecture,
    state: GridMachineState,
    first_action_id: int,
    priority: Sequence[tuple[int, str]],
    fixed_actions: Sequence[ScheduledAction[Action]],
    retained_ions: frozenset[int],
    *,
    max_states: int,
    allowed_crossings: frozenset[tuple[str, str]] | None,
    selected_gate_requests: frozenset[tuple[int, str]] = frozenset(),
    planning_requests: list[PrioritizedIon] | None = None,
    swap_guard: _SwapGuard | None = None,
) -> tuple[tuple[ScheduledAction[Action], ...], int, frozenset[str]]:
    """Pack next-step transport candidates in global ion-priority order.

    If ``planning_requests`` is a list, the function appends the status of each
    prioritized ion to it. With ``swap_guard``, an ion behind another ion may
    swap with it, see :func:`_greedy_swap`.

    Returns:
        The selected transport layer, total candidate count, and targets whose
        selected gate could not move.
    """

    def record_request(
        ion: int,
        target_segment_id: str,
        status: MoveStatus,
        actions: Sequence[Action] = (),
    ) -> None:
        if planning_requests is not None:
            planning_requests.append(PrioritizedIon(ion, target_segment_id, status, tuple(actions)))

    selected_layer: list[ScheduledAction[Action]] = []
    included_actions: set[Action] = set()
    claimed_by_gates = {
        ion for item in fixed_actions if isinstance(item.action, GateAction) for ion in item.action.ions
    }
    all_moves = junction_moves(architecture, state, allowed_crossings=allowed_crossings)
    available_moves = tuple(move for move in all_moves if not _moves_retained_ion(move, retained_ions))
    ion_priorities = {ion: rank for rank, (ion, _target) in enumerate(priority)}
    ion_to_target_segment = dict(priority)
    explored = 0
    blocked_targets: set[str] = set()
    for ion, target_segment_id in priority:
        if target_segment_id in blocked_targets:
            continue
        if ion in claimed_by_gates:
            record_request(ion, target_segment_id, MoveStatus.IN_GATE)
            continue
        source_segment_id = state.ion_segment(ion)
        if source_segment_id == target_segment_id:
            record_request(ion, target_segment_id, MoveStatus.AT_TARGET)
            continue
        destination_segment_ids = next_segments_toward(
            architecture,
            source_segment_id,
            target_segment_id,
            allowed_crossings,
        )
        if not destination_segment_ids:
            record_request(ion, target_segment_id, MoveStatus.NO_ROUTE)
            continue
        destination_segment_ids = tuple(
            sorted(
                destination_segment_ids,
                # Prefer free capacity when several next segments keep the route shortest.
                key=lambda segment_id: (
                    len(state.occupants(segment_id)) >= architecture.segment(segment_id).capacity,
                    len(state.occupants(segment_id)),
                    segment_id,
                ),
            )
        )
        # First try routes that leave every higher-priority active ion untouched.
        protected_ions = {other_ion for other_ion, rank in ion_priorities.items() if rank < ion_priorities[ion]}
        candidate_moves = tuple(move for move in available_moves if protected_ions.isdisjoint(move.ions))
        candidate_actions: tuple[Action, ...] | None = None
        candidates_explored = 0
        swap_request = _SwapRequest(
            ion,
            frozenset(destination_segment_ids),
            all_moves,
            ion_priorities,
            ion_to_target_segment,
            retained_ions | claimed_by_gates,
        )
        swap = None if swap_guard is None else _greedy_swap(architecture, state, swap_request, swap_guard)
        if swap is not None:
            candidate_actions = (swap,)
            destination_segment_ids = ()
        for destination_segment_id in destination_segment_ids:
            # All equally short choices share the configured routing-work budget.
            candidate_actions, destination_explored = route_junction_crossing_greedy_cycle(
                architecture,
                state,
                ion,
                destination_segment_id,
                max_states=max_states - candidates_explored,
                moves=candidate_moves,
                allowed_crossings=allowed_crossings,
                ion_priorities=ion_priorities,
            )
            candidates_explored += destination_explored
            if candidate_actions is not None or candidates_explored >= max_states:
                break
        explored += candidates_explored
        if not candidate_actions and swap_guard is not None:
            swap = _greedy_swap(architecture, state, swap_request, swap_guard, last_resort=True)
            candidate_actions = None if swap is None else (swap,)
        if not candidate_actions:
            record_request(ion, target_segment_id, MoveStatus.NO_CANDIDATE)
            if (ion, target_segment_id) in selected_gate_requests:
                blocked_targets.add(target_segment_id)
            continue
        if any(action in included_actions for action in candidate_actions):
            record_request(ion, target_segment_id, MoveStatus.ALREADY_SELECTED, candidate_actions)
            continue
        candidate_layer: tuple[ScheduledAction[Action], ...] = tuple(
            ScheduledAction(
                first_action_id + len(selected_layer) + offset,
                action,
                state.time,
                architecture.action_duration(action),
            )
            for offset, action in enumerate(candidate_actions)
        )
        try:
            state_before_candidate = architecture.apply_layer(state, (*fixed_actions, *selected_layer))
            state_after_candidate = architecture.apply_layer(
                state,
                (*fixed_actions, *selected_layer, *candidate_layer),
            )
        except ValueError:
            record_request(ion, target_segment_id, MoveStatus.CONFLICT, candidate_actions)
            continue
        if _harms_higher_priority_ion(
            architecture,
            state_before_candidate,
            state_after_candidate,
            ion,
            ion_priorities,
            ion_to_target_segment,
            allowed_crossings,
        ):
            record_request(ion, target_segment_id, MoveStatus.DISPLACES_HIGHER_PRIORITY, candidate_actions)
            continue
        included_actions.update(candidate_actions)
        selected_layer.extend(candidate_layer)
        record_request(ion, target_segment_id, MoveStatus.SELECTED, candidate_actions)
    return tuple(selected_layer), explored, frozenset(blocked_targets)


@dataclass(frozen=True)
class _SwapRequest:
    """Describe one prioritized ion that asks to cross a junction."""

    ion: int
    destination_segment_ids: frozenset[str]
    moves: tuple[JunctionMove, ...]
    ion_priorities: Mapping[int, int]
    ion_to_target_segment: Mapping[int, str]
    # Ions that a gate keeps in their processing zone or that start a gate in this layer.
    staying_ions: frozenset[int]


@dataclass
class _SwapGuard:
    """Keep the Greedy strategy from swapping one ion pair back and forth.

    A pair may swap again in the same segment only after local progress of the
    pair itself: a completed gate on one of its ions, or a smaller summed
    routing distance of both ions to their targets than at the last swap.
    """

    gates_by_ion: Mapping[int, tuple[int, ...]]
    completed: set[int]
    allowed_crossings: frozenset[tuple[str, str]] | None
    last_swap: dict[tuple[str, frozenset[int]], tuple[int, int]] = field(default_factory=dict)

    def progress(
        self, architecture: GridArchitecture, state: GridMachineState, ions: tuple[int, ...], targets: Mapping[int, str]
    ) -> tuple[int, int]:
        """Return the summed routing distance of ``ions`` to their targets and their completed gates."""
        distance = sum(
            routing_distance(architecture, state.ion_segment(ion), targets[ion], self.allowed_crossings) or 0
            for ion in ions
            if ion in targets
        )
        gates = sum(gate_id in self.completed for ion in ions for gate_id in self.gates_by_ion.get(ion, ()))
        return distance, gates

    def allows(self, swap: Rearrangement, progress: tuple[int, int]) -> bool:
        """Return whether the pair of ``swap`` made local progress since it last swapped in this segment."""
        last = self.last_swap.get((swap.segment_id, frozenset(swap.ions)))
        return last is None or progress[1] > last[1] or progress[0] < last[0]

    def record(self, swap: Rearrangement, progress: tuple[int, int]) -> None:
        """Remember the progress of the pair at this swap."""
        self.last_swap[swap.segment_id, frozenset(swap.ions)] = progress


def _greedy_swap(
    architecture: GridArchitecture,
    state: GridMachineState,
    request: _SwapRequest,
    guard: _SwapGuard,
    *,
    last_resort: bool = False,
) -> Rearrangement | None:
    """Return a swap that brings the requested ion to the exit of its segment, or ``None``.

    The ion must sit behind another ion of a full segment of capacity 2, at an
    exit toward one of the requested destinations. Without ``last_resort``, the
    rule swaps when the ion in front has a target but does not leave through
    the same exit: it stays (target reached, kept for a gate, or higher
    priority), or it would have to come back, which costs two junction moves.
    An ion in front without a target is moved away as before. With
    ``last_resort``, the rule swaps whenever the guard allows it, because no
    other transport was found.
    """
    if not architecture.supports(Rearrangement):
        return None
    ion = request.ion
    source_segment_id = state.ion_segment(ion)
    segment = architecture.segment(source_segment_id)
    occupants = state.occupants(source_segment_id)
    if segment.occupancy is SegmentOccupancy.UNORDERED or segment.capacity != 2 or len(occupants) != 2:
        return None
    toward_destination = [
        move
        for move in request.moves
        if move.source.segment_id == source_segment_id
        and move.destination.segment_id in request.destination_segment_ids
    ]
    # The ion can already leave, or no exit of this segment leads toward a destination.
    if any(move.ions == (ion,) for move in toward_destination) or not any(
        ion in move.ions for move in toward_destination
    ):
        return None
    front = occupants[0] if occupants[1] == ion else occupants[1]
    busy_until = dict(state.ions_busy_until)
    if any(busy_until.get(other, state.time) > state.time for other in occupants):
        return None
    if not last_resort:
        front_target = request.ion_to_target_segment.get(front)
        if front_target is None:
            return None
        front_route = (
            ()
            if front_target == source_segment_id
            else next_segments_toward(architecture, source_segment_id, front_target, guard.allowed_crossings)
        )
        if request.destination_segment_ids.intersection(front_route):
            return None
        stays = (
            front_target == source_segment_id
            or front in request.staying_ions
            or request.ion_priorities[front] < request.ion_priorities[ion]
        )
        timing = architecture.transport_timing
        if not stays and timing.swap > 2 * timing.junction_move:
            return None
    swap = Rearrangement(source_segment_id, occupants, occupants[::-1])
    progress = guard.progress(architecture, state, occupants, request.ion_to_target_segment)
    return swap if guard.allows(swap, progress) else None


def _moves_retained_ion(action: Action, retained_ions: frozenset[int]) -> bool:
    """Return whether a transport action moves a retained gate operand."""
    return isinstance(action, JunctionMove | Cycle) and not retained_ions.isdisjoint(action.ions)


def _build_clearing_layer_for_gate(
    architecture: GridArchitecture,
    state: GridMachineState,
    first_action_id: int,
    priority: Sequence[tuple[int, str]],
    selected_gates: Sequence[GateAction],
    gate_to_segment_lock: Mapping[int, str],
    *,
    blocked_targets: frozenset[str],
    fixed_actions: Sequence[ScheduledAction[Action]],
    max_states: int,
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> tuple[tuple[ScheduledAction[Action], ...], int]:
    """Build a movement layer that clears the path for the highest-priority blocked gate.

    The layer must not move an ion with a higher priority away from its target.

    Returns:
        The path-clearing movement layer and number of segment paths examined.
    """
    ion_ranks = {ion: rank for rank, (ion, _target) in enumerate(priority)}
    for ion, target_segment_id in priority:
        if target_segment_id not in blocked_targets or state.ion_segment(ion) == target_segment_id:
            continue
        focus_gate = next(
            (
                gate
                for gate in selected_gates
                if gate.gate_id is not None
                and ion in gate.ions
                and gate_to_segment_lock.get(gate.gate_id) == target_segment_id
            ),
            None,
        )
        if focus_gate is None:
            continue
        destination_segment_ids = next_segments_toward(
            architecture,
            state.ion_segment(ion),
            target_segment_id,
            allowed_crossings,
        )
        if not destination_segment_ids:
            return (), 0
        destination_segment_id = destination_segment_ids[0]
        retained_ions = frozenset(
            operand for operand in focus_gate.ions if state.ion_segment(operand) == target_segment_id
        )
        available_moves = tuple(
            move
            for move in junction_moves(architecture, state, allowed_crossings=allowed_crossings)
            if not _moves_retained_ion(move, retained_ions)
        )
        focus_priority = tuple((operand, target_segment_id) for operand in focus_gate.ions)
        ion_priorities = {operand: rank for rank, (operand, _target) in enumerate(focus_priority)}
        actions, explored = clear_blocked_path(
            architecture,
            state,
            ion,
            destination_segment_id,
            max_states=max_states,
            moves=available_moves,
            allowed_crossings=allowed_crossings,
            ion_priorities=ion_priorities,
        )
        if actions is None:
            return (), explored
        path_clearing_layer = tuple(
            ScheduledAction[Action](
                first_action_id + offset,
                action,
                state.time,
                architecture.action_duration(action),
            )
            for offset, action in enumerate(actions)
        )
        try:
            state_before_clearing = architecture.apply_layer(state, fixed_actions)
            state_after_clearing = architecture.apply_layer(state, (*fixed_actions, *path_clearing_layer))
        except ValueError:
            return (), explored
        if _harms_higher_priority_ion(
            architecture,
            state_before_clearing,
            state_after_clearing,
            ion,
            ion_ranks,
            dict(priority),
            allowed_crossings,
        ):
            return (), explored
        return path_clearing_layer, explored
    return (), 0


def _harms_higher_priority_ion(
    architecture: GridArchitecture,
    before: GridMachineState,
    after: GridMachineState,
    requesting_ion: int,
    priorities: Mapping[int, int],
    ion_to_target_segment: Mapping[int, str],
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> bool:
    """Return whether candidate actions move a higher-priority ion away from its target."""
    requesting_priority = priorities[requesting_ion]
    for ion, priority in priorities.items():
        if priority >= requesting_priority or before.ion_segment(ion) == after.ion_segment(ion):
            continue
        target_segment_id = ion_to_target_segment[ion]
        distance_before = routing_distance(architecture, before.ion_segment(ion), target_segment_id, allowed_crossings)
        distance_after = routing_distance(architecture, after.ion_segment(ion), target_segment_id, allowed_crossings)
        if distance_before is not None and (distance_after is None or distance_after > distance_before):
            return True
    return False


def _next_resource_release(state: GridMachineState) -> int | None:
    """Return the next time at which a busy Grid resource becomes free."""
    releases = [
        release
        for _resource, release in (
            *state.ions_busy_until,
            *state.junctions_busy_until,
            *state.pzs_busy_until,
        )
        if release > state.time
    ]
    return min(releases, default=None)


def _startable_gate_layer(
    architecture: GridArchitecture,
    state: GridMachineState,
    candidate_gates: Sequence[GateAction],
    first_action_id: int,
    *,
    gate_to_segment_lock: Mapping[int, str],
) -> tuple[ScheduledAction[Action], ...]:
    """Return ready gates that can start together on the current hardware state."""
    selected: list[ScheduledAction[Action]] = []
    for gate in candidate_gates:
        zone_ids: tuple[str | None, ...]
        if architecture.is_virtual_gate(gate):
            zone_ids = (None,)
        else:
            ion_segments = {state.ion_segment(ion) for ion in gate.ions}
            locked_segment_id = gate_to_segment_lock.get(gate.gate_id) if gate.gate_id is not None else None
            zone_ids = tuple(
                zone.zone_id
                for zone in architecture.processing_zones
                if zone.supports(gate)
                and architecture.processing_zone_segment(zone.zone_id).segment_id in ion_segments
                and (
                    locked_segment_id is None
                    or architecture.processing_zone_segment(zone.zone_id).segment_id == locked_segment_id
                )
                and len(ion_segments) == 1
            )
        for zone_id in zone_ids:
            candidate: ScheduledAction[Action] = ScheduledAction(
                first_action_id + len(selected),
                gate,
                state.time,
                architecture.action_duration(gate),
                zone_id,
            )
            try:
                architecture.apply_layer(state, (*selected, candidate))
            except ValueError:
                continue
            selected.append(candidate)
            break
    return tuple(selected)


__all__ = ["schedule_breadth_first", "schedule_greedy"]
