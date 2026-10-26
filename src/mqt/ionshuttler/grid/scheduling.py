# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Scheduling strategies for Grid compilation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.grid.actions import Cycle, JunctionMove
from mqt.ionshuttler.grid.planning_trace import IdleReason, MoveStatus, PlannedGate, PlanningSnapshot, PrioritizedIon
from mqt.ionshuttler.grid.result import GridDiagnostics, GridStopReason
from mqt.ionshuttler.grid.routing import (
    clear_blocked_path,
    junction_moves,
    next_segment_toward,
    route_gate_bfs,
    route_junction_crossing_greedy_cycle,
    route_placement_bfs,
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
    iterations = explored_states = movement_count = cycle_count = 0
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
            movement_count, cycle_count = _count_transport(action, movement_count, cycle_count)
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
    has_target = target_state is not None or partial_target_by_ion is not None
    if (
        has_target
        and gates_complete
        and not _placement_reached(state, target_state, partial_target_by_ion)
        and iterations < config.max_iterations
    ):
        iterations += 1
        decision_time = state.time
        found_route, explored = route_placement_bfs(
            architecture,
            state,
            lambda candidate: _placement_reached(candidate, target_state, partial_target_by_ion),
            max_states=config.max_routing_states,
            allowed_crossings=config.allowed_junction_crossings,
        )
        explored_states += explored
        route = found_route or ()
        for action in route:
            timed_action = ScheduledAction(
                len(scheduled_actions),
                action,
                state.time,
                architecture.action_duration(action),
            )
            state = architecture.apply_layer(state, (timed_action,)).at_time(timed_action.end_time)
            scheduled_actions.append(timed_action)
            movement_count, cycle_count = _count_transport(action, movement_count, cycle_count)
        if planning_trace is not None:
            planning_trace.append(
                _planning_snapshot(
                    iterations,
                    decision_time,
                    (),
                    (),
                    gate_to_target_segment={},
                    moves=route,
                    idle_reason=None if route else IdleReason.NO_ROUTE,
                )
            )
        if not route:
            stalled = True

    completed_objective = gates_complete and _placement_reached(state, target_state, partial_target_by_ion)

    return (
        Schedule(tuple(scheduled_actions), state.time, initial_state),
        GridDiagnostics(
            iterations=iterations,
            explored_routing_states=explored_states,
            junction_moves=movement_count,
            cycles=cycle_count,
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
    iterations = explored_states = movement_count = cycle_count = 0
    planning_trace: list[PlanningSnapshot] | None = [] if config.record_planning_trace else None
    stalled = False
    stall_clear_attempted_since_gate_completion = False

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
        gate_layer = _startable_gate_layer(
            architecture,
            state,
            plan.selected_ready_gates,
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
        state = architecture.apply_layer(state, action_layer)
        scheduled_actions.extend(action_layer)
        for movement_item in movement_layer:
            movement_count, cycle_count = _count_transport(
                movement_item.action,
                movement_count,
                cycle_count,
            )
        if any(item.end_time > state.time for item in action_layer):
            state = state.at_time(state.time + 1)

    if scheduled_actions:
        state = state.at_time(max(state.time, *(action.end_time for action in scheduled_actions)))
        _complete_started_gates(state.time, started, completed, gate_to_segment_lock)

    gates_complete = len(completed) == len(circuit.gates)
    seen_placements = {state.occupancy}
    while (
        (target_state is not None or partial_target_by_ion is not None)
        and gates_complete
        and not _placement_reached(state, target_state, partial_target_by_ion)
        and iterations < config.max_iterations
    ):
        iterations += 1
        priority = _target_placement_priority(state, target_state, partial_target_by_ion)
        planning_requests: list[PrioritizedIon] | None = None if planning_trace is None else []
        movement_layer, explored, _blocked_targets = _priority_movement_layer(
            architecture,
            state,
            len(scheduled_actions),
            priority,
            (),
            frozenset(),
            max_states=config.max_routing_states,
            allowed_crossings=config.allowed_junction_crossings,
            planning_requests=planning_requests,
        )
        explored_states += explored
        next_time = None if movement_layer else _next_resource_release(state)
        if planning_trace is not None:
            planning_trace.append(
                _planning_snapshot(
                    iterations,
                    state.time,
                    (),
                    (),
                    gate_to_target_segment={},
                    prioritized_ions=planning_requests or (),
                    moves=tuple(item.action for item in movement_layer),
                    idle_reason=_idle_reason(movement_layer, next_time),
                )
            )
        if not movement_layer:
            if next_time is None:
                stalled = True
                break
            state = state.at_time(next_time)
            continue
        following = architecture.apply_layer(state, movement_layer)
        repeated_placement = following.occupancy in seen_placements
        if not repeated_placement:
            seen_placements.add(following.occupancy)
        state = following
        scheduled_actions.extend(movement_layer)
        for movement_item in movement_layer:
            movement_count, cycle_count = _count_transport(
                movement_item.action,
                movement_count,
                cycle_count,
            )
        if any(item.end_time > state.time for item in movement_layer):
            state = state.at_time(state.time + 1)
        if repeated_placement:
            stalled = True
            break

    if scheduled_actions:
        state = state.at_time(max(state.time, *(action.end_time for action in scheduled_actions)))

    completed_objective = gates_complete and _placement_reached(state, target_state, partial_target_by_ion)

    return (
        Schedule(tuple(scheduled_actions), state.time, initial_state),
        GridDiagnostics(
            iterations=iterations,
            explored_routing_states=explored_states,
            junction_moves=movement_count,
            cycles=cycle_count,
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


def _target_placement_priority(
    state: GridMachineState,
    target_state: GridMachineState | None,
    partial_target_by_ion: Mapping[int, str] | None,
) -> tuple[tuple[int, str], ...]:
    """Return misplaced ions and their target segments in ion order.

    Returns:
        The deterministic greedy transport priority.
    """
    if partial_target_by_ion is not None:
        target_by_ion = dict(partial_target_by_ion)
    else:
        assert target_state is not None
        target_by_ion = {ion: segment_id for segment_id, occupants in target_state.occupancy for ion in occupants}
    return tuple(
        (ion, segment_id) for ion, segment_id in sorted(target_by_ion.items()) if state.ion_segment(ion) != segment_id
    )


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


def _count_transport(action: Action, moves: int, cycles: int) -> tuple[int, int]:
    """Add one transport action to the diagnostic counters.

    Returns:
        The updated junction-move and cycle counters.
    """
    if isinstance(action, Cycle):
        return moves + len(action.moves), cycles + 1
    if isinstance(action, JunctionMove):
        return moves + 1, cycles
    return moves, cycles


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
        virtual = tuple(gate for gate in ready if architecture.is_virtual_gate(gate))
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
) -> tuple[tuple[ScheduledAction[Action], ...], int, frozenset[str]]:
    """Pack next-step transport candidates in global ion-priority order.

    If ``planning_requests`` is a list, the function appends the status of each
    prioritized ion to it.

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
    available_moves = tuple(
        move
        for move in junction_moves(architecture, state, allowed_crossings=allowed_crossings)
        if not _moves_retained_ion(move, retained_ions)
    )
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
        destination_segment_id = next_segment_toward(
            architecture,
            source_segment_id,
            target_segment_id,
            allowed_crossings,
        )
        if destination_segment_id is None:
            record_request(ion, target_segment_id, MoveStatus.NO_ROUTE)
            continue
        candidate_actions, candidates_explored = route_junction_crossing_greedy_cycle(
            architecture,
            state,
            ion,
            destination_segment_id,
            max_states=max_states,
            moves=available_moves,
            allowed_crossings=allowed_crossings,
            ion_priorities=ion_priorities,
        )
        explored += candidates_explored
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
        destination_segment_id = next_segment_toward(
            architecture,
            state.ion_segment(ion),
            target_segment_id,
            allowed_crossings,
        )
        if destination_segment_id is None:
            return (), 0
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
