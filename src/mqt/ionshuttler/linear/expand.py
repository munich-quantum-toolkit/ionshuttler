# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate and apply the next transitions of a Linear schedule search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import (
    DEFAULT_ACTION_TYPES,
    Action,
    GateAction,
    PhysicalSwap,
    Shuttle,
    SingleQubitGate,
    TwoQubitGate,
    is_adjacent,
)
from mqt.ionshuttler.linear.state import (
    AdvanceTime,
    SearchTransition,
    State,
    advance_time,
    has_pending_timed_work,
    in_progress_dict,
    to_dict,
)

if TYPE_CHECKING:
    from mqt.ionshuttler.circuit import Circuit
    from mqt.ionshuttler.linear.architecture import LinearArchitecture

Predecessors = Sequence[frozenset[int]]
TransitionCandidate = tuple[SearchTransition, int | None]
ExpandedState = tuple[SearchTransition, int | None, State]


class GenerationMode(StrEnum):
    """Choose how broadly the compiler looks for its next action."""

    FULL = "full"
    INFORMED = "informed"
    UNINFORMED = "uninformed"


@dataclass(frozen=True)
class ExpansionOptions:
    """Configure candidate generation for one expansion.

    ``action_types`` selects the hardware operations the compiler may propose.
    The compiler generates shuttles and physical swaps from the current ion
    placement and takes gates from the circuit.
    """

    mode: GenerationMode = GenerationMode.FULL
    action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES


def ready_gate_ids(
    active_gate_ids: Sequence[int],
    predecessors: Predecessors,
    completed_gates: frozenset[int],
    in_progress_gates: tuple[tuple[int, int], ...],
) -> list[int]:
    """Return gates whose dependencies have finished and that have not started."""
    running = {gate_id for gate_id, _ in in_progress_gates}
    return [
        gate_id
        for gate_id in active_gate_ids
        if gate_id not in completed_gates and gate_id not in running and predecessors[gate_id].issubset(completed_gates)
    ]


def generate_actions(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    *,
    active_gate_ids: Sequence[int] | None = None,
    predecessors: Predecessors | None = None,
    action_types: Sequence[type[Action]] | None = None,
) -> list[SearchTransition]:
    """Return every transition that can start in the current state."""
    options = ExpansionOptions(
        action_types=DEFAULT_ACTION_TYPES if action_types is None else tuple(action_types),
    )
    return generate_actions_by_mode(
        state,
        architecture,
        circuit,
        active_gate_ids=active_gate_ids,
        predecessors=predecessors,
        options=options,
    )


def generate_actions_by_mode(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    *,
    active_gate_ids: Sequence[int] | None = None,
    predecessors: Predecessors | None = None,
    options: ExpansionOptions | None = None,
) -> list[SearchTransition]:
    """Return valid transitions from the selected candidate group."""
    return [
        transition
        for transition, _ in _valid_candidates(
            state,
            architecture,
            circuit,
            *_circuit_view(circuit, active_gate_ids, predecessors),
            options=options or ExpansionOptions(),
        )
    ]


def apply(
    state: State,
    architecture: LinearArchitecture,
    transition: SearchTransition,
    *,
    gate_id: int | None = None,
) -> State:
    """Apply one transition and record the selected circuit gate, if any.

    Args:
        state: State before the transition.
        architecture: Hardware on which an action runs.
        transition: Hardware action to start, or a time advance.
        gate_id: Circuit gate represented by a gate action.

    Returns:
        The updated hardware state and circuit progress.

    Raises:
        ValueError: If a gate action has no gate identifier.
    """
    if isinstance(transition, AdvanceTime):
        return advance_time(state)
    action = transition
    updated = architecture.apply_action(state, action)
    if not isinstance(action, GateAction):
        return updated
    if gate_id is None:
        msg = "gate_id is required when applying a gate action"
        raise ValueError(msg)

    duration = architecture.action_duration(action)
    if isinstance(action, SingleQubitGate) and (architecture.is_virtual_gate(action) or duration == 0):
        return replace(updated, completed_gates=updated.completed_gates | {gate_id})
    in_progress = in_progress_dict(updated)
    in_progress[gate_id] = updated.time + duration
    return replace(updated, in_progress_gates=tuple(in_progress.items()))


def expand(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    *,
    active_gate_ids: Sequence[int] | None = None,
    predecessors: Predecessors | None = None,
    options: ExpansionOptions | None = None,
) -> list[ExpandedState]:
    """Return each valid next transition with the state it produces."""
    candidates = _valid_candidates(
        state,
        architecture,
        circuit,
        *_circuit_view(circuit, active_gate_ids, predecessors),
        options=options or ExpansionOptions(),
    )
    return [
        (transition, gate_id, apply(state, architecture, transition, gate_id=gate_id))
        for transition, gate_id in candidates
    ]


def replay_path(
    initial_state: State,
    architecture: LinearArchitecture,
    path: Sequence[SearchTransition],
    circuit: Circuit,
    *,
    active_gate_ids: Sequence[int] | None = None,
    predecessors: Predecessors | None = None,
) -> State:
    """Replay a search path while checking every action before it starts.

    Returns:
        The state reached after the final transition.

    Raises:
        ValueError: If an action is unavailable or a gate action does not
            identify a ready circuit gate by its ``gate_id``.
    """
    state = initial_state
    gate_ids, effective_predecessors = _circuit_view(circuit, active_gate_ids, predecessors)
    for transition in path:
        if not _is_transition_valid(state, transition, architecture):
            msg = f"action {transition!r} is not valid at time {state.time}"
            raise ValueError(msg)
        gate_id = (
            _resolve_gate_id(transition, state, circuit, gate_ids, effective_predecessors)
            if isinstance(transition, GateAction)
            else None
        )
        state = apply(state, architecture, transition, gate_id=gate_id)
    return state


def _is_transition_valid(state: State, transition: SearchTransition, architecture: LinearArchitecture) -> bool:
    """Return whether a transition can start; waiting is always possible."""
    return isinstance(transition, AdvanceTime) or architecture.is_action_valid(state, transition)


def _valid_candidates(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    active_gate_ids: Sequence[int],
    predecessors: Predecessors,
    *,
    options: ExpansionOptions,
) -> list[TransitionCandidate]:
    """Return candidate transitions whose individual hardware requirements hold."""
    candidates = _candidate_actions(
        state,
        architecture,
        circuit,
        active_gate_ids,
        predecessors=predecessors,
        options=options,
    )
    return [candidate for candidate in candidates if _is_transition_valid(state, candidate[0], architecture)]


def _candidate_actions(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    active_gate_ids: Sequence[int],
    *,
    predecessors: Predecessors,
    options: ExpansionOptions,
) -> list[TransitionCandidate]:
    """Return the candidates selected by the configured generation mode.

    Raises:
        ValueError: If the generation mode is unsupported.
    """
    if options.mode is GenerationMode.FULL:
        return _all_candidates(
            state,
            architecture,
            circuit,
            active_gate_ids,
            predecessors=predecessors,
            action_types=options.action_types,
        )
    if options.mode is GenerationMode.INFORMED:
        return _informed_candidates(
            state,
            architecture,
            circuit,
            active_gate_ids,
            predecessors=predecessors,
            action_types=options.action_types,
        )
    if options.mode is GenerationMode.UNINFORMED:
        full = _valid_candidates(
            state,
            architecture,
            circuit,
            active_gate_ids,
            predecessors,
            options=ExpansionOptions(mode=GenerationMode.FULL, action_types=options.action_types),
        )
        informed = _valid_candidates(
            state,
            architecture,
            circuit,
            active_gate_ids,
            predecessors,
            options=ExpansionOptions(mode=GenerationMode.INFORMED, action_types=options.action_types),
        )
        return [candidate for candidate in full if candidate not in informed]
    msg = f"unsupported generation mode: {options.mode}"
    raise ValueError(msg)


def _all_candidates(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    active_gate_ids: Sequence[int],
    *,
    predecessors: Predecessors,
    action_types: tuple[type[Action], ...],
) -> list[TransitionCandidate]:
    """Return all ready gates and locally available transport actions."""
    candidates: list[TransitionCandidate] = []
    busy_ions = {ion for ion, free_time in state.ions_busy_until if free_time > state.time}
    for gate_id in ready_gate_ids(
        active_gate_ids,
        predecessors,
        state.completed_gates,
        state.in_progress_gates,
    ):
        gate = circuit.gates[gate_id]
        if type(gate) in action_types and _gate_ions_are_free(gate, busy_ions, architecture):
            candidates.append((gate, gate_id))

    # Transport candidates follow the order of the selected action types.
    for action_type in action_types:
        if action_type is Shuttle:
            candidates.extend((shuttle, None) for shuttle in _shuttle_candidates(state, architecture, busy_ions))
        elif action_type is PhysicalSwap:
            candidates.extend((swap, None) for swap in _swap_candidates(state, busy_ions))

    if has_pending_timed_work(state):
        candidates.append((AdvanceTime(), None))
    return candidates


def _shuttle_candidates(state: State, architecture: LinearArchitecture, busy_ions: set[int]) -> list[Shuttle]:
    """Return shuttles of free ions to adjacent empty sites."""
    occupied = {position for _, position in state.positions}
    return [
        Shuttle(ion=ion, src=position, dst=destination)
        for ion, position in state.positions
        if ion not in busy_ions
        for destination in (position - 1, position + 1)
        if 0 <= destination < architecture.num_sites and destination not in occupied
    ]


def _swap_candidates(state: State, busy_ions: set[int]) -> list[PhysicalSwap]:
    """Return physical swaps of adjacent free ions."""
    free_ions = [(ion, position) for ion, position in state.positions if ion not in busy_ions]
    return [
        PhysicalSwap(ion_a=ion_a, ion_b=ion_b, pos_a=pos_a, pos_b=pos_b)
        for index, (ion_a, pos_a) in enumerate(free_ions)
        for ion_b, pos_b in free_ions[index + 1 :]
        if is_adjacent(pos_a, pos_b)
    ]


def _informed_candidates(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    active_gate_ids: Sequence[int],
    *,
    predecessors: Predecessors,
    action_types: tuple[type[Action], ...],
) -> list[TransitionCandidate]:
    """Return ready gates or transports that move relevant ions toward a zone."""
    ready: list[TransitionCandidate] = [
        (circuit.gates[gate_id], gate_id)
        for gate_id in ready_gate_ids(
            active_gate_ids,
            predecessors,
            state.completed_gates,
            state.in_progress_gates,
        )
        if type(circuit.gates[gate_id]) in action_types
    ]
    valid_ready = [candidate for candidate in ready if _is_transition_valid(state, candidate[0], architecture)]
    if valid_ready:
        return valid_ready
    return [
        candidate
        for candidate in _good_moves(
            state,
            architecture,
            circuit,
            active_gate_ids,
            predecessors=predecessors,
        )
        if type(candidate[0]) in action_types
    ]


def _good_moves(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    active_gate_ids: Sequence[int],
    *,
    predecessors: Predecessors,
) -> list[TransitionCandidate]:
    """Return transports that reduce processing-zone distance without regressions."""
    positions = to_dict(state)
    occupied = set(positions.values())
    busy_ions = {ion for ion, free_time in state.ions_busy_until if free_time > state.time}
    considered_ions = _considered_ions(
        state,
        architecture,
        circuit,
        active_gate_ids,
        predecessors=predecessors,
    )
    if not considered_ions:
        return []

    zone_sites = tuple(site for sites in (architecture.processing_zones or {}).values() for site in sites)
    moves: list[TransitionCandidate] = []
    for ion, position in state.positions:
        if ion in busy_ions or ion not in considered_ions or architecture.get_processing_zone(position) is not None:
            continue
        distance = _distance_to_processing_zone(position, zone_sites)
        if distance is None:
            continue
        for delta in (-1, 1):
            destination = position + delta
            if not 0 <= destination < architecture.num_sites:
                continue
            next_distance = _distance_to_processing_zone(destination, zone_sites)
            if destination not in occupied and next_distance is not None and next_distance < distance:
                moves.append((Shuttle(ion=ion, src=position, dst=destination), None))

    free_ions = [(ion, position) for ion, position in state.positions if ion not in busy_ions]
    for index, (ion_a, pos_a) in enumerate(free_ions):
        for ion_b, pos_b in free_ions[index + 1 :]:
            if abs(pos_a - pos_b) == 1 and _is_good_swap(
                ion_a,
                pos_a,
                ion_b,
                pos_b,
                considered_ions,
                architecture,
                zone_sites,
            ):
                moves.append((PhysicalSwap(ion_a=ion_a, ion_b=ion_b, pos_a=pos_a, pos_b=pos_b), None))
    return moves


def _considered_ions(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    active_gate_ids: Sequence[int],
    *,
    predecessors: Predecessors,
) -> set[int]:
    """Return ions needed within the processing-zone-count dependency horizon."""
    running = {gate_id for gate_id, _ in state.in_progress_gates}
    gate_horizon = len(architecture.processing_zones or {})
    ions: set[int] = set()
    for gate_id in active_gate_ids:
        if gate_id in state.completed_gates or gate_id in running:
            continue
        remaining_predecessors = predecessors[gate_id].difference(state.completed_gates)
        if len(remaining_predecessors) > gate_horizon:
            continue
        gate = circuit.gates[gate_id]
        if isinstance(gate, SingleQubitGate):
            ions.add(gate.ion)
        elif isinstance(gate, TwoQubitGate):
            ions.update({gate.ion_a, gate.ion_b})
    return ions


def _distance_to_processing_zone(position: int, zone_sites: tuple[int, ...]) -> int | None:
    """Return the nearest processing-zone distance, or ``None`` if no zone exists."""
    return min((abs(position - site) for site in zone_sites), default=None)


def _is_good_swap(
    ion_a: int,
    pos_a: int,
    ion_b: int,
    pos_b: int,
    considered_ions: set[int],
    architecture: LinearArchitecture,
    zone_sites: tuple[int, ...],
) -> bool:
    """Return whether a swap helps a considered ion without moving one farther away."""
    improvements = 0
    for ion, current_position, next_position in (
        (ion_a, pos_a, pos_b),
        (ion_b, pos_b, pos_a),
    ):
        if ion not in considered_ions:
            continue
        if architecture.get_processing_zone(current_position) is not None:
            continue
        current_distance = _distance_to_processing_zone(current_position, zone_sites)
        next_distance = _distance_to_processing_zone(next_position, zone_sites)
        if current_distance is None or next_distance is None:
            continue
        if next_distance > current_distance:
            return False
        if next_distance < current_distance:
            improvements += 1
    return improvements > 0


def _gate_ions_are_free(gate: GateAction, busy_ions: set[int], architecture: LinearArchitecture) -> bool:
    """Return whether every physical participant of a gate is available."""
    if isinstance(gate, SingleQubitGate):
        return architecture.is_virtual_gate(gate) or gate.ion not in busy_ions
    if isinstance(gate, TwoQubitGate):
        return gate.ion_a not in busy_ions and gate.ion_b not in busy_ions
    return True


def _circuit_view(
    circuit: Circuit,
    active_gate_ids: Sequence[int] | None,
    predecessors: Predecessors | None,
) -> tuple[Sequence[int], Predecessors]:
    """Resolve optional circuit views without copying gate data.

    Returns:
        The active gate IDs and effective predecessor sequence.
    """
    return (
        circuit.gate_ids if active_gate_ids is None else active_gate_ids,
        circuit.predecessors if predecessors is None else predecessors,
    )


def _resolve_gate_id(
    action: GateAction,
    state: State,
    circuit: Circuit,
    active_gate_ids: Sequence[int],
    predecessors: Predecessors,
) -> int:
    """Return the circuit-gate ID of a replayed gate action.

    The action's ``gate_id`` identifies the circuit gate. Gate equality only
    checks that the action executes that gate.

    Raises:
        ValueError: If the action has no gate ID, differs from its circuit
            gate, or its circuit gate is not ready.
    """
    gate_id = action.gate_id
    if gate_id is None:
        msg = f"gate action {action!r} has no gate_id"
        raise ValueError(msg)
    if gate_id >= len(circuit.gates) or circuit.gates[gate_id] != action:
        msg = f"gate action {action!r} does not match circuit gate {gate_id}"
        raise ValueError(msg)
    ready = ready_gate_ids(
        active_gate_ids,
        predecessors,
        state.completed_gates,
        state.in_progress_gates,
    )
    if gate_id not in ready:
        msg = f"gate action {action!r} is not ready"
        raise ValueError(msg)
    return gate_id


__all__ = [
    "ExpansionOptions",
    "GenerationMode",
    "apply",
    "expand",
    "generate_actions",
    "generate_actions_by_mode",
    "ready_gate_ids",
    "replay_path",
]
