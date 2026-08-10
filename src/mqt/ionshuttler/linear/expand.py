# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate and apply the next actions of a Linear schedule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import (
    Action,
    AdvanceTime,
    GateAction,
    PhysicalSwap,
    Shuttle,
    SingleQubitGate,
    TwoQubitGate,
)
from mqt.ionshuttler.linear.config import TransportTiming
from mqt.ionshuttler.linear.state import (
    State,
    has_pending_timed_work,
    in_progress_dict,
    to_dict,
)
from mqt.ionshuttler.linear.validation import is_action_valid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.linear.architecture import Architecture

GateMap = Mapping[int, GateAction]
PredecessorMap = Mapping[int, frozenset[int]]
ActionCandidate = tuple[Action, int | None]
ExpandedState = tuple[Action, int | None, State]


class GenerationMode(str, Enum):
    """Choose how broadly the compiler looks for its next action."""

    FULL = "full"
    INFORMED = "informed"
    UNINFORMED = "uninformed"


@dataclass(frozen=True)
class ExpansionOptions:
    """Configure candidate generation for one expansion."""

    mode: GenerationMode = GenerationMode.FULL
    transport_timing: TransportTiming = field(default_factory=TransportTiming)


def ready_gate_ids(
    gate_order: Sequence[int],
    predecessors: PredecessorMap,
    completed_gates: frozenset[int],
    in_progress_gates: tuple[tuple[int, int], ...],
) -> list[int]:
    """Return gates whose dependencies have finished and that have not started."""
    running = {gate_id for gate_id, _ in in_progress_gates}
    return [
        gate_id
        for gate_id in gate_order
        if gate_id not in completed_gates
        and gate_id not in running
        and predecessors.get(gate_id, frozenset()).issubset(completed_gates)
    ]


def generate_actions(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None = None,
    transport_timing: TransportTiming | None = None,
) -> list[Action]:
    """Return every action that can start in the current state."""
    options = ExpansionOptions(transport_timing=transport_timing or TransportTiming())
    return generate_actions_by_mode(
        state,
        architecture,
        gate_order,
        gates,
        predecessors=predecessors,
        options=options,
    )


def generate_actions_by_mode(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None = None,
    options: ExpansionOptions | None = None,
) -> list[Action]:
    """Return valid actions from the selected candidate group."""
    return [
        action
        for action, _ in _valid_candidates(
            state,
            architecture,
            gate_order,
            gates,
            predecessors=predecessors,
            options=options or ExpansionOptions(),
        )
    ]


def apply(
    state: State,
    architecture: Architecture,
    action: Action,
    *,
    gate_id: int | None = None,
) -> State:
    """Apply one action and record the selected circuit gate, if any.

    Args:
        state: State before the action starts.
        architecture: Hardware on which the action runs.
        action: Action to start.
        gate_id: Circuit gate represented by a gate action.

    Returns:
        The updated hardware state and circuit progress.

    Raises:
        ValueError: If a gate action has no gate identifier.
    """
    updated = action.apply(state, architecture)
    if not isinstance(action, GateAction):
        return updated
    if gate_id is None:
        msg = "gate_id is required when applying a gate action"
        raise ValueError(msg)

    duration = _gate_duration(action)
    if isinstance(action, SingleQubitGate) and (action.virtual or duration == 0):
        return replace(updated, completed_gates=updated.completed_gates | {gate_id})
    in_progress = in_progress_dict(updated)
    in_progress[gate_id] = updated.time + duration
    return replace(updated, in_progress_gates=tuple(sorted(in_progress.items())))


def expand(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None = None,
    options: ExpansionOptions | None = None,
) -> list[ExpandedState]:
    """Return each valid next action with the state it produces."""
    candidates = _valid_candidates(
        state,
        architecture,
        gate_order,
        gates,
        predecessors=predecessors,
        options=options or ExpansionOptions(),
    )
    return [(action, gate_id, apply(state, architecture, action, gate_id=gate_id)) for action, gate_id in candidates]


def replay_path(
    initial_state: State,
    architecture: Architecture,
    path: Sequence[Action],
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None = None,
) -> State:
    """Replay a schedule while checking every action before it starts.

    Returns:
        The state reached after the final action.

    Raises:
        ValueError: If an action is unavailable or cannot be matched to a ready gate.
    """
    state = initial_state
    predecessor_map = _normalize_predecessors(gate_order, predecessors)
    for action in path:
        if not is_action_valid(state, action, architecture):
            msg = f"action {action!r} is not valid at time {state.time}"
            raise ValueError(msg)
        gate_id = (
            _resolve_gate_id(action, state, gate_order, gates, predecessor_map)
            if isinstance(action, GateAction)
            else None
        )
        state = apply(state, architecture, action, gate_id=gate_id)
    return state


def _valid_candidates(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None,
    options: ExpansionOptions,
) -> list[ActionCandidate]:
    candidates = _candidate_actions(
        state,
        architecture,
        gate_order,
        gates,
        predecessors=predecessors,
        options=options,
    )
    return [candidate for candidate in candidates if is_action_valid(state, candidate[0], architecture)]


def _candidate_actions(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None,
    options: ExpansionOptions,
) -> list[ActionCandidate]:
    if options.mode is GenerationMode.FULL:
        return _all_candidates(
            state,
            architecture,
            gate_order,
            gates,
            predecessors=predecessors,
            transport_timing=options.transport_timing,
        )
    if options.mode is GenerationMode.INFORMED:
        return _informed_candidates(
            state,
            architecture,
            gate_order,
            gates,
            predecessors=predecessors,
            transport_timing=options.transport_timing,
        )
    if options.mode is GenerationMode.UNINFORMED:
        full = _valid_candidates(
            state,
            architecture,
            gate_order,
            gates,
            predecessors=predecessors,
            options=ExpansionOptions(
                mode=GenerationMode.FULL,
                transport_timing=options.transport_timing,
            ),
        )
        informed = _valid_candidates(
            state,
            architecture,
            gate_order,
            gates,
            predecessors=predecessors,
            options=ExpansionOptions(
                mode=GenerationMode.INFORMED,
                transport_timing=options.transport_timing,
            ),
        )
        return [candidate for candidate in full if candidate not in informed]
    msg = f"unsupported generation mode: {options.mode}"
    raise ValueError(msg)


def _all_candidates(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None,
    transport_timing: TransportTiming,
) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    positions = to_dict(state)
    occupied = set(positions.values())
    busy_ions = {ion for ion, free_time in state.ions_busy_until if free_time > state.time}
    predecessor_map = _normalize_predecessors(gate_order, predecessors)

    for gate_id in ready_gate_ids(
        gate_order,
        predecessor_map,
        state.completed_gates,
        state.in_progress_gates,
    ):
        gate = gates[gate_id]
        if _gate_ions_are_free(gate, busy_ions):
            candidates.append((gate, gate_id))

    free_ions = [(ion, position) for ion, position in state.positions if ion not in busy_ions]
    for index, (ion_a, pos_a) in enumerate(free_ions):
        for ion_b, pos_b in free_ions[index + 1 :]:
            if abs(pos_a - pos_b) == 1:
                candidates.append((
                    PhysicalSwap(
                        ion_a=ion_a,
                        ion_b=ion_b,
                        pos_a=pos_a,
                        pos_b=pos_b,
                        duration=transport_timing.swap,
                    ),
                    None,
                ))

    for ion, position in state.positions:
        if ion in busy_ions:
            continue
        for delta in (-1, 1):
            destination = position + delta
            if 0 <= destination < architecture.num_sites and destination not in occupied:
                candidates.append((
                    Shuttle(
                        ion=ion,
                        src=position,
                        dst=destination,
                        duration=transport_timing.shuttle,
                    ),
                    None,
                ))

    if has_pending_timed_work(state):
        candidates.append((AdvanceTime(), None))
    return candidates


def _informed_candidates(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None,
    transport_timing: TransportTiming,
) -> list[ActionCandidate]:
    predecessor_map = _normalize_predecessors(gate_order, predecessors)
    ready: list[ActionCandidate] = [
        (gates[gate_id], gate_id)
        for gate_id in ready_gate_ids(
            gate_order,
            predecessor_map,
            state.completed_gates,
            state.in_progress_gates,
        )
    ]
    valid_ready = [candidate for candidate in ready if is_action_valid(state, candidate[0], architecture)]
    if valid_ready:
        return valid_ready
    return _good_moves(
        state,
        architecture,
        gate_order,
        gates,
        predecessors=predecessors,
        transport_timing=transport_timing,
    )


def _good_moves(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None,
    transport_timing: TransportTiming,
) -> list[ActionCandidate]:
    positions = to_dict(state)
    occupied = set(positions.values())
    busy_ions = {ion for ion, free_time in state.ions_busy_until if free_time > state.time}
    considered_ions = _considered_ions(
        state,
        architecture,
        gate_order,
        gates,
        predecessors=predecessors,
    )
    if not considered_ions:
        return []

    moves: list[ActionCandidate] = []
    for ion, position in state.positions:
        if ion in busy_ions or ion not in considered_ions or architecture.get_processing_zone(position) is not None:
            continue
        distance = _distance_to_processing_zone(position, architecture)
        if distance is None:
            continue
        for delta in (-1, 1):
            destination = position + delta
            next_distance = _distance_to_processing_zone(destination, architecture)
            if (
                0 <= destination < architecture.num_sites
                and destination not in occupied
                and next_distance is not None
                and next_distance < distance
            ):
                moves.append((
                    Shuttle(
                        ion=ion,
                        src=position,
                        dst=destination,
                        duration=transport_timing.shuttle,
                    ),
                    None,
                ))

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
            ):
                moves.append((
                    PhysicalSwap(
                        ion_a=ion_a,
                        ion_b=ion_b,
                        pos_a=pos_a,
                        pos_b=pos_b,
                        duration=transport_timing.swap,
                    ),
                    None,
                ))
    return moves


def _considered_ions(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: GateMap,
    *,
    predecessors: PredecessorMap | None,
) -> set[int]:
    predecessor_map = _normalize_predecessors(gate_order, predecessors)
    running = {gate_id for gate_id, _ in state.in_progress_gates}
    gate_horizon = len(architecture.processing_zones or {})
    ions: set[int] = set()
    for gate_id in gate_order:
        if gate_id in state.completed_gates or gate_id in running:
            continue
        remaining_predecessors = predecessor_map.get(gate_id, frozenset()).difference(state.completed_gates)
        if len(remaining_predecessors) > gate_horizon:
            continue
        gate = gates[gate_id]
        if isinstance(gate, SingleQubitGate):
            ions.add(gate.ion)
        elif isinstance(gate, TwoQubitGate):
            ions.update({gate.ion_a, gate.ion_b})
    return ions


def _distance_to_processing_zone(position: int, architecture: Architecture) -> int | None:
    zone_sites = [site for sites in (architecture.processing_zones or {}).values() for site in sites]
    return min((abs(position - site) for site in zone_sites), default=None)


def _is_good_swap(
    ion_a: int,
    pos_a: int,
    ion_b: int,
    pos_b: int,
    considered_ions: set[int],
    architecture: Architecture,
) -> bool:
    improvements = 0
    for ion, current_position, next_position in (
        (ion_a, pos_a, pos_b),
        (ion_b, pos_b, pos_a),
    ):
        if ion not in considered_ions:
            continue
        if architecture.get_processing_zone(current_position) is not None:
            continue
        current_distance = _distance_to_processing_zone(current_position, architecture)
        next_distance = _distance_to_processing_zone(next_position, architecture)
        if current_distance is None or next_distance is None:
            continue
        if next_distance > current_distance:
            return False
        if next_distance < current_distance:
            improvements += 1
    return improvements > 0


def _gate_ions_are_free(gate: GateAction, busy_ions: set[int]) -> bool:
    if isinstance(gate, SingleQubitGate):
        return gate.virtual or gate.ion not in busy_ions
    if isinstance(gate, TwoQubitGate):
        return gate.ion_a not in busy_ions and gate.ion_b not in busy_ions
    return True


def _gate_duration(gate: GateAction) -> int:
    duration = vars(gate).get("duration")
    if isinstance(duration, bool) or not isinstance(duration, int):
        msg = "gate action does not define an integer duration"
        raise TypeError(msg)
    return duration


def _normalize_predecessors(
    gate_order: Sequence[int],
    predecessors: PredecessorMap | None,
) -> dict[int, frozenset[int]]:
    if predecessors is not None:
        return {gate_id: predecessors.get(gate_id, frozenset()) for gate_id in gate_order}
    return {gate_id: frozenset(gate_order[index - 1 : index]) for index, gate_id in enumerate(gate_order)}


def _resolve_gate_id(
    action: GateAction,
    state: State,
    gate_order: Sequence[int],
    gates: GateMap,
    predecessors: PredecessorMap,
) -> int:
    ready = ready_gate_ids(
        gate_order,
        predecessors,
        state.completed_gates,
        state.in_progress_gates,
    )
    identity_matches = [gate_id for gate_id in ready if gates[gate_id] is action]
    if len(identity_matches) == 1:
        return identity_matches[0]
    equality_matches = [gate_id for gate_id in ready if gates[gate_id] == action]
    if len(equality_matches) == 1:
        return equality_matches[0]
    msg = f"gate action {action!r} does not identify exactly one ready gate"
    raise ValueError(msg)


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
