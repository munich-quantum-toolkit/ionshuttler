# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Cost estimates used to choose promising schedules, not admissible."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import GateAction, SingleQubitGate, TwoQubitGate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.state import State


def cost(state: State) -> int:
    """Return the cost, currently simply the elapsed schedule time."""
    return state.time


def min_distance_to_valid_pair(
    pos_a: int,
    pos_b: int,
    valid_pairs: tuple[tuple[int, int], ...],
) -> int:
    """Return the fewest simultaneous site moves needed to reach a valid pair."""
    if not valid_pairs:
        return 0
    return min(
        min(
            max(abs(pos_a - left), abs(pos_b - right)),
            max(abs(pos_a - right), abs(pos_b - left)),
        )
        for left, right in valid_pairs
    )


def heuristic(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: Mapping[int, GateAction],
    predecessors: Mapping[int, frozenset[int]] | None = None,
) -> int:
    """Estimate the work still needed to finish the requested gates.

    Movement and gate execution can overlap, so this estimate may overstate
    the remaining schedule time and does not guarantee an optimal result.

    Returns:
        A nonnegative estimate combining ion movement and remaining gate depth.
    """
    running = {gate_id for gate_id, _ in state.in_progress_gates}
    remaining = [gate_id for gate_id in gate_order if gate_id not in state.completed_gates and gate_id not in running]
    if not remaining:
        return 0

    positions = dict(state.positions)
    routing_estimate = 0
    for gate_id in remaining:
        gate = gates[gate_id]
        if isinstance(gate, SingleQubitGate):
            continue
        if isinstance(gate, TwoQubitGate):
            routing_estimate += min_distance_to_valid_pair(
                positions[gate.ion_a],
                positions[gate.ion_b],
                architecture.valid_two_qubit_site_pairs,
            )
        else:
            routing_estimate += 1

    gate_estimate = (
        _critical_path_length(remaining, predecessors)
        if predecessors is not None
        else ceil(len(remaining) / len(architecture.processing_zones or {}))
    )
    return routing_estimate + gate_estimate


def _critical_path_length(
    remaining_gate_ids: Sequence[int],
    predecessors: Mapping[int, frozenset[int]],
) -> int:
    if not remaining_gate_ids:
        return 0

    remaining = set(remaining_gate_ids)
    direct_predecessors = {
        gate_id: [predecessor for predecessor in predecessors.get(gate_id, frozenset()) if predecessor in remaining]
        for gate_id in remaining_gate_ids
    }
    successors: dict[int, list[int]] = {gate_id: [] for gate_id in remaining_gate_ids}
    in_degree: dict[int, int] = {}
    for gate_id, gate_predecessors in direct_predecessors.items():
        in_degree[gate_id] = len(gate_predecessors)
        for predecessor in gate_predecessors:
            successors[predecessor].append(gate_id)

    longest_path = dict.fromkeys(remaining_gate_ids, 1)
    ready = [gate_id for gate_id in remaining_gate_ids if in_degree[gate_id] == 0]
    result = 0
    while ready:
        gate_id = ready.pop()
        result = max(result, longest_path[gate_id])
        for successor in successors[gate_id]:
            longest_path[successor] = max(
                longest_path[successor],
                longest_path[gate_id] + 1,
            )
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                ready.append(successor)
    return result


__all__ = ["cost", "heuristic", "min_distance_to_valid_pair"]
