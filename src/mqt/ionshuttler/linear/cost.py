# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Cost estimates used to choose promising schedules, not admissible."""

from __future__ import annotations

from functools import lru_cache
from math import ceil
from sys import maxsize
from typing import TYPE_CHECKING, Protocol

from mqt.ionshuttler.linear.actions import GateAction, SingleQubitGate, TwoQubitGate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.state import State


class HeuristicFn(Protocol):
    """Estimate the remaining schedule work for one search state.

    A custom heuristic supplied through
    :attr:`~mqt.ionshuttler.linear.SearchConfig.heuristic` must accept these
    arguments. The search always passes them positionally, so an
    implementation may name its parameters freely. The built-in
    :func:`heuristic` additionally accepts optional caching arguments, which
    the search passes only to that default.
    """

    def __call__(
        self,
        state: State,
        architecture: Architecture,
        gate_order: Sequence[int],
        gates: Mapping[int, GateAction],
        predecessors: Mapping[int, frozenset[int]] | None = None,
        /,
    ) -> int:
        """Estimate the work still needed to finish the requested gates.

        Args:
            state: Search state to score.
            architecture: Hardware layout the schedule targets.
            gate_order: Gates to schedule, including completed and running
                ones; filter with ``state.completed_gates`` and
                ``state.in_progress_gates``.
            gates: Gate actions indexed by identifier.
            predecessors: Optional map of which gates must precede others.

        Returns:
            A nonnegative estimate of the remaining schedule time.
        """
        ...


def cost(state: State) -> int:
    """Return the cost, currently simply the elapsed schedule time."""
    return state.time


def zero_heuristic(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: Mapping[int, GateAction],
    predecessors: Mapping[int, frozenset[int]] | None = None,
    /,
) -> int:
    """Estimate nothing about the work left to finish the requested gates.

    This admissible estimate turns the search into a uniform-cost search, so
    the first complete schedule has the minimum makespan among the schedules
    the configured hardware and action model can express. It usually explores
    many more states than the default :func:`heuristic`.

    Args:
        state: Search state to score.
        architecture: Hardware layout the schedule targets.
        gate_order: Gates to schedule, including completed and running ones.
        gates: Gate actions indexed by identifier.
        predecessors: Optional map of which gates must precede others.

    Returns:
        Always ``0``.
    """
    del state, architecture, gate_order, gates, predecessors
    return 0


# Keys are bounded by the site count squared for one architecture, so this cap
# holds the whole working set for realistic layouts while keeping a long-lived
# process from accumulating entries across many different architectures.
@lru_cache(maxsize=2**16)
def min_distance_to_valid_pair(
    pos_a: int,
    pos_b: int,
    valid_pairs: tuple[tuple[int, int], ...],
) -> int:
    """Return the fewest simultaneous site moves needed to reach a valid pair.

    The result depends only on the two sites and the architecture's fixed pair
    list, so repeated lookups during search reuse a cached value.
    """
    if not valid_pairs:
        return 0
    best = maxsize
    for left, right in valid_pairs:
        forward = abs(pos_a - left)
        other = abs(pos_b - right)
        forward = max(forward, other)
        reverse = abs(pos_a - right)
        other = abs(pos_b - left)
        reverse = max(reverse, other)
        forward = min(forward, reverse)
        best = min(best, forward)
    return best


def heuristic(
    state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: Mapping[int, GateAction],
    predecessors: Mapping[int, frozenset[int]] | None = None,
    *,
    critical_path_cache: dict[tuple[int, ...], int] | None = None,
    gate_zone: Mapping[int, str] | None = None,
    zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
) -> int:
    """Estimate the work still needed to finish the requested gates.

    Movement and gate execution can overlap, so this estimate may overstate
    the remaining schedule time and does not guarantee an optimal result.

    Supplying ``critical_path_cache`` reuses gate-depth estimates across states
    that share the same outstanding gates. The caller owns the dictionary and
    must not reuse it across different circuits or dependency maps.

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
            valid_pairs = architecture.valid_two_qubit_site_pairs
            if gate_zone is not None and zone_site_pairs is not None and gate_id in gate_zone:
                valid_pairs = zone_site_pairs.get(gate_zone[gate_id], valid_pairs)
            routing_estimate += min_distance_to_valid_pair(
                positions[gate.ion_a],
                positions[gate.ion_b],
                valid_pairs,
            )
        else:
            routing_estimate += 1

    if predecessors is None:
        return routing_estimate + ceil(len(remaining) / len(architecture.processing_zones or {}))
    if critical_path_cache is None:
        return routing_estimate + _critical_path_length(remaining, predecessors)
    cache_key = tuple(remaining)
    gate_estimate = critical_path_cache.get(cache_key)
    if gate_estimate is None:
        gate_estimate = _critical_path_length(remaining, predecessors)
        critical_path_cache[cache_key] = gate_estimate
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


__all__ = ["HeuristicFn", "cost", "heuristic", "min_distance_to_valid_pair", "zero_heuristic"]
