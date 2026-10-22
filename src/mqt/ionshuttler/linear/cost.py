# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Cost estimates that guide search, and a lower bound on the schedule duration.

The search estimates are not admissible. :func:`makespan_lower_bound` is a
true lower bound and serves to judge the quality of a finished schedule.
"""

from __future__ import annotations

from functools import lru_cache
from math import ceil
from sys import maxsize
from typing import TYPE_CHECKING, Protocol

from mqt.ionshuttler.core.gates import SingleQubitGate, TwoQubitGate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.circuit import Circuit
    from mqt.ionshuttler.linear.architecture import LinearArchitecture
    from mqt.ionshuttler.linear.state import State


class HeuristicFn(Protocol):
    """Estimate the remaining schedule work for one search state.

    A custom heuristic supplied through
    :attr:`~mqt.ionshuttler.linear.SearchConfig.heuristic` must accept these
    arguments. The search passes the circuit inputs positionally and the
    optional pre-partition data by keyword. The built-in :func:`heuristic`
    additionally accepts a cache, which the search passes only to that default.
    """

    def __call__(
        self,
        state: State,
        architecture: LinearArchitecture,
        circuit: Circuit,
        active_gate_ids: Sequence[int],
        predecessors: Sequence[frozenset[int]],
        /,
        *,
        use_dependencies: bool = True,
        gate_zone: Mapping[int, str] | None = None,
        zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
    ) -> int:
        """Estimate the work still needed to finish the requested gates.

        Args:
            state: Search state to score.
            architecture: Hardware layout the schedule targets.
            circuit: Circuit whose gates are being scheduled.
            active_gate_ids: Gates to schedule, including completed and running
                ones; filter with ``state.completed_gates`` and
                ``state.in_progress_gates``.
            predecessors: Effective direct predecessors indexed by gate ID.
            use_dependencies: Whether the estimate follows the dependency DAG.
            gate_zone: Optional preferred processing zone for each gate.
            zone_site_pairs: Optional valid two-ion site pairs for each zone.

        Returns:
            A nonnegative estimate of the remaining schedule time.
        """
        ...


def cost(state: State) -> int:
    """Return the cost, currently simply the elapsed schedule time."""
    return state.time


def zero_heuristic(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    active_gate_ids: Sequence[int],
    predecessors: Sequence[frozenset[int]],
    /,
    *,
    use_dependencies: bool = True,
    gate_zone: Mapping[int, str] | None = None,
    zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
) -> int:
    """Estimate nothing about the work left to finish the requested gates.

    This admissible estimate turns the search into a uniform-cost search, so
    the first complete schedule has the minimum makespan among the schedules
    the configured hardware and action model can express. It usually explores
    many more states than the default :func:`heuristic`.

    Args:
        state: Search state to score.
        architecture: Hardware layout the schedule targets.
        circuit: Circuit whose gates are being scheduled.
        active_gate_ids: Gates to schedule, including completed and running ones.
        predecessors: Effective direct predecessors indexed by gate ID.
        use_dependencies: Whether the estimate follows the dependency DAG.
        gate_zone: Optional preferred processing zone for each gate.
        zone_site_pairs: Optional valid two-ion site pairs for each zone.

    Returns:
        Always ``0``.
    """
    del state, architecture, circuit, active_gate_ids, predecessors, use_dependencies, gate_zone, zone_site_pairs
    return 0


# Keys are bounded by the site count squared for one architecture, so this cap
# holds the whole working set for realistic layouts while keeping a long-lived
# process from accumulating entries across many different architectures.
@lru_cache(maxsize=2**16)
def min_distance_to_valid_pair(
    pos_a: int,
    pos_b: int,
    valid_pairs: tuple[tuple[int, int], ...],
    architecture: LinearArchitecture,
) -> int:
    """Return the fewest simultaneous site moves needed to reach a valid pair.

    The result depends only on the two sites, the fixed pair list, and the
    architecture, so repeated lookups during search reuse a cached value.
    """
    if not valid_pairs:
        return 0
    distance = architecture.distance
    best = maxsize
    for left, right in valid_pairs:
        forward = max(distance(pos_a, left), distance(pos_b, right))
        reverse = max(distance(pos_a, right), distance(pos_b, left))
        best = min(best, forward, reverse)
    return best


def heuristic(
    state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    active_gate_ids: Sequence[int],
    predecessors: Sequence[frozenset[int]],
    *,
    use_dependencies: bool = True,
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
    remaining = [
        gate_id for gate_id in active_gate_ids if gate_id not in state.completed_gates and gate_id not in running
    ]
    if not remaining:
        return 0

    positions = dict(state.positions)
    routing_estimate = 0
    for gate_id in remaining:
        gate = circuit.gates[gate_id]
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
                architecture,
            )
        else:
            routing_estimate += 1

    if not use_dependencies:
        return routing_estimate + ceil(len(remaining) / len(architecture.processing_zones or {}))
    if critical_path_cache is None:
        return routing_estimate + _critical_path_length(remaining, predecessors)
    cache_key = tuple(remaining)
    gate_estimate = critical_path_cache.get(cache_key)
    if gate_estimate is None:
        gate_estimate = _critical_path_length(remaining, predecessors)
        critical_path_cache[cache_key] = gate_estimate
    return routing_estimate + gate_estimate


def makespan_lower_bound(circuit: Circuit, architecture: LinearArchitecture) -> int:
    """Return a duration that no complete schedule of the circuit can undercut.

    Unlike the search estimates in this module, this value is a true lower
    bound. It is the largest of three times:

    - the longest chain of dependent gates, adding their durations;
    - the total time of all gates that occupy a processing zone, spread evenly
      over the zones, because each zone runs one gate at a time;
    - the total time of the two-ion gates, spread evenly over the zones with
      at least two sites.

    The bound ignores transport, so the gap between it and a schedule duration
    includes all waiting for ion movement.

    Returns:
        The lower bound in timesteps.
    """
    zone_sizes = [len(sites) for sites in (architecture.processing_zones or {}).values()]
    gate_end_times: list[int] = []
    processing_zone_work = 0
    two_ion_work = 0
    # Predecessors have smaller gate IDs, so one pass in circuit order suffices.
    for gate, predecessors in zip(circuit.gates, circuit.predecessors, strict=True):
        duration = architecture.action_duration(gate)
        earliest_start = max(
            (gate_end_times[predecessor] for predecessor in predecessors),
            default=0,
        )
        gate_end_times.append(earliest_start + duration)
        if isinstance(gate, TwoQubitGate):
            processing_zone_work += duration
            two_ion_work += duration
        elif isinstance(gate, SingleQubitGate) and not architecture.is_virtual_gate(gate):
            processing_zone_work += duration
    two_ion_zone_count = sum(1 for size in zone_sizes if size >= 2)
    return max(
        max(gate_end_times, default=0),
        ceil(processing_zone_work / len(zone_sizes)) if zone_sizes else 0,
        ceil(two_ion_work / two_ion_zone_count) if two_ion_zone_count else 0,
    )


def _critical_path_length(
    remaining_gate_ids: Sequence[int],
    predecessors: Sequence[frozenset[int]],
) -> int:
    if not remaining_gate_ids:
        return 0

    remaining = set(remaining_gate_ids)
    direct_predecessors = {
        gate_id: [predecessor for predecessor in predecessors[gate_id] if predecessor in remaining]
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


__all__ = [
    "HeuristicFn",
    "cost",
    "heuristic",
    "makespan_lower_bound",
    "min_distance_to_valid_pair",
    "zero_heuristic",
]
