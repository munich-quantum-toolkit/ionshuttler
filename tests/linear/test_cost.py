# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear schedule cost estimates."""

from __future__ import annotations

import mqt.ionshuttler.linear.cost as linear_cost
from mqt.ionshuttler.linear.actions import GateSpec, GlobalPulse, Rx, Rzz
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.cost import cost, heuristic, min_distance_to_valid_pair, zero_heuristic
from mqt.ionshuttler.linear.state import State


def make_state(
    positions: tuple[tuple[int, int], ...],
    *,
    completed: frozenset[int] = frozenset(),
    in_progress: tuple[tuple[int, int], ...] = (),
    time: int = 0,
) -> State:
    """Build a state with available ions and processing zones."""
    return State(
        positions=positions,
        completed_gates=completed,
        in_progress_gates=in_progress,
        ions_busy_until=tuple((ion, 0) for ion, _ in positions),
        pzs_busy_until=(("all_sites", 0),),
        time=time,
    )


def test_cost_is_elapsed_schedule_time() -> None:
    """Measure a partial schedule by its current timestep."""
    assert cost(make_state(((0, 0),), time=4)) == 4


def test_distance_chooses_the_closest_valid_site_pair() -> None:
    """Allow either ion ordering when choosing a processing-zone pair."""
    assert min_distance_to_valid_pair(0, 4, ((1, 3), (5, 6))) == 1
    assert min_distance_to_valid_pair(0, 4, ()) == 0


def test_two_qubit_estimate_uses_any_two_sites_in_one_zone() -> None:
    """Treat separated sites in a processing zone as directly gate-capable."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [1, 2, 3]})
    state = make_state(((0, 1), (1, 3)))
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0)}

    assert heuristic(state, architecture, [0], gates) == 1


def test_two_qubit_estimate_can_prefer_one_processing_zone() -> None:
    """Measure assigned gates against only their preferred zone's site pairs."""
    architecture = Architecture(
        num_sites=9,
        processing_zones={"left": [1, 2], "right": [7, 8]},
    )
    state = make_state(((0, 0), (1, 5)))
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0)}
    pairs = {"left": ((1, 2),), "right": ((7, 8),)}

    assert heuristic(state, architecture, [0], gates) == 4
    assert (
        heuristic(
            state,
            architecture,
            [0],
            gates,
            gate_zone={0: "right"},
            zone_site_pairs=pairs,
        )
        == 8
    )


def test_partition_parameters_omitted_preserve_heuristic_results() -> None:
    """Keep the original estimate unchanged when partition bias is not supplied."""
    architecture = Architecture(
        num_sites=9,
        processing_zones={"left": [1, 2], "right": [7, 8]},
    )
    state = make_state(((0, 0), (1, 5)))
    gates = {
        0: Rx(ion=0, theta=0.5),
        1: Rzz(ion_a=0, ion_b=1, theta=1.0),
        2: GlobalPulse(gate=GateSpec("rx", 0.25)),
    }
    predecessors = {0: frozenset(), 1: frozenset({0}), 2: frozenset()}

    original = heuristic(state, architecture, [0, 1, 2], gates, predecessors)
    explicit_defaults = heuristic(
        state,
        architecture,
        [0, 1, 2],
        gates,
        predecessors,
        gate_zone=None,
        zone_site_pairs=None,
    )

    assert explicit_defaults == original


def test_dependency_estimate_uses_the_remaining_critical_path() -> None:
    """Count serial gate depth while allowing independent gates in parallel."""
    architecture = Architecture(num_sites=2)
    state = make_state(((0, 0), (1, 1)), completed=frozenset({0}))
    gates = {
        0: Rx(ion=0, theta=1.0),
        1: Rx(ion=0, theta=0.5),
        2: Rx(ion=1, theta=0.25),
        3: Rx(ion=0, theta=0.125),
    }
    predecessors = {
        0: frozenset(),
        1: frozenset({0}),
        2: frozenset(),
        3: frozenset({1}),
    }

    assert heuristic(state, architecture, [0, 1, 2, 3], gates, predecessors) == 2


def test_running_gates_do_not_add_remaining_gate_cost() -> None:
    """Leave gates already in flight out of the remaining-work estimate."""
    architecture = Architecture(num_sites=1)
    state = make_state(((0, 0),), in_progress=((0, 2),))

    assert heuristic(state, architecture, [0], {0: Rx(ion=0, theta=1.0)}) == 0


def test_other_gate_types_add_one_unit_of_remaining_work() -> None:
    """Give an unfamiliar hardware gate a conservative nonzero estimate."""
    architecture = Architecture(num_sites=1)
    state = make_state(((0, 0),))
    gate = GlobalPulse(gate=GateSpec("rx", 0.5))

    assert heuristic(state, architecture, [0], {0: gate}) == 2


def test_zero_heuristic_estimates_nothing() -> None:
    """Return no remaining-cost estimate regardless of outstanding work."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1, 2]})
    state = State(
        positions=((0, 0), (1, 2)),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=(),
        pzs_busy_until=(),
        time=0,
    )
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0)}

    assert zero_heuristic(state, architecture, [0], gates) == 0
    assert zero_heuristic(state, architecture, [0], gates, {0: frozenset()}) == 0


def test_zero_heuristic_matches_the_search_short_circuit() -> None:
    """Agree with the constant the search substitutes for this estimate."""
    architecture = Architecture(num_sites=1)
    state = State(
        positions=((0, 0),),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=(),
        pzs_busy_until=(),
        time=0,
    )

    assert zero_heuristic(state, architecture, [0], {0: Rx(ion=0, theta=0.5)}) == 0


def test_empty_remaining_work_has_no_critical_path() -> None:
    """Report no depth once no gate is left to schedule."""
    assert linear_cost._critical_path_length([], {}) == 0


def test_estimate_without_dependencies_divides_across_processing_zones() -> None:
    """Spread remaining gates over the zones that can run them."""
    architecture = Architecture(num_sites=6, processing_zones={"a": [0, 1], "b": [4, 5]})
    state = State(
        positions=((0, 0), (1, 1), (2, 4), (3, 5)),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=(),
        pzs_busy_until=(),
        time=0,
    )
    gates = {
        0: Rx(ion=0, theta=0.5),
        1: Rx(ion=1, theta=0.5),
        2: Rx(ion=2, theta=0.5),
        3: Rx(ion=3, theta=0.5),
    }

    # Four single-qubit gates need no routing and split across two zones.
    assert heuristic(state, architecture, [0, 1, 2, 3], gates) == 2


def test_implicit_processing_zone_keeps_the_estimate_finite() -> None:
    """Estimate remaining work when no zone is configured explicitly."""
    architecture = Architecture(num_sites=1)
    state = State(
        positions=((0, 0),),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=(),
        pzs_busy_until=(),
        time=0,
    )
    gates = {0: Rx(ion=0, theta=0.5)}

    assert architecture.processing_zones is not None
    assert len(architecture.processing_zones) == 1
    assert heuristic(state, architecture, [0], gates) == 1


def test_unknown_preferred_zone_uses_architecture_pairs() -> None:
    """Ignore an assignment with no corresponding site-pair entry."""
    architecture = Architecture(num_sites=4, processing_zones={"left": [0, 1], "right": [2, 3]})
    state = make_state(((0, 0), (1, 1)))
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0)}

    assert heuristic(
        state, architecture, [0], gates, gate_zone={0: "missing"}, zone_site_pairs={"right": ((2, 3),)}
    ) == heuristic(state, architecture, [0], gates)
