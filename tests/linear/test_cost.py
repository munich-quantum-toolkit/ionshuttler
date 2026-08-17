# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear schedule cost estimates."""

from __future__ import annotations

from mqt.ionshuttler.linear.actions import GateSpec, GlobalPulse, Rx, Rzz
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.cost import cost, heuristic, min_distance_to_valid_pair
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
