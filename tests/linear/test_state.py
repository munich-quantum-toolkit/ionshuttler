# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for immutable Linear compiler state."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.state import (
    State,
    create_initial_state,
    has_pending_timed_work,
    in_progress_dict,
    ions_busy_dict,
    normalize_initial_state,
    pzs_busy_dict,
    to_dict,
    to_metadata_dict,
    to_site_occupancy,
)


def test_state_normalizes_tuple_backed_mappings_and_is_hashable() -> None:
    """Treat equivalent state data as equal regardless of input order."""
    state = State(
        positions=((1, 4), (0, 1)),
        completed_gates=frozenset({1}),
        in_progress_gates=((3, 5), (2, 4)),
        ions_busy_until=((1, 3), (0, 0)),
        pzs_busy_until=(("pz_1", 2), ("pz_0", 0)),
        time=2,
    )

    assert state.positions == ((0, 1), (1, 4))
    assert state.in_progress_gates == ((2, 4), (3, 5))
    assert hash(state)
    assert to_dict(state) == {0: 1, 1: 4}
    assert in_progress_dict(state) == {2: 4, 3: 5}
    assert ions_busy_dict(state) == {0: 0, 1: 3}
    assert pzs_busy_dict(state) == {"pz_0": 0, "pz_1": 2}


def test_state_metadata_uses_site_occupancy() -> None:
    """Represent the starting ion placement site by site."""
    state = create_initial_state(num_ions=2, architecture=Architecture(num_sites=5))

    assert to_site_occupancy(state, 5) == [None, 0, 1, None, None]
    assert to_metadata_dict(state, 5) == {"site_occupancy": [None, 0, 1, None, None]}


@pytest.mark.parametrize(
    ("in_progress_gates", "ions_busy_until", "pzs_busy_until", "expected"),
    [
        ((), ((0, 3),), (("pz", 3),), 0),
        (((4, 5),), ((0, 3),), (("pz", 3),), 1),
        ((), ((0, 4),), (("pz", 3),), 1),
        ((), ((0, 3),), (("pz", 4),), 1),
    ],
)
def test_pending_timed_work_covers_each_scheduled_resource(
    in_progress_gates: tuple[tuple[int, int], ...],
    ions_busy_until: tuple[tuple[int, int], ...],
    pzs_busy_until: tuple[tuple[str, int], ...],
    expected: int,
) -> None:
    """Recognize running gates and busy hardware as work worth waiting for."""
    state = State(
        positions=((0, 0),),
        completed_gates=frozenset(),
        in_progress_gates=in_progress_gates,
        ions_busy_until=ions_busy_until,
        pzs_busy_until=pzs_busy_until,
        time=3,
    )

    assert has_pending_timed_work(state) is bool(expected)


def test_create_initial_state_centers_tightly_packed_ions() -> None:
    """Place ions together at the center of the chain by default."""
    state = create_initial_state(num_ions=3, architecture=Architecture(num_sites=7))

    assert state.positions == ((0, 2), (1, 3), (2, 4))
    assert state.ions_busy_until == ((0, 0), (1, 0), (2, 0))
    assert state.pzs_busy_until == (("all_sites", 0),)
    assert state.time == 0


def test_create_initial_state_accepts_explicit_positions_and_zones() -> None:
    """Place ions by ID and initialize all configured processing zones."""
    architecture = Architecture(num_sites=7, processing_zones={"A": [0, 1], "B": [5, 6]})
    state = create_initial_state(num_ions=3, architecture=architecture, initial_positions=[0, 3, 6])

    assert state.positions == ((0, 0), (1, 3), (2, 6))
    assert state.pzs_busy_until == (("A", 0), ("B", 0))


def test_normalize_initial_state_validates_occupancy_and_zone_clocks() -> None:
    """Reject impossible placements and add missing processing-zone availability."""
    architecture = Architecture(num_sites=4, processing_zones={"A": [0, 1], "B": [2, 3]})
    state = State(
        positions=((0, 0),),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=((0, 0),),
        pzs_busy_until=(("A", 2),),
        time=0,
    )

    assert normalize_initial_state(state, architecture).pzs_busy_until == (("A", 2), ("B", 0))

    with pytest.raises(ValueError, match="unknown processing zone"):
        normalize_initial_state(replace(state, pzs_busy_until=(("ghost", 0),)), architecture)
    with pytest.raises(ValueError, match="duplicate occupancy"):
        normalize_initial_state(replace(state, positions=((0, 0), (1, 0))), architecture)
    with pytest.raises(ValueError, match="site must be within"):
        normalize_initial_state(replace(state, positions=((0, 4),)), architecture)


@pytest.mark.parametrize(
    ("num_ions", "initial_positions", "message"),
    [
        (-1, None, "num_ions"),
        (5, None, "num_sites"),
        (2, [0], "length num_ions"),
        (2, [1, 1], "duplicate sites"),
        (2, [0, 4], "within"),
    ],
)
def test_create_initial_state_rejects_invalid_placement(
    num_ions: int,
    initial_positions: list[int] | None,
    message: str,
) -> None:
    """Reject invalid ion counts and explicit placements."""
    with pytest.raises(ValueError, match=message):
        create_initial_state(
            num_ions=num_ions,
            architecture=Architecture(num_sites=4),
            initial_positions=initial_positions,
        )
