# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for bounded Grid routing work."""

from __future__ import annotations

import pytest

from mqt.ionshuttler.core.gates import Rx
from mqt.ionshuttler.grid import Cycle, GridArchitecture, Junction, JunctionMove, Segment, rectangular_grid
from mqt.ionshuttler.grid.routing import (
    clear_blocked_path,
    junction_moves,
    movement_candidates,
    next_segments_toward,
    route_gate_bfs,
)


@pytest.mark.parametrize("capacity", [2, 3])
def test_cycle_candidates_move_whole_crystals(capacity: int) -> None:
    """Construct cycles whose junction moves contain every ion in a crystal."""
    segments = tuple(Segment(name, capacity=capacity) for name in ("a", "b", "c"))
    a, b, c = segments
    architecture = GridArchitecture(
        segments,
        (
            Junction("ab", (a.end, b.start)),
            Junction("bc", (b.end, c.start)),
            Junction("ca", (c.end, a.start)),
        ),
    )
    state = architecture.initial_state({
        segment.segment_id: tuple(range(index * capacity, (index + 1) * capacity))
        for index, segment in enumerate(segments)
    })

    candidates, _explored = movement_candidates(architecture, state, max_cycle_work=10_000)

    assert any(
        isinstance(candidate, Cycle)
        and {move.source.segment_id for move in candidate.moves} == {"a", "b", "c"}
        and all(len(move.ions) == capacity for move in candidate.moves)
        for candidate in candidates
    )


def _blocked_exit_architecture(*, with_side: bool = True, side_capacity: int = 1) -> GridArchitecture:
    """Return a source whose exit junction leads to a dead-end target and an optional side segment."""
    source = Segment("source", capacity=2)
    target = Segment("target")
    side = Segment("side", capacity=side_capacity)
    spare = Segment("spare")
    if not with_side:
        return GridArchitecture((source, target), (Junction("exit", (source.end, target.start)),))
    return GridArchitecture(
        (source, target, side, spare),
        (
            Junction("exit", (source.end, target.start, side.start)),
            Junction("side-spare", (side.end, spare.start)),
        ),
    )


def test_clear_blocked_path_moves_the_ion_in_front_into_a_side_segment() -> None:
    """Move the ion between the requested ion and the exit junction out of the way."""
    architecture = _blocked_exit_architecture()
    state = architecture.initial_state({"source": (0, 1)})

    actions, _explored = clear_blocked_path(
        architecture,
        state,
        0,
        "target",
        max_states=100,
        moves=junction_moves(architecture, state),
        ion_priorities={0: 0},
    )

    assert actions == (JunctionMove(architecture.segment("source").end, architecture.segment("side").start, (1,)),)


def test_clear_blocked_path_clears_a_full_side_segment_from_its_free_end() -> None:
    """Shift the side segment's ion onward so that the ion in front can step aside in the same layer."""
    architecture = _blocked_exit_architecture()
    state = architecture.initial_state({"source": (0, 1), "side": (2,)})

    actions, _explored = clear_blocked_path(
        architecture,
        state,
        0,
        "target",
        max_states=100,
        moves=junction_moves(architecture, state),
        ion_priorities={0: 0},
    )

    assert actions is not None
    assert {(move.ions, move.destination.segment_id) for move in actions} == {((1,), "side"), ((2,), "spare")}


def test_clear_blocked_path_keeps_the_destination_free_for_the_ion() -> None:
    """Refuse to let the ion in front take the free capacity that the ion needs."""
    architecture = _blocked_exit_architecture(with_side=False)
    state = architecture.initial_state({"source": (0, 1)})

    actions, _explored = clear_blocked_path(
        architecture,
        state,
        0,
        "target",
        max_states=100,
        moves=junction_moves(architecture, state),
        ion_priorities={0: 0},
    )

    assert actions is None


def test_clear_blocked_path_does_not_move_a_higher_priority_ion_aside() -> None:
    """Leave an ion in front in place if it has a higher priority than the requested ion."""
    architecture = _blocked_exit_architecture()
    state = architecture.initial_state({"source": (0, 1)})

    actions, _explored = clear_blocked_path(
        architecture,
        state,
        0,
        "target",
        max_states=100,
        moves=junction_moves(architecture, state),
        ion_priorities={1: 0, 0: 1},
    )

    assert actions is None


def test_route_gate_bfs_returns_direct_placement_at_state_limit() -> None:
    """Return a legal one-move route without spending work on cycle discovery."""
    source = Segment("source")
    target = Segment("target")
    architecture = GridArchitecture(
        (source, target),
        (Junction("junction", (source.end, target.start)),),
    )
    state = architecture.initial_state({"source": (0,), "target": ()})

    route, explored = route_gate_bfs(
        architecture,
        state,
        Rx(ion=0, theta=0.5),
        frozenset({"target"}),
        max_states=1,
    )

    assert route is not None
    assert len(route) == 1
    assert explored == 1


def test_route_gate_bfs_counts_bounded_cycle_search_work() -> None:
    """Charge cycle enumeration to the shared routing work limit."""
    first = Segment("first")
    target = Segment("target")
    third = Segment("third")
    architecture = GridArchitecture(
        (first, target, third),
        (
            Junction("first-target", (first.end, target.start)),
            Junction("target-third", (target.end, third.start)),
            Junction("third-first", (third.end, first.start)),
        ),
    )
    state = architecture.initial_state({"first": (0,), "target": (1,), "third": (2,)})

    route, explored = route_gate_bfs(
        architecture,
        state,
        Rx(ion=0, theta=0.5),
        frozenset({"target"}),
        max_states=3,
    )

    assert route is None
    assert explored == 3


def test_next_segments_toward_returns_all_equal_shortest_choices() -> None:
    """Expose each shortest next segment so that the scheduler can prefer free capacity."""
    architecture = rectangular_grid(4, 4)

    candidates = next_segments_toward(architecture, "v:2:1", "v:2:0")

    assert candidates == ("h:2:0", "h:3:0")
