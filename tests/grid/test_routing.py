# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for bounded Grid routing work."""

from __future__ import annotations

from mqt.ionshuttler.core.gates import Rx
from mqt.ionshuttler.grid import GridArchitecture, Junction, JunctionMove, Segment
from mqt.ionshuttler.grid.routing import clear_blocked_path, junction_moves, route_gate_bfs


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
