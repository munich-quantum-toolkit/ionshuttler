# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for bounded Grid routing work."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mqt.ionshuttler.core.gates import Rx
from mqt.ionshuttler.core.schedule import ScheduledAction
from mqt.ionshuttler.grid import (
    Cycle,
    GridArchitecture,
    Junction,
    JunctionMove,
    Rearrangement,
    Segment,
    rectangular_grid,
    square_grid,
)
from mqt.ionshuttler.grid.actions import DEFAULT_ACTION_TYPES
from mqt.ionshuttler.grid.routing import (
    clear_blocked_path,
    junction_moves,
    movement_candidates,
    next_segments_toward,
    route_gate_bfs,
    route_to_placement,
)

if TYPE_CHECKING:
    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.grid import GridMachineState


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


def test_route_to_placement_reaches_an_exact_placement_of_eighteen_ions() -> None:
    """Find a route on a 4x4 Grid for a placement that a single cycle search within the bound missed."""
    architecture = square_grid(4, processing_zone_segments=("h:1:0", "h:1:2", "h:2:0", "h:2:2"), segment_capacity=2)
    start = architecture.initial_state({
        "v:2:0": [0],
        "h:0:2": [1],
        "h:1:2": [2, 9],
        "h:3:1": [3, 15],
        "v:2:3": [4, 12],
        "v:1:3": [5],
        "v:1:2": [6],
        "h:3:2": [7],
        "h:2:2": [8],
        "h:1:1": [10],
        "v:0:2": [11],
        "h:3:0": [13],
        "v:1:1": [14],
        "h:0:1": [16],
        "v:0:1": [17],
    })
    target = architecture.initial_state({
        "h:1:0": [0],
        "h:2:0": [1],
        "h:0:2": [2],
        "v:1:1": [3],
        "h:0:0": [4, 10],
        "h:3:1": [5],
        "v:2:1": [6],
        "v:1:2": [7],
        "v:1:0": [8, 14],
        "v:2:0": [9],
        "v:0:0": [11],
        "h:2:1": [12],
        "v:0:3": [13],
        "h:3:2": [15],
        "v:2:2": [16],
        "v:1:3": [17],
    })

    route, _explored = route_to_placement(architecture, start, max_states=10_000, target_state=target)

    assert route is not None
    state = start
    for action in route:
        timed = ScheduledAction(0, action, state.time, architecture.action_duration(action))
        state = architecture.apply_layer(state, (timed,)).at_time(timed.end_time)
    assert state.occupancy == target.occupancy


def _replay(architecture: GridArchitecture, state: GridMachineState, route: tuple[Action, ...]) -> GridMachineState:
    for action in route:
        timed = ScheduledAction(0, action, state.time, architecture.action_duration(action))
        state = architecture.apply_layer(state, (timed,)).at_time(timed.end_time)
    return state


def _ring_of_pairs(*, swaps: bool) -> GridArchitecture:
    a, b, c = (Segment(name, capacity=2) for name in "abc")
    action_types = tuple(action for action in DEFAULT_ACTION_TYPES if swaps or action is not Rearrangement)
    return GridArchitecture(
        (a, b, c),
        (Junction("ab", (a.end, b.start)), Junction("bc", (b.end, c.start)), Junction("ca", (c.end, a.start))),
        supported_action_types=action_types,
    )


def test_route_to_placement_reverses_a_full_segment_with_one_swap() -> None:
    """Restore the order inside a segment with one swap instead of a detour."""
    architecture = _ring_of_pairs(swaps=True)
    start = architecture.initial_state({"a": (0, 1)})
    target = architecture.initial_state({"a": (1, 0)})

    route, _explored = route_to_placement(architecture, start, max_states=100, target_state=target)

    assert route == (Rearrangement("a", (0, 1), (1, 0)),)


def test_route_to_placement_detours_without_swaps() -> None:
    """Reach the reversed order through junctions when the architecture has no swap."""
    architecture = _ring_of_pairs(swaps=False)
    start = architecture.initial_state({"a": (0, 1)})
    target = architecture.initial_state({"a": (1, 0)})

    route, _explored = route_to_placement(architecture, start, max_states=1_000, target_state=target)

    assert route is not None
    assert not any(isinstance(action, Rearrangement) for action in route)
    assert _replay(architecture, start, route).occupancy == target.occupancy


def test_route_gate_bfs_swaps_an_ion_to_the_only_exit() -> None:
    """Bring a gate ion past the ion in front when the only exit leads into a one-ion segment."""
    source = Segment("source", capacity=2)
    target = Segment("target")
    architecture = GridArchitecture((source, target), (Junction("j", (source.end, target.start)),))
    state = architecture.initial_state({"source": (0, 1), "target": ()})

    route, _explored = route_gate_bfs(architecture, state, Rx(ion=0, theta=0.5), frozenset({"target"}), max_states=100)

    assert route == (Rearrangement("source", (0, 1), (1, 0)), JunctionMove(source.end, target.start, (0,)))
