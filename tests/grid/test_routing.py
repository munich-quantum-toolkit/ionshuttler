# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for bounded Grid routing work."""

from __future__ import annotations

from mqt.ionshuttler.core.gates import Rx
from mqt.ionshuttler.grid import GridArchitecture, Junction, Segment
from mqt.ionshuttler.grid.routing import route_gate_bfs


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
