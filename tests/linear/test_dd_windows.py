# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for ion-local idle-window detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, Rx, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline
from mqt.ionshuttler.linear.dd.windows import find_idle_windows
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.state import create_initial_state

if TYPE_CHECKING:
    from collections.abc import Sequence


def _timeline(path: Sequence[Action], timesteps: int) -> CompiledTimeline:
    architecture = Architecture(num_sites=3, processing_zones={"pz": [0, 1, 2]})
    return build_timeline(
        CompilationResult(
            status=CompilationStatus.SUCCESS,
            path=list(path),
            num_timesteps=timesteps,
            architecture=architecture,
            initial_state=create_initial_state(1, architecture),
        )
    )


def test_idle_windows_split_around_full_gate_duration() -> None:
    """Exclude every occupied gate interval from maximal idle windows."""
    timeline = _timeline(
        [AdvanceTime(), Rx(ion=0, theta=1.0, duration=2), AdvanceTime(), AdvanceTime(), AdvanceTime()],
        4,
    )
    assert find_idle_windows(timeline, ion=0) == ((0, 1), (3, 4))


def test_transport_only_interval_remains_gate_idle() -> None:
    """Treat transport as phase exposure rather than a gate boundary."""
    timeline = _timeline([Shuttle(ion=0, src=0, dst=1), AdvanceTime(), AdvanceTime()], 2)
    assert find_idle_windows(timeline, ion=0) == ((0, 2),)
