# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for ion-local idle-window detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.ionshuttler.linear import GateTiming
from mqt.ionshuttler.linear.actions import Rx, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.dd.windows import find_idle_windows
from mqt.ionshuttler.linear.schedule import schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state
from mqt.ionshuttler.linear.timeline import CompiledTimeline, build_timeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.linear.state import SearchTransition


def _timeline(path: Sequence[SearchTransition]) -> CompiledTimeline:
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [0, 1, 2]},
        gate_timing=GateTiming(rx=2),
    )
    return build_timeline(schedule_from_path(path, create_initial_state(1, architecture), architecture), architecture)


def test_idle_windows_split_around_full_gate_duration() -> None:
    """Exclude every occupied gate interval from maximal idle windows."""
    timeline = _timeline([AdvanceTime(), Rx(ion=0, theta=1.0), AdvanceTime(), AdvanceTime(), AdvanceTime()])
    assert find_idle_windows(timeline, ion=0) == ((0, 1), (3, 4))


def test_transport_only_interval_remains_gate_idle() -> None:
    """Treat transport as phase exposure rather than a gate boundary."""
    timeline = _timeline([Shuttle(ion=0, src=0, dst=1), AdvanceTime(), AdvanceTime()])
    assert find_idle_windows(timeline, ion=0) == ((0, 2),)
