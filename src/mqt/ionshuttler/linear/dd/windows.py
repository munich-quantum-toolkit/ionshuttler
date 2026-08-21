# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Find ion-local idle windows in reconstructed schedules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline


def find_idle_windows(timeline: CompiledTimeline, ion: int) -> tuple[tuple[int, int], ...]:
    """Return maximal half-open intervals in which an ion has no gate."""
    windows: list[tuple[int, int]] = []
    window_start: int | None = None
    for timestep in range(timeline.makespan):
        if not timeline.ion_gate_busy(ion, timestep):
            if window_start is None:
                window_start = timestep
            continue
        if window_start is not None:
            windows.append((window_start, timestep))
            window_start = None
    if window_start is not None:
        windows.append((window_start, timeline.makespan))
    return tuple(windows)


__all__ = ["find_idle_windows"]
