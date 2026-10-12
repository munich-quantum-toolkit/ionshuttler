# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integrate field exposure along compiled ion trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.field_profile import FieldProfile
    from mqt.ionshuttler.linear.timeline import CompiledTimeline


def accumulated_phase(
    timeline: CompiledTimeline,
    ion: int,
    t_start: int,
    t_end: int,
    field_profile: FieldProfile | None,
) -> float:
    """Return the signed field exposure over a half-open time interval.

    Args:
        timeline: Reconstructed compiled schedule.
        ion: Ion whose trajectory is integrated.
        t_start: Inclusive start boundary.
        t_end: Exclusive end boundary.
        field_profile: Site-dependent field, or ``None`` for zero exposure.

    Returns:
        The accumulated field exposure in timestep-normalized units.

    Raises:
        ValueError: If the requested interval lies outside the timeline.
    """
    if not timeline.start_time <= t_start <= t_end <= timeline.end_time:
        msg = f"expected {timeline.start_time} <= t_start <= t_end <= {timeline.end_time}"
        raise ValueError(msg)
    if field_profile is None:
        return 0.0
    return sum(field_profile.field_at(timeline.ion_position(ion, timestep)) for timestep in range(t_start, t_end))


__all__ = ["accumulated_phase"]
