# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for dynamical-decoupling phase integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.phase import accumulated_phase
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state

if TYPE_CHECKING:
    from collections.abc import Sequence


def _timeline(path: Sequence[Action], _num_timesteps: int) -> CompiledTimeline:
    architecture = Architecture(num_sites=3)
    return build_timeline(
        ActionSchedule.from_actions(
            path,
            create_initial_state(1, architecture, initial_positions=[0]),
        ),
        architecture,
    )


def test_accumulated_phase_matches_static_and_moving_exposure() -> None:
    """Integrate the field at each source-compatible trajectory boundary."""
    static = _timeline([AdvanceTime(), AdvanceTime(), AdvanceTime()], 3)
    moving = _timeline(
        [
            AdvanceTime(),
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Shuttle(ion=0, src=1, dst=2),
            AdvanceTime(),
        ],
        3,
    )
    profile = FieldProfile(num_sites=3, site_field=((0, 0.2), (1, 0.7), (2, -0.1)))

    assert accumulated_phase(static, 0, 0, 3, profile) == pytest.approx(0.6)
    assert accumulated_phase(moving, 0, 0, 3, profile) == pytest.approx(0.8)
    assert accumulated_phase(moving, 0, 1, 1, profile) == pytest.approx(0.0)


def test_accumulated_phase_handles_absent_profile_and_invalid_interval() -> None:
    """Treat an absent field as zero and reject invalid half-open intervals."""
    timeline = _timeline([AdvanceTime(), AdvanceTime()], 2)

    assert accumulated_phase(timeline, 0, 0, 2, None) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="expected"):
        accumulated_phase(timeline, 0, 1, 3, FieldProfile(num_sites=3, site_field=()))
