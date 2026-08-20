# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for normalized critical-segment phase replay."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING, cast

import pytest

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, Rx, Rxx, Ry, Rz, Rzz
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.critical_segments import (
    SegmentationMode,
    compute_critical_segments,
    gate_z_effect,
    normalized_sensitivity_values,
)
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state

if TYPE_CHECKING:
    from collections.abc import Sequence


def _result(
    path: Sequence[Action],
    *,
    timesteps: int = 3,
) -> ActionSchedule:
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    program = ActionSchedule.from_actions(
        path,
        architecture,
        create_initial_state(1, architecture),
    )
    assert program.num_timesteps == timesteps
    return program


def test_gate_z_effect_classifies_preserving_flipping_and_mixing_gates() -> None:
    """Match the source gate classification used by the phase objective."""
    assert gate_z_effect(Rz(ion=0, theta=0.2), 0) == 1
    assert gate_z_effect(Rzz(ion_a=0, ion_b=1, theta=0.2), 0) == 1
    assert gate_z_effect(Rx(ion=0, theta=pi), 0) == -1
    assert gate_z_effect(Ry(ion=0, theta=2 * pi), 0) == 1
    assert gate_z_effect(Rxx(ion_a=0, ion_b=1, theta=0.2), 0) is None
    with pytest.raises(TypeError, match="gate action"):
        gate_z_effect(AdvanceTime(), 0)


def test_mixing_gate_closes_segment_and_pi_gate_only_flips_sign() -> None:
    """Split at mixing gates while carrying exact integer-pi toggling signs."""
    mixed = compute_critical_segments(_result([AdvanceTime(), Ry(ion=0, theta=pi / 2), AdvanceTime(), AdvanceTime()]))
    flipped = compute_critical_segments(_result([AdvanceTime(), Rx(ion=0, theta=pi), AdvanceTime(), AdvanceTime()]))

    assert [(segment.start, segment.end, segment.boundary_gate) for segment in mixed.segments] == [
        (0, 1, "Ry"),
        (1, 3, None),
    ]
    assert flipped.segments[0].toggling_signs == (1, -1, -1)
    assert flipped.segments[0].phase == pytest.approx(-1.0)


def test_recorded_dd_frame_persists_across_critical_boundary() -> None:
    """Keep the DD frame separate from algorithmic segment-local signs."""
    program = _result(
        [Rx(ion=0, theta=pi), AdvanceTime(), Ry(ion=0, theta=pi / 2), AdvanceTime()],
        timesteps=2,
    )
    local_pulse_action_ids = frozenset({program.scheduled_actions[0].action_id})
    assert [
        segment.toggling_signs
        for segment in compute_critical_segments(
            program,
            local_pulse_action_ids=local_pulse_action_ids,
        ).segments
    ] == [(-1,), (-1,)]


def test_whole_schedule_mode_and_j_phi_semantics() -> None:
    """Expose the unsegmented comparator and squared normalized phase surrogate."""
    result = _result([AdvanceTime(), Ry(ion=0, theta=pi / 2), AdvanceTime(), AdvanceTime()])
    trace = compute_critical_segments(result, segmentation="whole_schedule", dt=0.5)

    assert len(trace.segments) == 1
    assert trace.segments[0].phase == pytest.approx(1.5)
    assert trace.j_phi == pytest.approx(2.25)
    with pytest.raises(ValueError, match="segmentation"):
        compute_critical_segments(result, segmentation=cast("SegmentationMode", "invalid"))


def test_sensitivity_profile_is_rms_normalized_and_validated() -> None:
    """Restrict the generic field profile to the nonnegative DD envelope."""
    architecture = Architecture(
        num_sites=2,
        processing_zones={"pz": [0, 1]},
        field_profile=FieldProfile(2, ((0, 0.0), (1, 3.0))),
    )
    values = normalized_sensitivity_values(architecture)
    assert sum(value**2 for value in values) / 2 == pytest.approx(1.0)
    assert values[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="non-negative"):
        normalized_sensitivity_values(architecture, FieldProfile(2, ((0, -1.0),)))
    with pytest.raises(ValueError, match="positive RMS"):
        normalized_sensitivity_values(architecture, FieldProfile(2, (), default_field=0.0))
    with pytest.raises(ValueError, match="finite"):
        normalized_sensitivity_values(architecture, FieldProfile(2, ((0, float("inf")),)))
