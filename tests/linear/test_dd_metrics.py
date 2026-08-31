# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for dynamical-decoupling comparison metrics."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import pytest

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, GlobalPulse, Ry
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.metrics import (
    decoupling_ratio,
    phase_reduction_per_gate,
    rank_ions_by_residual_phase,
    relative_phase_reduction,
    residual_phase_at_timestep,
    residual_phase_at_window_end,
    residual_phase_at_window_end_reduction,
    residual_phase_by_ion,
    sum_absolute_residual_phase,
    sum_squared_residual_phase,
    summarize_residual_phases,
    window_residual_phase,
)
from mqt.ionshuttler.linear.dd.result import LocalDDSequence
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state

if TYPE_CHECKING:
    from collections.abc import Sequence


def _result(
    path: Sequence[Action],
    timesteps: int,
    *,
    num_ions: int = 1,
    fields: tuple[float, ...] | None = None,
) -> tuple[ActionSchedule, Architecture]:
    field_values = fields or tuple(1.0 for _ in range(num_ions))
    architecture = Architecture(
        num_sites=num_ions,
        processing_zones={"pz": list(range(num_ions))},
        field_profile=FieldProfile(num_ions, tuple(enumerate(field_values))),
    )
    program = ActionSchedule.from_actions(
        path,
        create_initial_state(num_ions, architecture),
    )
    assert program.num_timesteps == timesteps
    return program, architecture


def test_decoupling_ratio_merges_overlapping_windows_and_counts_ion_time() -> None:
    """Avoid double-counting overlapping recorded windows for one ion."""
    records = (
        LocalDDSequence(0, (0, 6), "hahn", (3, 6), (10, 11)),
        LocalDDSequence(0, (4, 10), "hahn", (7, 10), (12, 13)),
        LocalDDSequence(1, (2, 6), "hahn", (4, 6), (14, 15)),
    )
    result, _architecture = _result([AdvanceTime() for _ in range(10)], 10, num_ions=2)

    assert decoupling_ratio(result, records) == pytest.approx(0.7)
    assert phase_reduction_per_gate(result, records) == pytest.approx(0.0)


def test_decoupling_ratio_clamps_windows_to_schedule_horizon() -> None:
    """Keep recorded coverage within the schedule's ion-time volume."""
    result, _architecture = _result([AdvanceTime() for _ in range(4)], 4)
    records = (LocalDDSequence(0, (0, 10), "hahn", (2,), (5,)),)

    assert decoupling_ratio(result, records) == pytest.approx(1.0)


def test_decoupling_ratio_ignores_records_for_unknown_ions() -> None:
    """Exclude stale or foreign records from the schedule's covered volume."""
    result, _architecture = _result([AdvanceTime() for _ in range(4)], 4)
    records = (
        LocalDDSequence(0, (0, 2), "hahn", (1,), (5,)),
        LocalDDSequence(99, (0, 4), "hahn", (2,), (6,)),
    )

    ratio = decoupling_ratio(result, records)

    assert ratio == pytest.approx(0.5)
    assert ratio <= 1.0


def test_relative_phase_reduction_empty_sequences_and_zero_and_normal_cases() -> None:
    """Cover the empty-sequences shortcut, the zero-total-phase guard, and a normal ratio."""
    result, architecture = _result([AdvanceTime() for _ in range(4)], 4)
    records = (LocalDDSequence(0, (0, 4), "hahn", (2,), (5,), phase_reduction=2.0),)

    assert relative_phase_reduction(result, architecture, ()) == pytest.approx(0.0)

    zero_field_result, zero_field_architecture = _result([AdvanceTime() for _ in range(4)], 4, fields=(0.0,))
    assert relative_phase_reduction(zero_field_result, zero_field_architecture, records) == pytest.approx(0.0)

    assert relative_phase_reduction(result, architecture, records) == pytest.approx(0.5)


def test_global_pulses_refocus_schedule_end_metrics() -> None:
    """Compute residual metrics from frame-aware replay."""
    base, architecture = _result([AdvanceTime() for _ in range(4)], 4)
    refocused, refocused_architecture = _result(
        [
            AdvanceTime(),
            GlobalPulse(GateSpec("Rx", pi)),
            AdvanceTime(),
            AdvanceTime(),
            GlobalPulse(GateSpec("Rx", pi)),
            AdvanceTime(),
        ],
        4,
    )

    assert residual_phase_by_ion(base, architecture) == {0: 4.0}
    assert sum_absolute_residual_phase(base, architecture) == pytest.approx(4.0)
    assert sum_squared_residual_phase(base, architecture) == pytest.approx(16.0)
    assert residual_phase_by_ion(refocused, refocused_architecture) == {0: 0.0}
    assert summarize_residual_phases(refocused, refocused_architecture).max_absolute_residual_phase == pytest.approx(
        0.0
    )


def test_window_and_prefix_metrics_distinguish_inherited_phase() -> None:
    """Keep window-only exposure separate from its closing prefix value."""
    result, architecture = _result(
        [
            Ry(ion=0, theta=0.25),
            AdvanceTime(),
            AdvanceTime(),
            GlobalPulse(GateSpec("Rx", pi)),
            AdvanceTime(),
            AdvanceTime(),
        ],
        4,
    )
    assert window_residual_phase(result, architecture, 0, (1, 4)) == pytest.approx(-1.0)
    assert residual_phase_at_timestep(result, architecture, 0, 4) == pytest.approx(0.0)
    assert residual_phase_at_window_end(result, architecture, 0, (1, 4)) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="within"):
        residual_phase_at_timestep(result, architecture, 0, 5)


def test_window_end_reduction_and_ranking_are_deterministic() -> None:
    """Compare endpoint magnitudes and rank equal-phase ions by identifier."""
    before, architecture = _result([AdvanceTime(), AdvanceTime()], 2, num_ions=2, fields=(1.0, 2.0))
    after, _ = _result(
        [AdvanceTime(), GlobalPulse(GateSpec("Rx", pi)), AdvanceTime()],
        2,
        num_ions=2,
        fields=(1.0, 2.0),
    )
    assert residual_phase_at_window_end_reduction(before, after, architecture, 1, (0, 2)) == pytest.approx(4.0)
    assert rank_ions_by_residual_phase(before, architecture) == ((1, 4.0), (0, 2.0))
