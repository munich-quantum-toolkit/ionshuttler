# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for periodic global dynamical decoupling."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING, cast

import pytest

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, GlobalPulse, Rx, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd import GlobalDDConfig, GlobalDDReport, apply_periodic_global_dd
from mqt.ionshuttler.linear.dd.frame_replay import PauliFrame, build_frame_history
from mqt.ionshuttler.linear.dd.metrics import residual_phase_by_ion, sum_absolute_residual_phase
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus, GlobalDDRecord
from mqt.ionshuttler.linear.state import create_initial_state

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.dd.global_dd import ShiftObjective


def _result(
    architecture: Architecture,
    path: list[Action],
    *,
    num_timesteps: int,
    positions: list[int] | None = None,
) -> CompilationResult:
    initial_positions = positions or [0]
    return CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=path,
        num_timesteps=num_timesteps,
        architecture=architecture,
        initial_state=create_initial_state(len(initial_positions), architecture, initial_positions=initial_positions),
    )


def test_periodic_global_dd_overlaps_existing_actions_and_records_summary() -> None:
    """Insert global pulses first at each selected boundary and serialize them."""
    architecture = Architecture(
        num_sites=2,
        processing_zones={"pz": [0, 1]},
        field_profile=FieldProfile(2, ((0, 1.0), (1, 1.0))),
    )
    original = _result(
        architecture,
        [
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Rx(ion=0, theta=0.5),
            AdvanceTime(),
        ],
        num_timesteps=2,
    )

    output = apply_periodic_global_dd(original, GlobalDDConfig(spacing=1))
    timeline = build_timeline(output.program)
    restored = CompilationResult.from_json(output.program.to_json())
    expected_record = GlobalDDRecord(
        scheme_name="periodic_x",
        pulse_timesteps=(0, 1),
        spacing=1,
        sum_abs_residual_phase=0.0,
        sum_squared_residual_phase=0.0,
        max_abs_residual_phase=0.0,
    )

    assert timeline.action_at(0) == (
        GlobalPulse(GateSpec("Rx", pi)),
        Shuttle(ion=0, src=0, dst=1),
        AdvanceTime(),
    )
    assert timeline.action_at(1) == (
        GlobalPulse(GateSpec("Rx", pi)),
        Rx(ion=0, theta=0.5),
        AdvanceTime(),
    )
    assert output.program.global_dd_records == (expected_record,)
    assert output.report.pulse_timesteps == (0, 1)
    assert restored.to_dict() == output.program.to_dict()
    assert restored.to_dict()["actions"] == [
        {"type": "GlobalPulse", "gate_name": "Rx", "theta": pi, "start_time": 0, "duration": 1},
        {"type": "Shuttle", "ion": 0, "src": 0, "dst": 1, "start_time": 0, "duration": 1},
        {"type": "AdvanceTime", "start_time": 0, "duration": 1},
        {"type": "GlobalPulse", "gate_name": "Rx", "theta": pi, "start_time": 1, "duration": 1},
        {"type": "Rx", "ion": 0, "theta": 0.5, "start_time": 1, "duration": 1},
        {"type": "AdvanceTime", "start_time": 1, "duration": 1},
    ]
    assert restored.to_dict()["global_dd_records"] == [
        {
            "scheme_name": "periodic_x",
            "pulse_timesteps": [0, 1],
            "spacing": 1,
            "sum_abs_residual_phase": 0.0,
            "sum_squared_residual_phase": 0.0,
            "max_abs_residual_phase": 0.0,
        }
    ]


def test_periodic_global_dd_centers_the_first_odd_spacing_window() -> None:
    """Use the source comparator's centered first pulse by default."""
    architecture = Architecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        field_profile=FieldProfile(1, ((0, 1.0),), default_field=0.0),
    )
    original = _result(
        architecture,
        [AdvanceTime() for _ in range(5)],
        num_timesteps=5,
    )

    output = apply_periodic_global_dd(original, GlobalDDConfig(spacing=5))
    history = build_frame_history(build_timeline(output.program), result=output.program)

    assert output.program.global_dd_records == (
        GlobalDDRecord(
            scheme_name="periodic_x",
            pulse_timesteps=(2,),
            spacing=5,
            sum_abs_residual_phase=1.0,
            sum_squared_residual_phase=1.0,
            max_abs_residual_phase=1.0,
        ),
    )
    assert output.report.max_absolute_residual_phase == pytest.approx(1.0)
    assert history.frame_for_ion(0, 1) == PauliFrame("I")
    assert history.frame_for_ion(0, 2) == PauliFrame("X")


@pytest.mark.parametrize("objective", ["sum_abs", "sum_squared"])
def test_periodic_global_dd_shifts_pulses_to_reduce_cross_ion_residuals(objective: str) -> None:
    """Match source coordinate-descent choices for both supported objectives."""
    architecture = Architecture(
        num_sites=2,
        processing_zones={"pz": [0, 1]},
        field_profile=FieldProfile(2, ((0, 1.0), (1, -1.0)), default_field=0.0),
    )
    original = _result(
        architecture,
        [AdvanceTime() for _ in range(5)],
        num_timesteps=5,
        positions=[0, 1],
    )
    periodic = apply_periodic_global_dd(
        original,
        GlobalDDConfig(spacing=5, half_first_window=False),
    ).program
    shifted = apply_periodic_global_dd(
        original,
        GlobalDDConfig(
            spacing=5,
            shift_range=2,
            shift_objective=cast("ShiftObjective", objective),
            half_first_window=False,
        ),
    ).program

    assert periodic.global_dd_records[0].pulse_timesteps == (4,)
    assert periodic.global_dd_records[0].sum_abs_residual_phase == pytest.approx(6.0)
    assert periodic.global_dd_records[0].sum_squared_residual_phase == pytest.approx(18.0)
    assert shifted.global_dd_records[0].pulse_timesteps == (2,)
    assert shifted.global_dd_records[0].sum_abs_residual_phase == pytest.approx(2.0)
    assert shifted.global_dd_records[0].sum_squared_residual_phase == pytest.approx(2.0)
    assert sum(residual_phase_by_ion(periodic).values()) == pytest.approx(0.0)
    assert sum(residual_phase_by_ion(shifted).values()) == pytest.approx(0.0)
    assert sum_absolute_residual_phase(shifted) < sum_absolute_residual_phase(periodic)


def test_periodic_global_dd_keeps_unimproved_times_and_returns_short_noop() -> None:
    """Keep strict-improvement ties stable and avoid rebuilding empty sequences."""
    architecture = Architecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        field_profile=FieldProfile(1, ((0, 1.0),), default_field=0.0),
    )
    original = _result(architecture, [AdvanceTime() for _ in range(4)], num_timesteps=4)
    periodic = apply_periodic_global_dd(original, GlobalDDConfig(spacing=2))
    shifted = apply_periodic_global_dd(original, GlobalDDConfig(spacing=2, shift_range=1))
    short = _result(architecture, [AdvanceTime()], num_timesteps=1)
    unchanged = apply_periodic_global_dd(short, GlobalDDConfig(spacing=5))

    assert shifted.program.global_dd_records == periodic.program.global_dd_records
    assert unchanged.program is short
    assert unchanged.report.pulse_timesteps == ()
    assert unchanged.program.global_dd_records == ()


def test_periodic_global_dd_rejects_existing_global_pulses() -> None:
    """Reject ambiguous augmentation of a schedule with global pulses."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    original = _result(
        architecture,
        [GlobalPulse(GateSpec("Rx", pi)), AdvanceTime()],
        num_timesteps=1,
    )

    with pytest.raises(ValueError, match="without global pulses"):
        apply_periodic_global_dd(original, GlobalDDConfig(spacing=1))


def test_global_dd_config_rejects_invalid_values_and_pulses() -> None:
    """Validate placement controls and the global X-frame restriction."""
    with pytest.raises(TypeError, match="spacing"):
        GlobalDDConfig(spacing=True)
    with pytest.raises(ValueError, match="spacing"):
        GlobalDDConfig(spacing=0)
    with pytest.raises(ValueError, match="shift_range"):
        GlobalDDConfig(spacing=2, shift_range=-1)
    with pytest.raises(TypeError, match="shift_objective"):
        GlobalDDConfig(spacing=2, shift_objective=cast("ShiftObjective", []))
    with pytest.raises(ValueError, match="shift_objective"):
        GlobalDDConfig(spacing=2, shift_objective=cast("ShiftObjective", "signed_sum"))
    with pytest.raises(ValueError, match="only global X"):
        GlobalDDConfig(spacing=2, pulse=GateSpec("Ry", pi))


def test_global_dd_report_rejects_malformed_public_values() -> None:
    """Keep comparator summaries immutable and internally consistent."""
    with pytest.raises(ValueError, match="strictly increasing"):
        GlobalDDReport(
            scheme_name="periodic_x",
            pulse_timesteps=(3, 1),
            spacing=2,
            sum_absolute_residual_phase=0.0,
            sum_squared_residual_phase=0.0,
            max_absolute_residual_phase=0.0,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        GlobalDDReport(
            scheme_name="periodic_x",
            pulse_timesteps=(1, 3),
            spacing=2,
            sum_absolute_residual_phase=-1.0,
            sum_squared_residual_phase=0.0,
            max_absolute_residual_phase=0.0,
        )
