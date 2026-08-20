# Copyright (c) 2026 Chair for Design Automation, TUM
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

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, GlobalPulse, Rx, Ry, Rzz
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.metrics import (
    decoupling_ratio,
    gate_residual_events_by_ion,
    phase_reduction_per_gate,
    rank_ions_by_residual_phase,
    residual_phase_at_timestep,
    residual_phase_at_window_end,
    residual_phase_at_window_end_reduction,
    residual_phase_by_ion,
    sum_absolute_residual_phase,
    sum_absolute_residual_phases_at_gates,
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


def test_gate_events_ignore_local_dd_and_preserve_order_and_duration() -> None:
    """Observe algorithmic gates without counting recorded DD pulses."""
    result, architecture = _result(
        [
            Rx(ion=0, theta=pi),
            Ry(ion=0, theta=0.25, duration=2),
            AdvanceTime(),
            AdvanceTime(),
            Ry(ion=0, theta=0.5),
            AdvanceTime(),
        ],
        3,
    )
    events = gate_residual_events_by_ion(
        result,
        architecture,
        local_pulse_action_ids=frozenset({result.scheduled_actions[0].action_id}),
    )[0]

    assert [(event.timestep, event.duration, event.gate_name) for event in events] == [
        (0, 2, "Ry"),
        (2, 1, "Ry"),
    ]


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


def test_gate_summed_metric_counts_each_two_qubit_operand() -> None:
    """Sum prefix phase separately for both participants of a two-ion gate."""
    result, architecture = _result(
        [AdvanceTime(), Ry(ion=0, theta=0.5), AdvanceTime(), Rzz(0, 1, 1.0), AdvanceTime()],
        3,
        num_ions=2,
        fields=(1.0, 2.0),
    )
    assert sum_absolute_residual_phases_at_gates(result, architecture) == pytest.approx(7.0)
