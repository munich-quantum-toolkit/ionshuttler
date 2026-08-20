# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the idealized full-control Hahn reference."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, Rx, Ry, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd import IdealizedHahnConfig, IdealizedHahnReport, apply_idealized_hahn
from mqt.ionshuttler.linear.dd.frame_replay import PauliFrame, build_frame_history, framed_action_events
from mqt.ionshuttler.linear.dd.result import DDPassResult, LocalDDSequence
from mqt.ionshuttler.linear.dd.schemes import DDScheme
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state


def _result(
    architecture: Architecture,
    path: list[Action],
    *,
    num_timesteps: int,
    positions: list[int],
) -> ActionSchedule:
    program = ActionSchedule.from_actions(
        path,
        architecture,
        create_initial_state(len(positions), architecture, initial_positions=positions),
    )
    assert program.num_timesteps == num_timesteps
    return program


def test_idealized_hahn_inserts_before_transport_and_at_terminal_boundary() -> None:
    """Preserve the source comparator's overlapping and closing-pulse order."""
    architecture = Architecture(num_sites=3, processing_zones={"remote": [2]})
    original = _result(
        architecture,
        [Shuttle(ion=0, src=0, dst=1), AdvanceTime(), AdvanceTime()],
        num_timesteps=2,
        positions=[0],
    )

    output = apply_idealized_hahn(original)
    timeline = build_timeline(output.schedule)
    expected_record = LocalDDSequence(
        ion=0,
        window=(0, 2),
        scheme_name="IdealizedHahn",
        pulse_timesteps=(1, 2),
        action_ids=(3, 4),
    )

    assert output.report.sequences == (expected_record,)
    assert output.report.sequences[0].action_ids == tuple(
        item.action_id for item in output.schedule.scheduled_actions if isinstance(item.action, Rx)
    )
    assert timeline.action_at(0) == (Shuttle(ion=0, src=0, dst=1), AdvanceTime())
    assert timeline.action_at(1) == (Rx(ion=0, theta=pi), AdvanceTime())
    assert timeline.action_at(2) == (Rx(ion=0, theta=pi),)
    assert set(output.report.sequences[0].action_ids).isdisjoint(item.action_id for item in original.scheduled_actions)


def test_idealized_hahn_orders_terminal_pulses_before_logical_gates() -> None:
    """Insert all local terminal pulses before an existing logical gate."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    original = _result(
        architecture,
        [AdvanceTime(), AdvanceTime(), Ry(ion=1, theta=0.25)],
        num_timesteps=2,
        positions=[0, 1],
    )

    output = apply_idealized_hahn(original)
    timeline = build_timeline(output.schedule)
    local_pulse_action_ids = frozenset(
        action_id for sequence in output.report.sequences for action_id in sequence.action_ids
    )
    events = framed_action_events(output.schedule, timeline, local_pulse_action_ids)

    assert timeline.action_at(2) == (
        Rx(ion=0, theta=pi),
        Rx(ion=1, theta=pi),
        Ry(ion=1, theta=0.25),
    )
    assert [event.kind for event in events[-3:]] == [
        "local_dd_pulse",
        "local_dd_pulse",
        "algorithmic_gate",
    ]


def test_idealized_hahn_rounds_clamps_and_deduplicates_in_sequence_order() -> None:
    """Retain the first pulse when rounded scheme positions share a boundary."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    original = _result(
        architecture,
        [AdvanceTime() for _ in range(4)],
        num_timesteps=4,
        positions=[0],
    )
    scheme = DDScheme(
        "rounding_probe",
        relative_gate_times=(0.1, 0.12, 0.2, 0.95),
        gate_specs=(GateSpec("Rx", pi), GateSpec("Ry", pi), GateSpec("Ry", pi), GateSpec("Rx", pi)),
    )

    output = apply_idealized_hahn(
        original,
        config=IdealizedHahnConfig(scheme=scheme, label="RoundingProbe"),
    )

    assert output.report.sequences[0].pulse_timesteps == (0, 1, 4)
    assert [type(action) for action in output.schedule.path[:2]] == [Rx, AdvanceTime]
    assert build_timeline(output.schedule).action_at(4) == (Rx(ion=0, theta=pi),)


def test_idealized_hahn_replays_frames_and_round_trips_result_json() -> None:
    """Match midpoint frame semantics and retain legacy record serialization."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    original = _result(
        architecture,
        [AdvanceTime() for _ in range(4)],
        num_timesteps=4,
        positions=[0],
    )

    output = apply_idealized_hahn(original)
    restored = DDPassResult.from_json(output.to_json(), IdealizedHahnReport)
    local_pulse_action_ids = frozenset(output.report.sequences[0].action_ids)
    history = build_frame_history(build_timeline(output.schedule), local_pulse_action_ids)

    assert output.report.sequences[0].pulse_timesteps == (2, 4)
    assert history.frame_for_ion(0, 4) == PauliFrame("I")
    assert restored == output
    assert restored.report.sequences[0].action_ids == output.report.sequences[0].action_ids
    assert restored.report.sequences[0].to_dict()["pulse_timesteps"] == [2, 4]


def test_idealized_hahn_returns_unchanged_program_without_eligible_windows() -> None:
    """Avoid rebuilding when every idle window is shorter than configured."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    original = _result(architecture, [AdvanceTime()], num_timesteps=1, positions=[0])

    output = apply_idealized_hahn(original)

    assert output.schedule is original
    assert output.report.sequences == ()


def test_idealized_hahn_rejects_unknown_schemes() -> None:
    """Report unknown scheme names at the pass boundary."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    original = _result(architecture, [AdvanceTime()], num_timesteps=1, positions=[0])

    with pytest.raises(ValueError, match="unknown DD scheme"):
        apply_idealized_hahn(original, config=IdealizedHahnConfig(scheme="unknown", min_idle_timesteps=1))


def test_idealized_hahn_config_rejects_invalid_values() -> None:
    """Reject invalid idealized-reference configuration eagerly."""
    with pytest.raises(TypeError, match="min_idle_timesteps"):
        IdealizedHahnConfig(min_idle_timesteps=True)
    with pytest.raises(ValueError, match="min_idle_timesteps"):
        IdealizedHahnConfig(min_idle_timesteps=0)
    with pytest.raises(ValueError, match="label"):
        IdealizedHahnConfig(label="")
