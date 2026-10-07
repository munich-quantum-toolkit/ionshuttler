# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the schedule-to-DD ownership contract."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES, GlobalGate, Rx, SingleQubitGate
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.dd import (
    GlobalDDConfig,
    IdealizedHahnConfig,
    NearestHahnConfig,
    apply_idealized_hahn,
    apply_periodic_global_dd,
    compute_critical_segments,
    decoupling_ratio,
    residual_phase_by_ion,
    run_nearest_hahn,
)
from mqt.ionshuttler.linear.dd.frame_replay import build_frame_history, framed_action_events
from mqt.ionshuttler.linear.dd.result import LocalDDSequence
from mqt.ionshuttler.linear.dd.windows import find_idle_windows
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.schedule import Schedule, schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state
from mqt.ionshuttler.linear.timeline import build_timeline


def test_local_pulse_identity_is_owned_by_the_dd_report() -> None:
    """Keep local-pulse provenance out of the hardware-facing schedule."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    original = schedule_from_path(
        [AdvanceTime() for _ in range(4)],
        create_initial_state(1, architecture),
        architecture,
    )

    output = apply_idealized_hahn(original, architecture)
    sequence = output.report.sequences[0]
    scheduled_by_id = {item.action_id: item.action for item in output.schedule.scheduled_actions}

    assert len(sequence.action_ids) == len(sequence.pulse_timesteps)
    assert set(sequence.action_ids).isdisjoint(item.action_id for item in original.scheduled_actions)
    assert all(isinstance(scheduled_by_id[action_id], SingleQubitGate) for action_id in sequence.action_ids)
    actions = output.schedule.to_dict()["actions"]
    assert isinstance(actions, list)
    assert all(isinstance(item, dict) and "purpose" not in item for item in actions)


def test_report_identity_distinguishes_equal_local_and_algorithmic_gates() -> None:
    """Classify equal rotations without encoding DD metadata in the schedule."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    schedule = schedule_from_path(
        [Rx(ion=0, theta=math.pi), AdvanceTime(), Rx(ion=0, theta=math.pi), AdvanceTime()],
        create_initial_state(1, architecture),
        architecture,
    )
    local_pulse_action_ids = frozenset({schedule.scheduled_actions[0].action_id})

    events = framed_action_events(
        schedule,
        architecture,
        build_timeline(schedule, architecture),
        local_pulse_action_ids,
    )

    assert [event.kind for event in events] == ["local_dd_pulse", "algorithmic_gate"]


def test_dd_analysis_is_invariant_under_absolute_time_translation() -> None:
    """Preserve physical analysis while translating every reported timestamp."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        field_profile=FieldProfile(1, ((0, 1.0),)),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    base = schedule_from_path(
        [
            AdvanceTime(),
            GlobalGate(gate_name="rx", theta=math.pi, ions=(0,)),
            AdvanceTime(),
            AdvanceTime(),
            GlobalGate(gate_name="rx", theta=math.pi, ions=(0,)),
            AdvanceTime(),
        ],
        create_initial_state(1, architecture),
        architecture,
    )
    offset = 3
    shifted = _shift_schedule(base, offset)
    base_timeline = build_timeline(base, architecture)
    shifted_timeline = build_timeline(shifted, architecture)

    assert shifted.start_time == offset
    assert shifted.end_time == base.end_time + offset
    assert shifted.duration == base.duration
    assert find_idle_windows(shifted_timeline, 0) == tuple(
        (start + offset, end + offset) for start, end in find_idle_windows(base_timeline, 0)
    )
    assert residual_phase_by_ion(shifted, architecture) == residual_phase_by_ion(base, architecture)

    base_segments = compute_critical_segments(base, architecture).segments
    shifted_segments = compute_critical_segments(shifted, architecture).segments
    assert [
        replace(segment, start=segment.start - offset, end=segment.end - offset) for segment in shifted_segments
    ] == list(base_segments)

    base_events = framed_action_events(base, architecture)
    shifted_events = framed_action_events(shifted, architecture)
    assert [replace(event, timestep=event.timestep - offset) for event in shifted_events] == list(base_events)
    shifted_history = build_frame_history(shifted_timeline)
    assert shifted_history.start_time == offset
    assert shifted_history.frame_for_ion(0, offset + 2) == build_frame_history(base_timeline).frame_for_ion(0, 2)
    with pytest.raises(ValueError, match=r"within \[3, 7\]"):
        shifted_history.frame_for_ion(0, 2)

    whole_schedule = LocalDDSequence(0, (offset, offset + shifted.duration), "test", (), ())
    assert decoupling_ratio(shifted, (whole_schedule,)) == pytest.approx(1.0)


def test_dd_passes_translate_pulse_times_from_the_schedule_start() -> None:
    """Anchor local and global pulse grids to the absolute schedule start."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        field_profile=FieldProfile(1, ((0, 1.0),)),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    base = schedule_from_path(
        [AdvanceTime() for _ in range(8)],
        create_initial_state(1, architecture),
        architecture,
    )
    offset = 3
    shifted = _shift_schedule(base, offset)

    base_idealized = apply_idealized_hahn(base, architecture, IdealizedHahnConfig(min_idle_timesteps=2))
    shifted_idealized = apply_idealized_hahn(shifted, architecture, IdealizedHahnConfig(min_idle_timesteps=2))
    assert shifted_idealized.report.sequences[0].window == (offset, base.end_time + offset)
    assert shifted_idealized.report.sequences[0].pulse_timesteps == tuple(
        timestep + offset for timestep in base_idealized.report.sequences[0].pulse_timesteps
    )

    base_nearest = run_nearest_hahn(base, architecture, NearestHahnConfig(min_idle_timesteps=2))
    shifted_nearest = run_nearest_hahn(shifted, architecture, NearestHahnConfig(min_idle_timesteps=2))
    assert shifted_nearest.report.sequences[0].pulse_timesteps == tuple(
        timestep + offset for timestep in base_nearest.report.sequences[0].pulse_timesteps
    )

    base_global = apply_periodic_global_dd(base, architecture, GlobalDDConfig(spacing=4))
    shifted_global = apply_periodic_global_dd(shifted, architecture, GlobalDDConfig(spacing=4))
    assert shifted_global.report.pulse_timesteps == tuple(
        timestep + offset for timestep in base_global.report.pulse_timesteps
    )
    assert shifted_global.report.phase_cost == pytest.approx(base_global.report.phase_cost)


def _shift_schedule(schedule: Schedule, offset: int) -> Schedule:
    initial_state = replace(
        schedule.initial_state,
        time=schedule.start_time + offset,
        ions_busy_until=tuple((ion, timestep + offset) for ion, timestep in schedule.initial_state.ions_busy_until),
        pzs_busy_until=tuple((zone, timestep + offset) for zone, timestep in schedule.initial_state.pzs_busy_until),
    )
    return Schedule(
        scheduled_actions=tuple(
            replace(scheduled_action, start_time=scheduled_action.start_time + offset)
            for scheduled_action in schedule.scheduled_actions
        ),
        end_time=schedule.end_time + offset,
        initial_state=initial_state,
    )
