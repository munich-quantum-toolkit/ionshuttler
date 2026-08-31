# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the schedule-to-DD ownership contract."""

from __future__ import annotations

import math

from mqt.ionshuttler.linear.actions import AdvanceTime, Rx, SingleQubitGate
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd import apply_idealized_hahn
from mqt.ionshuttler.linear.dd.frame_replay import framed_action_events
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state


def test_local_pulse_identity_is_owned_by_the_dd_report() -> None:
    """Keep local-pulse provenance out of the hardware-facing schedule."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    original = ActionSchedule.from_actions(
        [AdvanceTime() for _ in range(4)],
        create_initial_state(1, architecture),
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
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    schedule = ActionSchedule.from_actions(
        [Rx(ion=0, theta=math.pi), Rx(ion=0, theta=math.pi), AdvanceTime()],
        create_initial_state(1, architecture),
    )
    local_pulse_action_ids = frozenset({schedule.scheduled_actions[0].action_id})

    events = framed_action_events(
        schedule,
        architecture,
        build_timeline(schedule, architecture),
        local_pulse_action_ids,
    )

    assert [event.kind for event in events] == ["local_dd_pulse", "algorithmic_gate", "advance_time"]
