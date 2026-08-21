# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for rebuilding and validating transformed DD schedules."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import AdvanceTime, GateSpec, GlobalPulse, Rx, Rz, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.schedule_transform import (
    insert_action_at_time,
    rebuild_schedule,
    validate_rebuilt_schedule,
)
from mqt.ionshuttler.linear.schedule import ActionSchedule, ScheduledAction
from mqt.ionshuttler.linear.state import create_initial_state


def _base_program() -> tuple[ActionSchedule, Architecture]:
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1, 2]})
    return (
        ActionSchedule.from_actions(
            [Shuttle(ion=0, src=0, dst=1), AdvanceTime(), AdvanceTime()],
            create_initial_state(1, architecture, initial_positions=[0]),
        ),
        architecture,
    )


def test_insert_action_preserves_existing_identity_without_mutating_source() -> None:
    """Keep existing IDs and metadata while assigning one fresh pulse ID."""
    original, architecture = _base_program()
    inserted = Rx(ion=0, theta=pi)

    rebuilt = insert_action_at_time(original, architecture, 1, inserted)

    assert original.path == (Shuttle(ion=0, src=0, dst=1), AdvanceTime(), AdvanceTime())
    assert rebuilt.path == (Shuttle(ion=0, src=0, dst=1), AdvanceTime(), inserted, AdvanceTime())
    assert tuple(item.action_id for item in rebuilt.scheduled_actions) == (0, 1, 3, 2)
    assert ActionSchedule.from_json(rebuilt.to_json()) == rebuilt


def test_rebuild_program_reconstructs_makespan_and_preserves_metadata() -> None:
    """Derive the makespan from replacement scheduled actions."""
    original, architecture = _base_program()
    replacement = (
        original.scheduled_actions[0],
        original.scheduled_actions[1],
        ScheduledAction(3, Shuttle(ion=0, src=1, dst=2)),
        original.scheduled_actions[2],
    )

    rebuilt = rebuild_schedule(original, replacement)

    assert rebuilt.num_timesteps == 2
    assert rebuilt.initial_state == original.initial_state
    assert validate_rebuilt_schedule(rebuilt, architecture)


def test_schedule_validation_accepts_concurrent_and_terminal_actions() -> None:
    """Accept a valid transport layer, destination gate, and terminal pulses."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1, 2]})
    program = ActionSchedule.from_actions(
        [
            Shuttle(ion=0, src=0, dst=1),
            Rx(ion=1, theta=pi),
            AdvanceTime(),
            GlobalPulse(GateSpec("Rx", pi)),
            Rz(ion=0, theta=0.2),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
    )

    assert validate_rebuilt_schedule(program, architecture)


def test_schedule_validation_rejects_conflicts_and_physical_terminal_gate() -> None:
    """Reject colliding transport and unfinished physical resource use."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [0, 1, 2]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 2])
    conflict = ActionSchedule.from_actions(
        [Shuttle(ion=0, src=0, dst=1), Shuttle(ion=1, src=2, dst=1), AdvanceTime()],
        initial_state,
    )
    terminal = ActionSchedule.from_actions([Rx(ion=0, theta=pi)], initial_state)

    assert not validate_rebuilt_schedule(conflict, architecture)
    assert not validate_rebuilt_schedule(terminal, architecture)


def test_transform_rejects_invalid_time_and_action() -> None:
    """Report invalid transform requests without mutating the program."""
    base, architecture = _base_program()
    with pytest.raises(ValueError, match="within"):
        insert_action_at_time(base, architecture, 3, Rz(ion=0, theta=0.2))
    with pytest.raises(ValueError, match="not valid"):
        insert_action_at_time(base, architecture, 0, Rx(ion=0, theta=pi))
