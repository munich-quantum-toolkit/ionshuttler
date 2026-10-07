# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for rebuilding and validating transformed DD schedules."""

from __future__ import annotations

from dataclasses import replace
from math import pi

import pytest

from mqt.ionshuttler.linear import GateTiming, schedule_from_json
from mqt.ionshuttler.linear.actions import (
    DEFAULT_ACTION_TYPES,
    GlobalGate,
    PhysicalSwap,
    Rx,
    Ry,
    Rz,
    Shuttle,
)
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.dd import GateSpec
from mqt.ionshuttler.linear.dd.schedule_transform import (
    insert_action_at_time,
    local_gate_for_spec,
    rebuild_schedule,
)
from mqt.ionshuttler.linear.replay import is_schedule_valid, replay_schedule
from mqt.ionshuttler.linear.schedule import Schedule, ScheduledAction, schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state


def _base_program() -> tuple[Schedule, LinearArchitecture]:
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1, 2]})
    return (
        schedule_from_path(
            [Shuttle(ion=0, src=0, dst=1), AdvanceTime(), AdvanceTime()],
            create_initial_state(1, architecture, initial_positions=[0]),
            architecture,
        ),
        architecture,
    )


def test_local_gate_for_spec_builds_the_local_rotation() -> None:
    """Realize a decoupling pulse specification as a gate on one ion."""
    assert local_gate_for_spec(GateSpec("Ry", pi), ion=2) == Ry(ion=2, theta=pi)
    with pytest.raises(ValueError, match="unsupported local DD gate"):
        local_gate_for_spec(GateSpec("Rzz", pi), ion=2)


def test_insert_action_preserves_existing_identity_without_mutating_source() -> None:
    """Keep existing IDs and metadata while assigning one fresh pulse ID."""
    original, architecture = _base_program()
    inserted = Rx(ion=0, theta=pi)

    rebuilt = insert_action_at_time(original, architecture, 1, inserted)

    assert original.path == (Shuttle(ion=0, src=0, dst=1),)
    assert rebuilt.path == (Shuttle(ion=0, src=0, dst=1), inserted)
    assert tuple(item.action_id for item in rebuilt.scheduled_actions) == (0, 1)
    assert schedule_from_json(rebuilt.to_json()) == rebuilt


def test_rebuild_program_reconstructs_makespan_and_preserves_metadata() -> None:
    """Derive the makespan from replacement scheduled actions."""
    original, architecture = _base_program()
    replacement = (
        original.scheduled_actions[0],
        ScheduledAction(1, Shuttle(ion=0, src=1, dst=2), start_time=1, duration=1),
    )

    rebuilt = rebuild_schedule(original, replacement)

    assert rebuilt.end_time == 2
    assert rebuilt.initial_state == original.initial_state
    assert is_schedule_valid(rebuilt, architecture)


def test_schedule_validation_accepts_concurrent_and_terminal_actions() -> None:
    """Accept a valid transport layer, destination gate, and terminal pulses."""
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [1, 2]},
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    program = schedule_from_path(
        [
            Shuttle(ion=0, src=0, dst=1),
            Rx(ion=1, theta=pi),
            AdvanceTime(),
            GlobalGate(gate_name="rx", theta=pi, ions=(0, 1)),
            Rz(ion=0, theta=0.2),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
        architecture,
    )

    assert is_schedule_valid(program, architecture)


def test_schedule_validation_rejects_conflicts_and_accepts_complete_terminal_gate() -> None:
    """Reject colliding transport and include terminal gate duration in the makespan."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [0, 1, 2]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 2])
    conflict = schedule_from_path(
        [Shuttle(ion=0, src=0, dst=1), Shuttle(ion=1, src=2, dst=1), AdvanceTime()],
        initial_state,
        architecture,
    )
    terminal = schedule_from_path([Rx(ion=0, theta=pi)], initial_state, architecture)

    assert not is_schedule_valid(conflict, architecture)
    assert is_schedule_valid(terminal, architecture)


def test_schedule_compatibility_rejects_out_of_range_initial_positions() -> None:
    """Reject a schedule whose initial ion positions fall outside the architecture."""
    source_architecture = LinearArchitecture(num_sites=3)
    program = schedule_from_path(
        [AdvanceTime()],
        create_initial_state(1, source_architecture, initial_positions=[2]),
        source_architecture,
    )
    small_architecture = LinearArchitecture(num_sites=2)

    with pytest.raises(ValueError, match="initial positions"):
        replay_schedule(program, small_architecture)


def test_schedule_compatibility_rejects_mismatched_processing_zones() -> None:
    """Reject a schedule built against a different set of processing zones."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1, 2]})
    program = schedule_from_path(
        [AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )
    other_architecture = LinearArchitecture(num_sites=3, processing_zones={"other_pz": [1, 2]})

    with pytest.raises(ValueError, match="processing-zone resources"):
        replay_schedule(program, other_architecture)


def test_schedule_compatibility_rejects_unsupported_action_types() -> None:
    """Reject a schedule using an action class the architecture does not support."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1, 2]})
    program = schedule_from_path(
        [Rx(ion=0, theta=pi), AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )
    restricted_architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [1, 2]},
        supported_action_types=(PhysicalSwap, Shuttle),
    )

    with pytest.raises(ValueError, match="unsupported by the architecture"):
        replay_schedule(program, restricted_architecture)


def test_schedule_compatibility_rejects_failed_replay() -> None:
    """Reject a schedule that cannot be replayed against the architecture."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1, 2]})
    terminal = schedule_from_path(
        [Rx(ion=0, theta=pi)],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )

    with pytest.raises(ValueError, match="not valid"):
        replay_schedule(terminal, architecture)


def test_transform_rejects_invalid_time_and_action() -> None:
    """Report invalid transform requests without mutating the program."""
    base, architecture = _base_program()
    with pytest.raises(ValueError, match="within"):
        insert_action_at_time(base, architecture, 3, Rz(ion=0, theta=0.2))
    with pytest.raises(ValueError, match="not valid"):
        insert_action_at_time(base, architecture, 0, Rx(ion=0, theta=pi))


def test_transform_rejects_a_time_before_the_initial_state() -> None:
    """Insert actions only within the interval that the schedule describes."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    initial_state = replace(create_initial_state(1, architecture), time=2)
    program = schedule_from_path([AdvanceTime(), AdvanceTime()], initial_state, architecture)

    with pytest.raises(ValueError, match=r"within \[2, 4\]"):
        insert_action_at_time(program, architecture, 1, Rz(ion=0, theta=0.2))


def test_transform_rejects_an_action_that_overlaps_a_later_action() -> None:
    """Validate interactions that an action-local precheck cannot detect."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        gate_timing=GateTiming(rx=2),
    )
    program = schedule_from_path(
        [AdvanceTime(), Rx(ion=0, theta=0.2), AdvanceTime(), AdvanceTime()],
        create_initial_state(1, architecture),
        architecture,
    )

    with pytest.raises(ValueError, match="conflicts with the rebuilt schedule"):
        insert_action_at_time(program, architecture, 0, Rx(ion=0, theta=pi))


def test_transform_appends_an_action_after_a_zero_duration_gate_at_the_same_time() -> None:
    """Start a transport after a zero-duration gate on the same ion finishes."""
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [0, 1, 2]},
        gate_timing=GateTiming(rx=0, virtual_single_qubit_gates=frozenset()),
    )
    program = schedule_from_path(
        [Rx(ion=0, theta=0.2), AdvanceTime()],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
        architecture,
    )

    rebuilt = insert_action_at_time(program, architecture, 0, Shuttle(ion=0, src=0, dst=1))

    assert rebuilt.path == (Rx(ion=0, theta=0.2), Shuttle(ion=0, src=0, dst=1))
