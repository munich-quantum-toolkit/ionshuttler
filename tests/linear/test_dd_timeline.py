# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for dynamical-decoupling timeline reconstruction."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, GlobalPulse, PhysicalSwap, Rz, Rzz, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.schedule_transform import insert_action_at_time
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.schedule import ActionSchedule, ScheduledAction
from mqt.ionshuttler.linear.state import State, create_initial_state


def test_timeline_reconstructs_positions_resources_and_action_order() -> None:
    """Match source schedule-boundary and duration-occupancy semantics."""
    architecture = Architecture(num_sites=4, processing_zones={"pz": [1, 2]})
    program = ActionSchedule.from_actions(
        [
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Rzz(ion_a=0, ion_b=1, theta=0.5, duration=2),
            AdvanceTime(),
            AdvanceTime(),
            PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2),
            AdvanceTime(),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
    )

    timeline = build_timeline(program, architecture)

    assert [timeline.ion_position(0, timestep) for timestep in range(5)] == [1, 1, 1, 2, 2]
    assert [timeline.ion_position(1, timestep) for timestep in range(5)] == [2, 2, 2, 1, 1]
    assert {timestep for timestep in range(5) if timeline.ion_busy(0, timestep)} == {0, 1, 2, 3}
    assert {timestep for timestep in range(5) if timeline.ion_busy(1, timestep)} == {1, 2, 3}
    assert {timestep for timestep in range(5) if timeline.pz_busy("pz", timestep)} == {1, 2}
    assert timeline.action_at(1) == (Rzz(ion_a=0, ion_b=1, theta=0.5, duration=2), AdvanceTime())


def test_timeline_preserves_same_boundary_and_terminal_action_order() -> None:
    """Keep global pulses and terminal virtual gates in path order."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    pulse = GlobalPulse(gate=GateSpec("Rx", theta=pi))
    terminal = Rz(ion=0, theta=0.25)
    program = ActionSchedule.from_actions(
        [pulse, Shuttle(ion=0, src=0, dst=1), AdvanceTime(), terminal],
        create_initial_state(1, architecture, initial_positions=[0]),
    )

    timeline = build_timeline(program, architecture)

    assert timeline.action_at(0) == (pulse, Shuttle(ion=0, src=0, dst=1), AdvanceTime())
    assert timeline.action_at(1) == (terminal,)
    assert timeline.ion_position(0, 0) == 1
    assert timeline.ion_position(0, 1) == 1


def test_timeline_materializes_the_last_position_checkpoint_at_each_boundary() -> None:
    """Retain all simultaneous transport updates when materializing positions."""
    architecture = Architecture(num_sites=4)
    program = ActionSchedule.from_actions(
        [
            Shuttle(ion=0, src=0, dst=1),
            Shuttle(ion=1, src=2, dst=3),
            AdvanceTime(),
            AdvanceTime(),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
    )

    timeline = build_timeline(program, architecture)

    assert [timeline.ion_position(0, timestep) for timestep in range(3)] == [1, 1, 1]
    assert [timeline.ion_position(1, timestep) for timestep in range(3)] == [3, 3, 3]


def test_incremental_gate_timeline_matches_a_full_rebuild() -> None:
    """Keep resources, ordering, and stable identities coherent after a local patch."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    program = ActionSchedule.from_actions(
        [Rz(ion=0, theta=0.25), AdvanceTime(), AdvanceTime()],
        create_initial_state(1, architecture),
    )
    original_timeline = build_timeline(program, architecture)
    gate = Rz(ion=0, theta=pi, duration=1, virtual=False)
    inserted = ScheduledAction(program.next_action_id, gate)

    incremental = original_timeline.with_inserted_single_qubit_gate(
        inserted,
        "pz",
        0,
    )
    rebuilt_schedule = insert_action_at_time(program, architecture, 0, gate)
    rebuilt = build_timeline(rebuilt_schedule, architecture)

    for timestep in range(program.num_timesteps + 1):
        assert incremental.action_at(timestep) == rebuilt.action_at(timestep)
        assert incremental.scheduled_action_at(timestep) == rebuilt.scheduled_action_at(timestep)
        assert incremental.ion_position(0, timestep) == rebuilt.ion_position(0, timestep)
        assert incremental.ion_busy(0, timestep) == rebuilt.ion_busy(0, timestep)
        assert incremental.ion_gate_busy(0, timestep) == rebuilt.ion_gate_busy(0, timestep)
        assert incremental.pz_busy("pz", timestep) == rebuilt.pz_busy("pz", timestep)
    assert not original_timeline.ion_busy(0, 0)


def test_timeline_distinguishes_virtual_and_physical_rz_resources() -> None:
    """Treat only virtual rotations as resource-free in the Linear model."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    initial_state = create_initial_state(1, architecture)
    virtual = ActionSchedule.from_actions(
        [Rz(ion=0, theta=0.2), AdvanceTime()],
        initial_state,
    )
    physical = ActionSchedule.from_actions(
        [Rz(ion=0, theta=0.2, duration=1, virtual=False), AdvanceTime()],
        initial_state,
    )

    assert not build_timeline(virtual, architecture).ion_gate_busy(0, 0)
    assert build_timeline(physical, architecture).ion_gate_busy(0, 0)
    assert build_timeline(physical, architecture).pz_busy("pz", 0)


def test_timeline_seeds_relative_occupancy_from_initial_availability() -> None:
    """Carry unfinished entry-state resource reservations into replay time."""
    architecture = Architecture(num_sites=2, processing_zones={"active": [0], "free": [1]})
    initial_state = State(
        positions=((0, 0), (1, 1)),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=((0, 7), (1, 5)),
        pzs_busy_until=(("active", 8), ("free", 5)),
        time=5,
    )
    program = ActionSchedule.from_actions([AdvanceTime() for _ in range(4)], initial_state)

    timeline = build_timeline(program, architecture)

    assert {timestep for timestep in range(5) if timeline.ion_busy(0, timestep)} == {0, 1}
    assert not any(timeline.ion_busy(1, timestep) for timestep in range(5))
    assert {timestep for timestep in range(5) if timeline.pz_busy("active", timestep)} == {0, 1, 2}
    assert not any(timeline.pz_busy("free", timestep) for timestep in range(5))


@pytest.mark.parametrize(
    ("num_timesteps", "path", "message"),
    [
        (-1, [], "non-negative"),
        (0, [AdvanceTime(), AdvanceTime()], "total time advancement"),
        (2, [AdvanceTime()], "total time advancement"),
    ],
)
def test_action_schedule_rejects_malformed_makespans(
    num_timesteps: int,
    path: list[Action],
    message: str,
) -> None:
    """Reject negative, overlong, and underlong schedule clocks at construction."""
    architecture = Architecture(num_sites=1)
    valid = ActionSchedule.from_actions(path, create_initial_state(1, architecture))

    with pytest.raises(ValueError, match=message):
        ActionSchedule(
            scheduled_actions=valid.scheduled_actions,
            num_timesteps=num_timesteps,
            initial_state=valid.initial_state,
        )


def test_timeline_rejects_invalid_query_boundaries() -> None:
    """Report out-of-range and non-integer boundary queries clearly."""
    architecture = Architecture(num_sites=1)
    program = ActionSchedule.from_actions(
        [],
        State(
            positions=((0, 0),),
            completed_gates=frozenset(),
            in_progress_gates=(),
            ions_busy_until=((0, 0),),
            pzs_busy_until=architecture.initial_pzs_busy_until(),
            time=0,
        ),
    )
    timeline = build_timeline(program, architecture)
    invalid_timestep: int = True
    with pytest.raises(ValueError, match="within"):
        timeline.state_at(1)
    with pytest.raises(TypeError, match="integer"):
        timeline.action_at(invalid_timestep)
