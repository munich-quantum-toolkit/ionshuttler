# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for dynamical-decoupling timeline reconstruction."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import pytest

from mqt.ionshuttler.core.gates import GlobalGate, Rz, Rzz
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.linear import GateTiming, TransportTiming
from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES, PhysicalSwap, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.dd.schedule_transform import insert_action_at_time
from mqt.ionshuttler.linear.schedule import schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, State, create_initial_state
from mqt.ionshuttler.linear.timeline import build_timeline

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.state import SearchTransition


def test_timeline_reconstructs_positions_resources_and_action_order() -> None:
    """Match source schedule-boundary and duration-occupancy semantics."""
    architecture = LinearArchitecture(
        num_sites=4,
        processing_zones={"pz": [1, 2]},
        transport_timing=TransportTiming(swap=1),
    )
    program = schedule_from_path(
        [
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Rzz(ion_a=0, ion_b=1, theta=0.5),
            AdvanceTime(),
            AdvanceTime(),
            PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2),
            AdvanceTime(),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
        architecture,
    )

    timeline = build_timeline(program, architecture)

    assert [timeline.ion_position(0, timestep) for timestep in range(5)] == [1, 1, 1, 2, 2]
    assert [timeline.ion_position(1, timestep) for timestep in range(5)] == [2, 2, 2, 1, 1]
    assert {timestep for timestep in range(5) if timeline.ion_busy(0, timestep)} == {0, 1, 2, 3}
    assert {timestep for timestep in range(5) if timeline.ion_busy(1, timestep)} == {1, 2, 3}
    assert {timestep for timestep in range(5) if timeline.pz_busy("pz", timestep)} == {1, 2}
    assert timeline.action_at(1) == (Rzz(ion_a=0, ion_b=1, theta=0.5),)


def test_timeline_preserves_same_boundary_and_terminal_action_order() -> None:
    """Keep global pulses and terminal virtual gates in path order."""
    architecture = LinearArchitecture(
        num_sites=2,
        processing_zones={"pz": [0, 1]},
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    pulse = GlobalGate(gate_name="rx", theta=pi, ions=(0,))
    terminal = Rz(ion=0, theta=0.25)
    program = schedule_from_path(
        [pulse, Shuttle(ion=0, src=0, dst=1), AdvanceTime(), terminal],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )

    timeline = build_timeline(program, architecture)

    assert timeline.action_at(0) == (pulse, Shuttle(ion=0, src=0, dst=1))
    assert timeline.action_at(1) == (terminal,)
    assert timeline.ion_position(0, 0) == 1
    assert timeline.ion_position(0, 1) == 1


def test_timeline_materializes_the_last_position_checkpoint_at_each_boundary() -> None:
    """Retain all simultaneous transport updates when materializing positions."""
    architecture = LinearArchitecture(num_sites=4)
    program = schedule_from_path(
        [
            Shuttle(ion=0, src=0, dst=1),
            Shuttle(ion=1, src=2, dst=3),
            AdvanceTime(),
            AdvanceTime(),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
        architecture,
    )

    timeline = build_timeline(program, architecture)

    assert [timeline.ion_position(0, timestep) for timestep in range(3)] == [1, 1, 1]
    assert [timeline.ion_position(1, timestep) for timestep in range(3)] == [3, 3, 3]


def test_incremental_gate_timeline_matches_a_full_rebuild() -> None:
    """Keep resources, ordering, and stable identities coherent after a local patch."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        gate_timing=GateTiming(rz=1, virtual_single_qubit_gates=frozenset()),
    )
    program = schedule_from_path(
        [AdvanceTime(), AdvanceTime()],
        create_initial_state(1, architecture),
        architecture,
    )
    original_timeline = build_timeline(program, architecture)
    gate = Rz(ion=0, theta=pi)
    inserted = ScheduledAction(program.next_action_id, gate, start_time=0, duration=1, processing_zone_id="pz")

    incremental = original_timeline.with_inserted_single_qubit_gate(
        inserted,
        "pz",
        0,
    )
    rebuilt_schedule = insert_action_at_time(program, architecture, 0, gate)
    rebuilt = build_timeline(rebuilt_schedule, architecture)

    for timestep in range(program.end_time + 1):
        assert incremental.action_at(timestep) == rebuilt.action_at(timestep)
        assert incremental.scheduled_action_at(timestep) == rebuilt.scheduled_action_at(timestep)
        assert incremental.ion_position(0, timestep) == rebuilt.ion_position(0, timestep)
        assert incremental.ion_busy(0, timestep) == rebuilt.ion_busy(0, timestep)
        assert incremental.ion_gate_busy(0, timestep) == rebuilt.ion_gate_busy(0, timestep)
        assert incremental.pz_busy("pz", timestep) == rebuilt.pz_busy("pz", timestep)
    assert not original_timeline.ion_busy(0, 0)


def test_timeline_distinguishes_virtual_and_physical_rz_resources() -> None:
    """Treat only virtual rotations as resource-free in the Linear model."""
    virtual_architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    physical_architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        gate_timing=GateTiming(rz=1, virtual_single_qubit_gates=frozenset()),
    )
    virtual = schedule_from_path(
        [Rz(ion=0, theta=0.2), AdvanceTime()],
        create_initial_state(1, virtual_architecture),
        virtual_architecture,
    )
    physical = schedule_from_path(
        [Rz(ion=0, theta=0.2), AdvanceTime()],
        create_initial_state(1, physical_architecture),
        physical_architecture,
    )

    assert not build_timeline(virtual, virtual_architecture).ion_gate_busy(0, 0)
    assert build_timeline(physical, physical_architecture).ion_gate_busy(0, 0)
    assert build_timeline(physical, physical_architecture).pz_busy("pz", 0)


def test_timeline_keeps_initial_occupancy_at_absolute_times() -> None:
    """Carry unfinished initial-state reservations at their schedule times."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"active": [0], "free": [1]})
    initial_state = State(
        positions=((0, 0), (1, 1)),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=((0, 7), (1, 5)),
        pzs_busy_until=(("active", 8), ("free", 5)),
        time=5,
    )
    program = schedule_from_path(
        [Shuttle(ion=1, src=1, dst=2), *(AdvanceTime() for _ in range(4))],
        initial_state,
        architecture,
    )

    timeline = build_timeline(program, architecture)
    times = range(5, 10)

    assert timeline.start_time == 5
    assert {timestep for timestep in times if timeline.ion_busy(0, timestep)} == {5, 6}
    assert {timestep for timestep in times if timeline.ion_busy(1, timestep)} == {5}
    assert {timestep for timestep in times if timeline.pz_busy("active", timestep)} == {5, 6, 7}
    assert not any(timeline.pz_busy("free", timestep) for timestep in times)
    assert [timeline.ion_position(1, timestep) for timestep in times] == [2, 2, 2, 2, 2]
    assert timeline.action_at(5) == (Shuttle(ion=1, src=1, dst=2),)
    assert timeline.state_at(5).time == 5


def test_timeline_rejects_queries_before_the_initial_state() -> None:
    """Describe no boundary before the schedule's initial state time."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = State(
        positions=((0, 0),),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=((0, 3),),
        pzs_busy_until=architecture.initial_pzs_busy_until(),
        time=3,
    )
    program = schedule_from_path([AdvanceTime()], initial_state, architecture)

    timeline = build_timeline(program, architecture)

    with pytest.raises(ValueError, match=r"within \[3, 4\]"):
        timeline.ion_busy(0, 2)
    with pytest.raises(ValueError, match=r"within \[3, 4\]"):
        timeline.ion_position(0, 2)
    with pytest.raises(ValueError, match=r"within \[3, 4\]"):
        timeline.state_at(2)


@pytest.mark.parametrize(
    ("end_time", "path", "message"),
    [
        (-1, [], "non-negative"),
        (0, [Shuttle(ion=0, src=0, dst=1)], "within end_time"),
    ],
)
def test_schedule_rejects_malformed_makespans(
    end_time: int,
    path: list[SearchTransition],
    message: str,
) -> None:
    """Reject negative and overlong schedule clocks at construction."""
    architecture = LinearArchitecture(num_sites=2)
    valid = schedule_from_path(path, create_initial_state(1, architecture), architecture)

    with pytest.raises(ValueError, match=message):
        Schedule(
            scheduled_actions=valid.scheduled_actions,
            end_time=end_time,
            initial_state=valid.initial_state,
        )


def test_timeline_rejects_invalid_query_boundaries() -> None:
    """Report out-of-range and non-integer boundary queries clearly."""
    architecture = LinearArchitecture(num_sites=1)
    program = schedule_from_path(
        [],
        State(
            positions=((0, 0),),
            completed_gates=frozenset(),
            in_progress_gates=(),
            ions_busy_until=((0, 0),),
            pzs_busy_until=architecture.initial_pzs_busy_until(),
            time=0,
        ),
        architecture,
    )
    timeline = build_timeline(program, architecture)
    invalid_timestep: int = True
    with pytest.raises(ValueError, match="within"):
        timeline.state_at(1)
    with pytest.raises(TypeError, match="integer"):
        timeline.action_at(invalid_timestep)
