# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for replaying explicit Linear schedules."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mqt.ionshuttler.linear import GateTiming
from mqt.ionshuttler.linear.actions import Rx, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.replay import is_schedule_valid, replay_schedule
from mqt.ionshuttler.linear.schedule import LinearMachineState, Schedule, ScheduledAction, schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state


def _zero_duration_rx_architecture() -> LinearArchitecture:
    return LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [0]},
        gate_timing=GateTiming(rx=0, virtual_single_qubit_gates=frozenset()),
    )


def test_replay_runs_a_zero_duration_gate_before_a_transport_of_its_ion() -> None:
    """Apply same-time actions in stored order, as the search path did."""
    architecture = _zero_duration_rx_architecture()
    schedule = schedule_from_path(
        [Rx(ion=0, theta=0.2), Shuttle(ion=0, src=0, dst=1), AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )

    final_state = replay_schedule(schedule, architecture)

    assert [item.start_time for item in schedule.scheduled_actions] == [0, 0]
    assert final_state.positions == ((0, 1),)
    assert architecture.replay_schedule(schedule).positions == ((0, 1),)


def test_replay_rejects_a_gate_stored_after_a_transport_of_its_ion() -> None:
    """Keep a transported ion busy for later actions at the same start time."""
    architecture = _zero_duration_rx_architecture()
    initial_state = LinearMachineState.from_compiler_state(
        create_initial_state(1, architecture, initial_positions=[0]),
    )
    schedule = Schedule(
        scheduled_actions=(
            ScheduledAction(0, Shuttle(ion=0, src=0, dst=1), start_time=0, duration=1),
            ScheduledAction(1, Rx(ion=0, theta=0.2), start_time=0, duration=0, processing_zone_id="pz"),
        ),
        end_time=1,
        initial_state=initial_state,
    )

    assert not is_schedule_valid(schedule, architecture)
    with pytest.raises(ValueError, match="is not valid at time 0"):
        replay_schedule(schedule, architecture)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_replay_moves_a_transport_layer_together_in_any_stored_order(order: tuple[int, int]) -> None:
    """Let a shuttle enter a site that another shuttle of its layer leaves."""
    architecture = LinearArchitecture(num_sites=3)
    initial_state = LinearMachineState.from_compiler_state(
        create_initial_state(2, architecture, initial_positions=[0, 1]),
    )
    shuttles = (Shuttle(ion=0, src=0, dst=1), Shuttle(ion=1, src=1, dst=2))
    schedule = Schedule(
        scheduled_actions=tuple(
            ScheduledAction(action_id, shuttles[index], start_time=0, duration=1)
            for action_id, index in enumerate(order)
        ),
        end_time=1,
        initial_state=initial_state,
    )

    assert replay_schedule(schedule, architecture).positions == ((0, 1), (1, 2))


def test_replay_rejects_a_transport_of_an_ion_that_an_earlier_gate_occupies() -> None:
    """Check each transport against the actions stored before it."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [0]})
    initial_state = LinearMachineState.from_compiler_state(
        create_initial_state(1, architecture, initial_positions=[0]),
    )
    schedule = Schedule(
        scheduled_actions=(
            ScheduledAction(0, Rx(ion=0, theta=0.2), start_time=0, duration=1, processing_zone_id="pz"),
            ScheduledAction(1, Shuttle(ion=0, src=0, dst=1), start_time=0, duration=1),
        ),
        end_time=1,
        initial_state=initial_state,
    )

    with pytest.raises(ValueError, match=r"Shuttle.*is not valid at time 0"):
        replay_schedule(schedule, architecture)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"duration": 2}, "scheduled duration"),
        ({"processing_zone_id": None}, "scheduled processing zone"),
    ],
)
def test_replay_rejects_incorrect_execution_metadata(changes: dict[str, object], message: str) -> None:
    """Reject durations and processing zones that disagree with the architecture."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    schedule = schedule_from_path(
        [Rx(ion=0, theta=0.2)],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )
    invalid_item = replace(schedule.scheduled_actions[0], **changes)
    invalid = replace(
        schedule,
        scheduled_actions=(invalid_item,),
        end_time=max(schedule.end_time, invalid_item.end_time),
    )

    with pytest.raises(ValueError, match=message):
        replay_schedule(invalid, architecture)
