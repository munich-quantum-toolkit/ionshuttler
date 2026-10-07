# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Rebuild and validate schedules after control transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import (
    Action,
    Rx,
    Ry,
    Rz,
)
from mqt.ionshuttler.linear.replay import is_schedule_valid
from mqt.ionshuttler.linear.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.linear.timeline import CompiledTimeline, build_timeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.linear.architecture import LinearArchitecture
    from mqt.ionshuttler.linear.dd.schemes import GateSpec


def local_gate_for_spec(spec: GateSpec, ion: int) -> Action:
    """Build the ion-local rotation described by a pulse specification.

    Args:
        spec: Gate name and rotation angle to realize.
        ion: Ion the rotation acts on.

    Returns:
        The concrete single-qubit gate action.

    Raises:
        ValueError: If the specification lacks an angle or names an unsupported gate.
    """
    if spec.theta is None:
        msg = f"gate specification for {spec.gate_name} requires theta"
        raise ValueError(msg)
    gate_types: dict[str, type[Rx | Ry | Rz]] = {"Rx": Rx, "Ry": Ry, "Rz": Rz}
    try:
        gate_type = gate_types[spec.gate_name]
    except KeyError as error:
        msg = f"unsupported local DD gate: {spec.gate_name!r}"
        raise ValueError(msg) from error
    return gate_type(ion=ion, theta=spec.theta)


def insert_action_at_time(
    schedule: Schedule,
    architecture: LinearArchitecture,
    timestep: int,
    action: Action,
    *,
    timeline: CompiledTimeline | None = None,
) -> Schedule:
    """Insert an action immediately before a boundary's time advance.

    Args:
        schedule: Schedule to transform.
        architecture: Hardware model used to validate the insertion.
        timestep: Boundary at which the action starts.
        action: Action to insert.
        timeline: Existing timeline of ``schedule``, if already available.

    Returns:
        A rebuilt schedule containing the inserted action.

    Raises:
        ValueError: If the timestep or action is invalid.
    """
    if not schedule.start_time <= timestep <= schedule.end_time:
        msg = f"timestep must be within [{schedule.start_time}, {schedule.end_time}]"
        raise ValueError(msg)
    resolved_timeline = build_timeline(schedule, architecture) if timeline is None else timeline
    if not architecture.is_action_valid(resolved_timeline.state_at(timestep), action):
        msg = "action is not valid at the requested timestep"
        raise ValueError(msg)
    insert_index = _path_insert_index(schedule.scheduled_actions, timestep)
    duration = architecture.action_duration(action)
    inserted = ScheduledAction(
        schedule.next_action_id,
        action,
        start_time=timestep,
        duration=duration,
        processing_zone_id=architecture.action_processing_zone(action, resolved_timeline.state_at(timestep)),
    )
    rebuilt = rebuild_schedule(
        schedule,
        (*schedule.scheduled_actions[:insert_index], inserted, *schedule.scheduled_actions[insert_index:]),
    )
    if not is_schedule_valid(rebuilt, architecture):
        msg = "inserted action conflicts with the rebuilt schedule"
        raise ValueError(msg)
    return rebuilt


def rebuild_schedule(
    original_schedule: Schedule,
    scheduled_actions: Sequence[ScheduledAction],
) -> Schedule:
    """Rebuild a schedule while preserving its initial state and action identity.

    Args:
        original_schedule: Schedule whose initial state is preserved.
        scheduled_actions: Complete replacement ordered schedule.

    Returns:
        A new immutable schedule.
    """
    actions = tuple(scheduled_actions)
    end_time = max([original_schedule.end_time, *(item.end_time for item in actions)])
    return Schedule(
        scheduled_actions=actions,
        end_time=end_time,
        initial_state=original_schedule.initial_state,
    )


def _path_insert_index(path: Sequence[ScheduledAction], target_time: int) -> int:
    return next((index for index, item in enumerate(path) if item.start_time > target_time), len(path))


__all__ = [
    "insert_action_at_time",
    "local_gate_for_spec",
    "rebuild_schedule",
]
