# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Rebuild and validate action schedules after control transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from mqt.ionshuttler.linear.actions import (
    Action,
    AdvanceTime,
    GlobalPulse,
    SingleQubitGate,
    TransportAction,
)
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.schedule import ActionSchedule, ScheduledAction
from mqt.ionshuttler.linear.validation import is_action_valid, is_transport_layer_valid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.state import State


class _TimedGate(Protocol):
    """Describe the duration field shared by concrete physical gates."""

    duration: int


def insert_action_at_time(
    schedule: ActionSchedule,
    architecture: Architecture,
    timestep: int,
    action: Action,
) -> ActionSchedule:
    """Insert an action immediately before a boundary's time advance.

    Args:
        schedule: Schedule to transform.
        architecture: Hardware model used to validate the insertion.
        timestep: Boundary at which the action starts.
        action: Action to insert.

    Returns:
        A rebuilt schedule containing the inserted action.

    Raises:
        ValueError: If the timestep or action is invalid.
    """
    if not 0 <= timestep <= schedule.num_timesteps:
        msg = f"timestep must be within [0, {schedule.num_timesteps}]"
        raise ValueError(msg)
    timeline = build_timeline(schedule, architecture)
    if not is_action_valid(timeline.state_at(timestep), action, architecture):
        msg = "action is not valid at the requested timestep"
        raise ValueError(msg)
    insert_index = _path_insert_index(schedule.scheduled_actions, timestep)
    inserted = ScheduledAction(schedule.next_action_id, action)
    return rebuild_schedule(
        schedule,
        (*schedule.scheduled_actions[:insert_index], inserted, *schedule.scheduled_actions[insert_index:]),
    )


def rebuild_schedule(
    original_schedule: ActionSchedule,
    scheduled_actions: Sequence[ScheduledAction],
) -> ActionSchedule:
    """Rebuild a schedule while preserving its initial state and action identity.

    Args:
        original_schedule: Schedule whose initial state is preserved.
        scheduled_actions: Complete replacement ordered schedule.

    Returns:
        A new immutable action schedule.
    """
    actions = tuple(scheduled_actions)
    return ActionSchedule(
        scheduled_actions=actions,
        num_timesteps=sum(item.action.timestep_increment for item in actions if isinstance(item.action, AdvanceTime)),
        initial_state=original_schedule.initial_state,
    )


def validate_rebuilt_schedule(schedule: ActionSchedule, architecture: Architecture) -> bool:
    """Return whether every concurrent action layer is physically valid."""
    state = schedule.initial_state.to_replay_state()
    timestep_actions: list[Action] = []
    for item in schedule.scheduled_actions:
        action = item.action
        if isinstance(action, AdvanceTime):
            updated = _apply_valid_timestep_actions(state, timestep_actions, architecture)
            if updated is None:
                return False
            state = action.apply(updated, architecture)
            timestep_actions = []
        else:
            timestep_actions.append(action)

    if state.time != schedule.num_timesteps:
        return False
    if not timestep_actions:
        return True
    if any(not _can_end_without_time_advance(action) for action in timestep_actions):
        return False
    return _apply_valid_timestep_actions(state, timestep_actions, architecture) is not None


def validate_schedule_compatibility(schedule: ActionSchedule, architecture: Architecture) -> None:
    """Require that an architecture can execute every action in a schedule.

    Raises:
        ValueError: If sites, resources, capabilities, or action replay are
            incompatible with the architecture.
    """
    if any(not 0 <= site < architecture.num_sites for _ion, site in schedule.initial_state.positions):
        msg = "schedule initial positions fall outside the architecture"
        raise ValueError(msg)
    schedule_zones = {zone for zone, _free_time in schedule.initial_state.pzs_busy_until}
    architecture_zones = set(architecture.processing_zones or {})
    if schedule_zones != architecture_zones:
        msg = "schedule processing-zone resources do not match the architecture"
        raise ValueError(msg)
    unsupported = sorted({
        type(item.action).__name__
        for item in schedule.scheduled_actions
        if not isinstance(item.action, AdvanceTime) and not architecture.supports(type(item.action))
    })
    if unsupported:
        msg = f"schedule uses actions unsupported by the architecture: {', '.join(unsupported)}"
        raise ValueError(msg)
    if not validate_rebuilt_schedule(schedule, architecture):
        msg = "schedule actions are not valid for the architecture"
        raise ValueError(msg)


def _apply_valid_timestep_actions(
    state: State,
    actions: Sequence[Action],
    architecture: Architecture,
) -> State | None:
    transport_actions = tuple(action for action in actions if isinstance(action, TransportAction))
    if not is_transport_layer_valid(state, transport_actions, architecture):
        return None

    post_transport_state = state
    for action in transport_actions:
        post_transport_state = action.apply(post_transport_state, architecture)

    validation_state = post_transport_state
    for action in actions:
        if isinstance(action, TransportAction):
            continue
        if not is_action_valid(validation_state, action, architecture):
            return None
        validation_state = action.apply(validation_state, architecture)
    return validation_state


def _can_end_without_time_advance(action: Action) -> bool:
    return isinstance(action, GlobalPulse) or (
        isinstance(action, SingleQubitGate) and cast("_TimedGate", action).duration == 0
    )


def _path_insert_index(path: Sequence[ScheduledAction], target_time: int) -> int:
    current_time = 0
    for index, item in enumerate(path):
        if current_time == target_time and isinstance(item.action, AdvanceTime):
            return index
        if isinstance(item.action, AdvanceTime):
            current_time += item.action.timestep_increment
    if current_time != target_time:
        msg = "target_time does not correspond to a valid schedule timestep"
        raise ValueError(msg)
    return len(path)


__all__ = [
    "insert_action_at_time",
    "rebuild_schedule",
    "validate_rebuilt_schedule",
    "validate_schedule_compatibility",
]
