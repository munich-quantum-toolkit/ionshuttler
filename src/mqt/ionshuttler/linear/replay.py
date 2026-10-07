# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validate and replay explicit schedules on a Linear architecture."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import TransportAction
from mqt.ionshuttler.linear.validation import is_transport_layer_valid, is_transport_valid_in_layer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mqt.ionshuttler.linear.architecture import LinearArchitecture
    from mqt.ionshuttler.linear.schedule import Schedule, ScheduledAction
    from mqt.ionshuttler.linear.state import State


def replay_schedule(schedule: Schedule, architecture: LinearArchitecture) -> State:
    """Replay a valid explicit schedule and return its final hardware state.

    Actions with the same start time execute in stored order. The transports
    among them form one transport layer: they move together, so a transport
    may enter a site that another transport of the layer leaves.

    Returns:
        The final Linear state.
    """
    return apply_schedule(schedule, architecture)


def apply_schedule(
    schedule: Schedule,
    architecture: LinearArchitecture,
    on_layer_applied: Callable[[State, Sequence[ScheduledAction], State], None] | None = None,
    *,
    validate: bool = True,
) -> State:
    """Apply a schedule and optionally report each applied action layer.

    ``validate=False`` supports timeline analysis of intermediate schedule
    candidates. Public replay always validates the complete schedule.

    Returns:
        The final Linear state.
    """
    if validate:
        _validate_compatibility(schedule, architecture)
    state = schedule.initial_state.to_replay_state()
    index = 0
    while index < len(schedule.scheduled_actions):
        start_time = schedule.scheduled_actions[index].start_time
        state = replace(state, time=start_time)
        end_index = index + 1
        while (
            end_index < len(schedule.scheduled_actions)
            and schedule.scheduled_actions[end_index].start_time == start_time
        ):
            end_index += 1
        layer = schedule.scheduled_actions[index:end_index]
        before = state
        state = _apply_timestep_actions(state, layer, architecture, validate=validate)
        if on_layer_applied is not None:
            on_layer_applied(before, layer, state)
        index = end_index
    return replace(state, time=max(state.time, schedule.end_time))


def _validate_compatibility(schedule: Schedule, architecture: LinearArchitecture) -> None:
    """Check schedule data that action-by-action replay cannot validate.

    Raises:
        ValueError: If the schedule is incompatible with the architecture.
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
        if not architecture.supports(type(item.action))
    })
    if unsupported:
        msg = f"schedule uses actions unsupported by the architecture: {', '.join(unsupported)}"
        raise ValueError(msg)


def is_schedule_valid(schedule: Schedule, architecture: LinearArchitecture) -> bool:
    """Return whether an explicit schedule is valid for an architecture."""
    try:
        replay_schedule(schedule, architecture)
    except ValueError:
        return False
    return True


def _apply_timestep_actions(
    state: State,
    scheduled_actions: Sequence[ScheduledAction],
    architecture: LinearArchitecture,
    *,
    validate: bool,
) -> State:
    transport_actions = tuple(item.action for item in scheduled_actions if isinstance(item.action, TransportAction))
    if validate and not is_transport_layer_valid(state, transport_actions, architecture):
        msg = f"transport layer is not valid at time {state.time}"
        raise ValueError(msg)

    updated = state
    for item in scheduled_actions:
        action = item.action
        # Earlier actions of this timestep can occupy the ions of a later transport.
        if validate:
            is_valid = (
                is_transport_valid_in_layer(updated, action, transport_actions, architecture)
                if isinstance(action, TransportAction)
                else architecture.is_action_valid(updated, action)
            )
            if not is_valid:
                msg = f"action {action!r} is not valid at time {state.time}"
                raise ValueError(msg)
            _validate_execution_metadata(item, updated, architecture)
        updated = architecture.apply_action(updated, action)
    return updated


def _validate_execution_metadata(
    item: ScheduledAction,
    state: State,
    architecture: LinearArchitecture,
) -> None:
    """Check explicit schedule choices against one Linear action.

    Raises:
        ValueError: If duration or processing-zone metadata is inconsistent.
    """
    action = item.action
    duration = architecture.action_duration(action)
    if item.duration != duration:
        msg = f"scheduled duration for {type(action).__name__} does not match the architecture action"
        raise ValueError(msg)
    expected_zone = architecture.action_processing_zone(action, state)
    if item.processing_zone_id != expected_zone:
        msg = f"scheduled processing zone for {type(action).__name__} does not match the architecture state"
        raise ValueError(msg)


__all__ = ["is_schedule_valid", "replay_schedule"]
