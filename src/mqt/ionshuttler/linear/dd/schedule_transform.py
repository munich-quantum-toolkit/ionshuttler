# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Rebuild and validate compiled schedules after decoupling transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from mqt.ionshuttler.linear.actions import (
    Action,
    AdvanceTime,
    GlobalPulse,
    SingleQubitGate,
    TransportAction,
)
from mqt.ionshuttler.linear.cost import cost
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.result import CompilationResult
from mqt.ionshuttler.linear.state import State, normalize_initial_state
from mqt.ionshuttler.linear.validation import is_action_valid, is_transport_layer_valid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.linear.architecture import Architecture


class _TimedGate(Protocol):
    """Describe the duration field shared by concrete physical gates."""

    duration: int


def insert_action_at_time(
    result: CompilationResult,
    timestep: int,
    action: Action,
    architecture: Architecture | None = None,
) -> CompilationResult:
    """Insert an action immediately before a boundary's time advance.

    Args:
        result: Schedule to transform.
        timestep: Boundary at which the action starts.
        action: Action to insert.
        architecture: Hardware model, defaulting to the result metadata.

    Returns:
        A rebuilt result containing the inserted action.

    Raises:
        ValueError: If schedule metadata, the timestep, or the action is invalid.
    """
    resolved_architecture = _require_architecture(result, architecture)
    if result.initial_state is None:
        msg = "result.initial_state is required for action insertion"
        raise ValueError(msg)
    if not 0 <= timestep <= result.num_timesteps:
        msg = f"timestep must be within [0, {result.num_timesteps}]"
        raise ValueError(msg)

    timeline = build_timeline(result, resolved_architecture)
    if not is_action_valid(timeline.state_at(timestep), action, resolved_architecture):
        msg = "action is not valid at the requested timestep"
        raise ValueError(msg)
    insert_index = _path_insert_index(result.path, timestep)
    new_path = [*result.path[:insert_index], action, *result.path[insert_index:]]
    return rebuild_result(result, new_path, resolved_architecture)


def rebuild_result(
    original_result: CompilationResult,
    path: Sequence[Action],
    architecture: Architecture | None = None,
) -> CompilationResult:
    """Rebuild derived result state while preserving compiler diagnostics.

    Args:
        original_result: Result whose non-schedule diagnostics are preserved.
        path: Replacement ordered action schedule.
        architecture: Hardware model, defaulting to the result metadata.

    Returns:
        A new compilation result with reconstructed makespan and final hardware state.

    """
    resolved_architecture = _require_architecture(original_result, architecture)
    new_path = list(path)
    temp_result = CompilationResult(
        status=original_result.status,
        path=new_path,
        num_timesteps=sum(isinstance(action, AdvanceTime) for action in new_path),
        wall_clock_s=original_result.wall_clock_s,
        architecture=resolved_architecture,
        initial_state=original_result.initial_state,
        action_types=original_result.action_types,
    )
    timeline = build_timeline(temp_result, resolved_architecture)
    rebuilt_final_state = timeline.state_at(timeline.makespan)
    if original_result.final_state is not None:
        rebuilt_final_state = State(
            positions=rebuilt_final_state.positions,
            completed_gates=original_result.final_state.completed_gates,
            in_progress_gates=original_result.final_state.in_progress_gates,
            ions_busy_until=rebuilt_final_state.ions_busy_until,
            pzs_busy_until=rebuilt_final_state.pzs_busy_until,
            time=rebuilt_final_state.time,
        )
    return CompilationResult(
        status=original_result.status,
        path=new_path,
        num_timesteps=temp_result.num_timesteps,
        wall_clock_s=original_result.wall_clock_s,
        score=cost(rebuilt_final_state),
        final_state=rebuilt_final_state,
        architecture=resolved_architecture,
        initial_state=original_result.initial_state,
        dd_insertions=original_result.dd_insertions,
        global_dd_records=original_result.global_dd_records,
        explored_nodes=original_result.explored_nodes,
        action_types=original_result.action_types,
    )


def validate_rebuilt_schedule(
    result: CompilationResult,
    architecture: Architecture | None = None,
) -> bool:
    """Return whether every concurrent action layer is physically valid."""
    resolved_architecture = _require_architecture(result, architecture)
    if result.initial_state is None:
        return False

    state = normalize_initial_state(result.initial_state, resolved_architecture)
    timestep_actions: list[Action] = []
    for action in result.path:
        if isinstance(action, AdvanceTime):
            updated = _apply_valid_timestep_actions(state, timestep_actions, resolved_architecture)
            if updated is None:
                return False
            state = action.apply(updated, resolved_architecture)
            timestep_actions = []
        else:
            timestep_actions.append(action)

    if state.time != result.num_timesteps:
        return False
    if not timestep_actions:
        return True
    if any(not _can_end_without_time_advance(action) for action in timestep_actions):
        return False
    return _apply_valid_timestep_actions(state, timestep_actions, resolved_architecture) is not None


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


def _path_insert_index(path: Sequence[Action], target_time: int) -> int:
    current_time = 0
    for index, action in enumerate(path):
        if current_time == target_time and isinstance(action, AdvanceTime):
            return index
        if isinstance(action, AdvanceTime):
            current_time += action.timestep_increment
    if current_time != target_time:
        msg = "target_time does not correspond to a valid schedule timestep"
        raise ValueError(msg)
    return len(path)


def _require_architecture(
    result: CompilationResult,
    architecture: Architecture | None,
) -> Architecture:
    resolved = architecture or result.architecture
    if resolved is None:
        msg = "architecture must be provided directly or via result.architecture"
        raise ValueError(msg)
    return resolved


__all__ = ["insert_action_at_time", "rebuild_result", "validate_rebuilt_schedule"]
