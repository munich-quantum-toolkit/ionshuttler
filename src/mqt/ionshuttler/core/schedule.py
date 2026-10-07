# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared explicit-timeline schedule values."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from mqt.ionshuttler.core.actions import Action

from .._json_utils import require_int, require_list, require_mapping

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

SCHEDULE_SCHEMA = "mqt.ionshuttler.schedule"
SCHEDULE_VERSION = 4


class MachineState(Protocol):
    """Describe the state operations required by schedule persistence."""

    @property
    def time(self) -> int:
        """Current machine time."""

    def to_dict(self) -> dict[str, object]:
        """Return this state using JSON-compatible values."""


ActionT = TypeVar("ActionT", bound=Action)
MachineStateT = TypeVar("MachineStateT", bound=MachineState)


@dataclass(frozen=True)
class ScheduledAction(Generic[ActionT]):
    """Store one schedulable operation and its explicit interval."""

    action_id: int
    action: ActionT
    start_time: int
    duration: int
    processing_zone_id: str | None = None

    def __post_init__(self) -> None:
        """Validate identity, timing, and optional resource selection.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If an identifier, time, duration, or resource is invalid.
        """
        _require_non_negative_int(self.action_id, "action_id")
        if not isinstance(self.action, Action):
            msg = "action must be an Action"
            raise TypeError(msg)
        _require_non_negative_int(self.start_time, "start_time")
        _require_non_negative_int(self.duration, "duration")
        if self.processing_zone_id is not None:
            if not isinstance(self.processing_zone_id, str):
                msg = "processing_zone_id must be a string or None"
                raise TypeError(msg)
            if not self.processing_zone_id:
                msg = "processing_zone_id must be non-empty or None"
                raise ValueError(msg)

    @property
    def end_time(self) -> int:
        """Exclusive end of this action interval."""
        return self.start_time + self.duration


@dataclass(frozen=True)
class Schedule(Generic[ActionT, MachineStateT]):
    """Store a flat sequence of explicitly timed actions.

    Actions are stored in execution order. Start times never decrease, and
    actions with the same start time execute in stored order. This order
    matters when an action depends on a zero-duration action that starts at
    the same time.
    """

    scheduled_actions: tuple[ScheduledAction[ActionT], ...]
    end_time: int
    initial_state: MachineStateT

    def __post_init__(self) -> None:
        """Validate action identity, sequence order, and time bounds.

        Raises:
            TypeError: If a schedule field has the wrong type.
            ValueError: If identifiers or timing are inconsistent.
        """
        scheduled_actions = tuple(self.scheduled_actions)
        if any(not isinstance(item, ScheduledAction) for item in scheduled_actions):
            msg = "scheduled_actions must contain ScheduledAction values"
            raise TypeError(msg)
        action_ids = tuple(item.action_id for item in scheduled_actions)
        if len(set(action_ids)) != len(action_ids):
            msg = "scheduled action identifiers must be unique"
            raise ValueError(msg)
        starts = tuple(item.start_time for item in scheduled_actions)
        if starts != tuple(sorted(starts)):
            msg = "scheduled action start times must be nondecreasing"
            raise ValueError(msg)
        if starts and starts[0] < self.initial_state.time:
            msg = "scheduled actions must not start before the initial state time"
            raise ValueError(msg)
        _require_non_negative_int(self.end_time, "end_time")
        if self.end_time < self.start_time:
            msg = "end_time must not precede start_time"
            raise ValueError(msg)
        latest_end = max((item.end_time for item in scheduled_actions), default=self.initial_state.time)
        if latest_end > self.end_time:
            msg = "scheduled actions must finish within end_time"
            raise ValueError(msg)
        object.__setattr__(self, "scheduled_actions", scheduled_actions)

    @property
    def path(self) -> tuple[ActionT, ...]:
        """The bare series of operations without occurrence metadata."""
        return tuple(item.action for item in self.scheduled_actions)

    @property
    def start_time(self) -> int:
        """Absolute start time of the schedule."""
        return self.initial_state.time

    @property
    def duration(self) -> int:
        """Number of elapsed timesteps covered by the schedule."""
        return self.end_time - self.start_time

    @property
    def next_action_id(self) -> int:
        """An unused identifier for an inserted action."""
        return max((item.action_id for item in self.scheduled_actions), default=-1) + 1

    def to_dict(self) -> dict[str, object]:
        """Return this schedule using the versioned JSON representation."""
        return {
            "schema": SCHEDULE_SCHEMA,
            "version": SCHEDULE_VERSION,
            "end_time": self.end_time,
            "initial_state": self.initial_state.to_dict(),
            "actions": [
                {
                    "action_id": item.action_id,
                    "start_time": item.start_time,
                    "duration": item.duration,
                    "processing_zone_id": item.processing_zone_id,
                    "action": item.action.to_dict(),
                }
                for item in self.scheduled_actions
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        decode_action: Callable[[Mapping[str, object]], ActionT],
        decode_state: Callable[[object], MachineStateT],
    ) -> Schedule[ActionT, MachineStateT]:
        """Restore a schedule from the current versioned representation.

        Each architecture supplies the decoders for its own actions and machine
        state.

        Args:
            data: Serialized schedule.
            decode_action: Restores one serialized action.
            decode_state: Restores the serialized initial machine state.

        Returns:
            The restored schedule.

        Raises:
            ValueError: If the schema, version, or scheduled-action data is invalid.
        """
        mapping = require_mapping(data, "serialized schedule")
        if mapping.get("schema") != SCHEDULE_SCHEMA or mapping.get("version") != SCHEDULE_VERSION:
            msg = "unsupported schedule schema or version"
            raise ValueError(msg)
        scheduled_actions: list[ScheduledAction[ActionT]] = []
        for raw_item in require_list(mapping, "actions"):
            item = require_mapping(raw_item, "each scheduled action")
            processing_zone_id = item.get("processing_zone_id")
            if processing_zone_id is not None and not isinstance(processing_zone_id, str):
                msg = "processing_zone_id must be a string or null"
                raise ValueError(msg)
            scheduled_actions.append(
                ScheduledAction(
                    action_id=require_int(item, "action_id"),
                    action=decode_action(require_mapping(item.get("action"), "action")),
                    start_time=require_int(item, "start_time"),
                    duration=require_int(item, "duration"),
                    processing_zone_id=processing_zone_id,
                )
            )
        return cls(
            scheduled_actions=tuple(scheduled_actions),
            end_time=require_int(mapping, "end_time"),
            initial_state=decode_state(mapping.get("initial_state")),
        )

    def to_json(self) -> str:
        """Serialize this schedule as JSON text.

        Returns:
            The JSON document.
        """
        return json.dumps(self.to_dict())

    def save(self, filename: str | Path) -> Path:
        """Write this schedule to an explicit UTF-8 JSON file.

        Returns:
            The path written.
        """
        output_path = Path(filename)
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")
        return output_path


def _require_non_negative_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer"
        raise TypeError(msg)
    if value < 0:
        msg = f"{label} must be non-negative"
        raise ValueError(msg)


__all__ = ["SCHEDULE_SCHEMA", "SCHEDULE_VERSION", "MachineState", "Schedule", "ScheduledAction"]
