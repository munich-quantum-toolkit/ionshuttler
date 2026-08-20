# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Executable schedules for the Linear hardware model."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mqt.ionshuttler.linear.actions import (
    BUILTIN_ACTION_TYPES,
    Action,
    AdvanceTime,
)
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.state import State

from .._json_utils import (
    require_int,
    require_int_pairs,
    require_list,
    require_mapping,
    require_str_int_pairs,
    require_str_list,
)

ActionDecoder = Callable[[Mapping[str, object]], Action]
ActionDecoders = Mapping[str, ActionDecoder]


@dataclass(frozen=True)
class MachineState:
    """Describe hardware state without compiler circuit-progress fields."""

    positions: tuple[tuple[int, int], ...]
    ions_busy_until: tuple[tuple[int, int], ...]
    pzs_busy_until: tuple[tuple[str, int], ...]
    time: int = 0

    def __post_init__(self) -> None:
        """Normalize and validate machine-state values.

        Raises:
            TypeError: If the machine clock has the wrong type.
            ValueError: If state mappings, occupancy, or timestamps are inconsistent.
        """
        positions = tuple(sorted(self.positions))
        ions_busy_until = tuple(sorted(self.ions_busy_until))
        pzs_busy_until = tuple(sorted(self.pzs_busy_until))
        _require_unique_keys(positions, "positions")
        _require_unique_keys(ions_busy_until, "ions_busy_until")
        _require_unique_keys(pzs_busy_until, "pzs_busy_until")
        if len({site for _ion, site in positions}) != len(positions):
            msg = "positions must not contain duplicate occupied sites"
            raise ValueError(msg)
        if isinstance(self.time, bool) or not isinstance(self.time, int):
            msg = "time must be an integer"
            raise TypeError(msg)
        if self.time < 0:
            msg = "time must be non-negative"
            raise ValueError(msg)
        if {ion for ion, _site in positions} != {ion for ion, _time in ions_busy_until}:
            msg = "ions_busy_until must contain exactly the positioned ions"
            raise ValueError(msg)
        if any(free_time < self.time for _ion, free_time in ions_busy_until):
            msg = "ion availability times must not precede the machine time"
            raise ValueError(msg)
        if any(free_time < self.time for _zone, free_time in pzs_busy_until):
            msg = "processing-zone availability times must not precede the machine time"
            raise ValueError(msg)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "ions_busy_until", ions_busy_until)
        object.__setattr__(self, "pzs_busy_until", pzs_busy_until)

    @classmethod
    def from_compiler_state(cls, state: State) -> MachineState:
        """Copy only hardware fields from a compiler state.

        Availability timestamps in a compiler state may refer to resources
        that became free before its current clock.  The machine-only view
        canonicalizes those timestamps to the current clock.

        Args:
            state: Compiler state whose circuit progress is discarded.

        Returns:
            The canonical machine-only state.
        """
        return cls(
            positions=state.positions,
            ions_busy_until=tuple((ion, max(free_time, state.time)) for ion, free_time in state.ions_busy_until),
            pzs_busy_until=tuple((zone, max(free_time, state.time)) for zone, free_time in state.pzs_busy_until),
            time=state.time,
        )

    def to_replay_state(self) -> State:
        """Return a state suitable for schedule replay.

        Returns:
            A compiler-state value with empty circuit-progress fields.
        """
        return State(
            positions=self.positions,
            completed_gates=frozenset(),
            in_progress_gates=(),
            ions_busy_until=self.ions_busy_until,
            pzs_busy_until=self.pzs_busy_until,
            time=self.time,
        )

    def to_dict(self) -> dict[str, object]:
        """Return this machine state using JSON-compatible values."""
        return {
            "positions": [list(item) for item in self.positions],
            "ions_busy_until": [list(item) for item in self.ions_busy_until],
            "pzs_busy_until": [list(item) for item in self.pzs_busy_until],
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, data: object) -> MachineState:
        """Restore a machine state from JSON-compatible values.

        Returns:
            The restored machine-only state.

        """
        mapping = require_mapping(data, "machine_state")
        return cls(
            positions=tuple(require_int_pairs(mapping, "positions")),
            ions_busy_until=tuple(require_int_pairs(mapping, "ions_busy_until")),
            pzs_busy_until=tuple(require_str_int_pairs(mapping, "pzs_busy_until")),
            time=require_int(mapping, "time"),
        )


@dataclass(frozen=True)
class ScheduledAction:
    """Pair one operation with an ID stable across schedule passes."""

    action_id: int
    action: Action

    def __post_init__(self) -> None:
        """Validate the action ID and value.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If the ID is invalid.
        """
        if isinstance(self.action_id, bool) or not isinstance(self.action_id, int):
            msg = "action_id must be an integer"
            raise TypeError(msg)
        if self.action_id < 0:
            msg = "action_id must be non-negative"
            raise ValueError(msg)
        if not isinstance(self.action, Action):
            msg = "action must be an Action"
            raise TypeError(msg)


@dataclass(frozen=True)
class ActionSchedule:
    """Contain the immutable executable boundary shared by downstream stages.

    Action IDs are unique within a schedule and remain attached to
    preserved actions when a transformation inserts or replaces operations.
    Compiler search status and dynamical-decoupling reports are intentionally
    absent.
    """

    scheduled_actions: tuple[ScheduledAction, ...]
    num_timesteps: int
    architecture: Architecture
    initial_state: MachineState
    action_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze and validate the executable schedule.

        Raises:
            TypeError: If a schedule field has the wrong type.
            ValueError: If identifiers, timing, or capabilities are inconsistent.
        """
        scheduled_actions = tuple(self.scheduled_actions)
        if any(not isinstance(item, ScheduledAction) for item in scheduled_actions):
            msg = "scheduled_actions must contain ScheduledAction values"
            raise TypeError(msg)
        action_ids = tuple(item.action_id for item in scheduled_actions)
        if len(set(action_ids)) != len(action_ids):
            msg = "scheduled action identifiers must be unique"
            raise ValueError(msg)
        if isinstance(self.num_timesteps, bool) or not isinstance(self.num_timesteps, int):
            msg = "num_timesteps must be an integer"
            raise TypeError(msg)
        if self.num_timesteps < 0:
            msg = "num_timesteps must be non-negative"
            raise ValueError(msg)
        actual_timesteps = sum(
            item.action.timestep_increment for item in scheduled_actions if isinstance(item.action, AdvanceTime)
        )
        if actual_timesteps != self.num_timesteps:
            msg = "num_timesteps must equal the schedule's total time advancement"
            raise ValueError(msg)
        if not isinstance(self.architecture, Architecture):
            msg = "architecture must be an Architecture"
            raise TypeError(msg)
        if not isinstance(self.initial_state, MachineState):
            msg = "initial_state must be a MachineState"
            raise TypeError(msg)
        action_types = tuple(self.action_types)
        if any(not isinstance(action_type, str) for action_type in action_types):
            msg = "action_types must contain strings"
            raise TypeError(msg)
        if len(set(action_types)) != len(action_types):
            msg = "action_types must not contain duplicates"
            raise ValueError(msg)
        object.__setattr__(self, "scheduled_actions", scheduled_actions)
        object.__setattr__(self, "action_types", action_types)

    @property
    def path(self) -> tuple[Action, ...]:
        """The ordered executable operations without their metadata."""
        return tuple(item.action for item in self.scheduled_actions)

    @property
    def next_action_id(self) -> int:
        """An unused identifier suitable for the next inserted action."""
        return max((item.action_id for item in self.scheduled_actions), default=-1) + 1

    @classmethod
    def from_actions(
        cls,
        actions: Sequence[Action],
        architecture: Architecture,
        initial_state: State | MachineState,
        *,
        action_types: Sequence[str] = (),
    ) -> ActionSchedule:
        """Create a schedule and assign deterministic action identifiers.

        Args:
            actions: Ordered executable operations.
            architecture: Hardware model used by the schedule.
            initial_state: Initial hardware or compiler state. Compiler progress
                fields are deliberately discarded.
            action_types: Serialized names of hardware capabilities.

        Returns:
            The immutable action schedule.
        """
        action_values = tuple(actions)
        machine_state = (
            initial_state
            if isinstance(initial_state, MachineState)
            else MachineState.from_compiler_state(initial_state)
        )
        return cls(
            scheduled_actions=tuple(
                ScheduledAction(action_id=index, action=action) for index, action in enumerate(action_values)
            ),
            num_timesteps=sum(action.timestep_increment for action in action_values if isinstance(action, AdvanceTime)),
            architecture=architecture,
            initial_state=machine_state,
            action_types=tuple(action_types),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the action schedule using JSON-compatible values."""
        current_time = self.initial_state.time
        actions: list[dict[str, object]] = []
        for item in self.scheduled_actions:
            actions.append({
                "action_id": item.action_id,
                "start_time": current_time,
                "action": item.action.to_dict(),
            })
            if isinstance(item.action, AdvanceTime):
                current_time += item.action.timestep_increment
        return {
            "num_timesteps": self.num_timesteps,
            "architecture": self.architecture.to_dict(),
            "initial_state": self.initial_state.to_dict(),
            "action_types": list(self.action_types),
            "actions": actions,
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        action_types: Sequence[type[Action]] | None = None,
        action_decoders: ActionDecoders | None = None,
    ) -> ActionSchedule:
        """Restore a schedule from its JSON-compatible representation.

        Returns:
            The restored action schedule.

        Raises:
            ValueError: If the action stream or metadata is malformed.
        """
        mapping = require_mapping(data, "serialized action schedule")
        raw_actions = require_list(mapping, "actions")
        registry = build_action_type_registry(action_types)
        scheduled_actions: list[ScheduledAction] = []
        current_time = require_mapping(mapping.get("initial_state"), "initial_state").get("time")
        if isinstance(current_time, bool) or not isinstance(current_time, int):
            msg = "initial_state.time must be an integer"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
        for raw_item in raw_actions:
            item = require_mapping(raw_item, "each scheduled action")
            start_time = require_int(item, "start_time")
            if start_time != current_time:
                msg = "scheduled action start_time does not match ordered time advancement"
                raise ValueError(msg)
            action = decode_action(item.get("action"), registry, action_decoders)
            scheduled_actions.append(ScheduledAction(action_id=require_int(item, "action_id"), action=action))
            if isinstance(action, AdvanceTime):
                current_time += action.timestep_increment
        architecture_data = mapping.get("architecture")
        if not isinstance(architecture_data, dict):
            msg = "architecture must be a JSON object"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
        return cls(
            scheduled_actions=tuple(scheduled_actions),
            num_timesteps=require_int(mapping, "num_timesteps"),
            architecture=Architecture.from_dict(architecture_data),
            initial_state=MachineState.from_dict(mapping.get("initial_state")),
            action_types=tuple(require_str_list(mapping, "action_types")),
        )

    def to_json(self) -> str:
        """Serialize this schedule as JSON text.

        Returns:
            The JSON document.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        action_types: Sequence[type[Action]] | None = None,
        action_decoders: ActionDecoders | None = None,
    ) -> ActionSchedule:
        """Restore a schedule from JSON text.

        Returns:
            The restored action schedule.
        """
        return cls.from_dict(json.loads(raw), action_types=action_types, action_decoders=action_decoders)

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

    @classmethod
    def load(
        cls,
        filename: str | Path,
        *,
        action_types: Sequence[type[Action]] | None = None,
        action_decoders: ActionDecoders | None = None,
    ) -> ActionSchedule:
        """Load a schedule from an explicit UTF-8 JSON file.

        Returns:
            The restored action schedule.
        """
        return cls.from_json(
            Path(filename).read_text(encoding="utf-8"),
            action_types=action_types,
            action_decoders=action_decoders,
        )


def build_action_type_registry(action_types: Sequence[type[Action]] | None) -> dict[str, type[Action]]:
    registry = {action_type.__name__: action_type for action_type in (*BUILTIN_ACTION_TYPES, AdvanceTime)}
    for action_type in () if action_types is None else action_types:
        if not isinstance(action_type, type) or not issubclass(action_type, Action):
            msg = "action_types must contain Action subclasses"
            raise TypeError(msg)
        name = action_type.__name__
        existing = registry.get(name)
        if existing is not None and existing is not action_type:
            msg = f"duplicate serialized action type {name!r}"
            raise ValueError(msg)
        registry[name] = action_type
    return registry


def decode_action(
    data: object,
    action_types: Mapping[str, type[Action]],
    action_decoders: ActionDecoders | None,
) -> Action:
    mapping = require_mapping(data, "action")
    action_type = mapping.get("type")
    if not isinstance(action_type, str):
        msg = "action.type must be a string"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    if action_type in action_types:
        return action_types[action_type].from_dict(mapping)
    if action_decoders is not None and action_type in action_decoders:
        return action_decoders[action_type](mapping)
    msg = f"unknown action type: {action_type}"
    raise ValueError(msg)


def _require_unique_keys(values: Sequence[tuple[object, object]], label: str) -> None:
    if len({key for key, _value in values}) != len(values):
        msg = f"{label} must not contain duplicate keys"
        raise ValueError(msg)


__all__ = [
    "ActionDecoder",
    "ActionSchedule",
    "MachineState",
    "ScheduledAction",
]
