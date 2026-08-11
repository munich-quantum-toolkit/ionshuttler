# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compilation outcomes and their portable JSON representation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

from mqt.ionshuttler.linear.actions import (
    BUILTIN_ACTION_TYPES,
    Action,
    AdvanceTime,
)
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.state import State, to_metadata_dict

ActionDecoder = Callable[[Mapping[str, object]], Action]
ActionDecoders = Mapping[str, ActionDecoder]


class CompilationStatus(str, Enum):
    """Describe how compilation ended."""

    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class DDInsertionRecord:
    """Describe a local pulse sequence added after compilation."""

    ion: int
    window: tuple[int, int]
    scheme_name: str
    gate_timesteps: tuple[int, ...]
    remaining_phase: float = 0.0
    phase_reduction: float = 0.0
    residual_phase_at_window_end_before: float | None = None
    residual_phase_at_window_end_after: float | None = None
    residual_phase_at_window_end_reduction: float | None = None


@dataclass(frozen=True)
class GlobalDDRecord:
    """Describe a schedule-wide pulse sequence added after compilation."""

    scheme_name: str
    pulse_timesteps: tuple[int, ...]
    spacing: int
    sum_abs_residual_phase: float = 0.0
    sum_squared_residual_phase: float = 0.0
    max_abs_residual_phase: float = 0.0


@dataclass(frozen=True)
class CompilationResult:
    """Contain a schedule, its outcome, and supporting hardware information.

    Compiled results retain the ordered action-type names exposed by the
    hardware so saved schedules remain clear about available capabilities.
    """

    status: CompilationStatus
    path: list[Action]
    num_timesteps: int
    wall_clock_s: float = 0.0
    score: int | None = None
    final_state: State | None = None
    architecture: Architecture | None = None
    initial_state: State | None = None
    dd_insertions: tuple[DDInsertionRecord, ...] = ()
    global_dd_records: tuple[GlobalDDRecord, ...] = ()
    explored_nodes: int | None = None
    action_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Ensure state metadata can be interpreted unambiguously.

        Raises:
            TypeError: If an action type name is not a string.
            ValueError: If an initial state is provided without its architecture.
        """
        if self.initial_state is not None and self.architecture is None:
            msg = "architecture is required when initial_state is provided"
            raise ValueError(msg)
        if any(not isinstance(action_type, str) for action_type in self.action_types):
            msg = "action_types must contain strings"
            raise TypeError(msg)
        if len(set(self.action_types)) != len(self.action_types):
            msg = "action_types must not contain duplicates"
            raise ValueError(msg)

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        action_types: Sequence[type[Action]] | None = None,
        action_decoders: ActionDecoders | None = None,
    ) -> CompilationResult:
        """Restore a result from a JSON-style mapping.

        Args:
            data: Serialized result mapping.
            action_types: Custom action classes needed to restore the schedule.
            action_decoders: Optional decoders for downstream action types.

        Returns:
            The restored compilation result.

        Raises:
            ValueError: If the serialized result is malformed.
        """
        if not isinstance(data, dict):
            msg = "serialized compilation result must be a JSON object"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
        mapping = cast("dict[str, object]", data)
        metadata = mapping.get("metadata")
        architecture = _architecture_from_metadata(metadata)
        initial_state = _initial_state_from_metadata(metadata, architecture)
        raw_actions = _require_list(mapping, "actions", default=[])
        raw_dd_insertions = _require_list(mapping, "dd_insertions", default=[])
        raw_global_records = _require_list(mapping, "global_dd_records", default=[])
        serialized_action_types = tuple(_require_str_list(mapping, "action_types", default=[]))
        action_type_registry = _action_type_registry(action_types)

        status_raw = mapping.get("status", CompilationStatus.FAILED.value)
        try:
            status = CompilationStatus(status_raw)
        except ValueError as error:
            msg = f"unknown compilation status: {status_raw}"
            raise ValueError(msg) from error

        num_timesteps = _require_int_with_default(mapping, "num_timesteps", 0)
        wall_clock_s = _require_number_with_default(mapping, "wall_clock_s", 0.0)
        score = mapping.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, int)):
            msg = "score must be an integer or null"
            raise ValueError(msg)

        return cls(
            status=status,
            path=[_action_from_dict(action, action_type_registry, action_decoders) for action in raw_actions],
            num_timesteps=num_timesteps,
            wall_clock_s=wall_clock_s,
            score=score,
            architecture=architecture,
            initial_state=initial_state,
            dd_insertions=tuple(_dd_insertion_from_dict(record) for record in raw_dd_insertions),
            global_dd_records=tuple(_global_dd_record_from_dict(record) for record in raw_global_records),
            action_types=serialized_action_types,
        )

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        action_types: Sequence[type[Action]] | None = None,
        action_decoders: ActionDecoders | None = None,
    ) -> CompilationResult:
        """Restore a result from JSON text.

        Args:
            raw: Serialized result text.
            action_types: Custom action classes needed to restore the schedule.
            action_decoders: Optional decoders for downstream action types.

        Returns:
            The restored compilation result.
        """
        return cls.from_dict(
            json.loads(raw),
            action_types=action_types,
            action_decoders=action_decoders,
        )

    @classmethod
    def load(
        cls,
        filename: str | Path,
        *,
        action_types: Sequence[type[Action]] | None = None,
        action_decoders: ActionDecoders | None = None,
    ) -> CompilationResult:
        """Load a result from a UTF-8 JSON file.

        Args:
            filename: File to read.
            action_types: Custom action classes needed to restore the schedule.
            action_decoders: Optional decoders for downstream action types.

        Returns:
            The restored compilation result.
        """
        return cls.from_json(
            Path(filename).read_text(encoding="utf-8"),
            action_types=action_types,
            action_decoders=action_decoders,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the schedule and metadata using JSON-compatible values."""
        actions: list[dict[str, object]] = []
        current_time = 0
        for action in self.path:
            actions.append(_scheduled_action_to_dict(action, current_time))
            if isinstance(action, AdvanceTime):
                current_time += action.timestep_increment

        result: dict[str, object] = {}
        if self.architecture is not None:
            metadata: dict[str, object] = {"architecture": self.architecture.to_dict()}
            if self.initial_state is not None:
                metadata["initial_state"] = to_metadata_dict(
                    self.initial_state,
                    self.architecture.num_sites,
                )
            result["metadata"] = metadata

        result.update({
            "status": self.status.value,
            "num_timesteps": self.num_timesteps,
            "wall_clock_s": self.wall_clock_s,
            "score": self.score,
            "action_types": list(self.action_types),
            "actions": actions,
            "dd_insertions": [_dd_insertion_to_dict(record) for record in self.dd_insertions],
            "global_dd_records": [_global_dd_record_to_dict(record) for record in self.global_dd_records],
        })
        return result

    def to_json(self) -> str:
        """Serialize the result as JSON text.

        Returns:
            The serialized result.
        """
        return json.dumps(self.to_dict())

    def save(
        self,
        filename: str | Path,
        directory: str | Path = "outputs/results/json",
    ) -> Path:
        """Write the result to a UTF-8 JSON file.

        Args:
            filename: Output filename, with ``.json`` added when omitted.
            directory: Directory in which to create the file.

        Returns:
            The path written.
        """
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.write_text(self.to_json(), encoding="utf-8")
        return output_path


def _scheduled_action_to_dict(action: Action, start_time: int) -> dict[str, object]:
    data = action.to_dict()
    duration = data.pop("duration", None)
    data["start_time"] = start_time
    if duration is not None:
        data["duration"] = duration
    elif isinstance(action, AdvanceTime):
        data["duration"] = action.timestep_increment
    return data


def _action_type_registry(
    action_types: Sequence[type[Action]] | None,
) -> dict[str, type[Action]]:
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


def _action_from_dict(
    data: object,
    action_types: Mapping[str, type[Action]],
    action_decoders: ActionDecoders | None,
) -> Action:
    mapping = _require_mapping(data, "each action")
    action_type = _require_str(mapping, "type")
    if action_type in action_types:
        return action_types[action_type].from_dict(mapping)
    if action_decoders is not None and action_type in action_decoders:
        return action_decoders[action_type](mapping)
    msg = f"unknown action type: {action_type}"
    raise ValueError(msg)


def _dd_insertion_to_dict(record: DDInsertionRecord) -> dict[str, object]:
    data: dict[str, object] = {
        "ion": record.ion,
        "window": list(record.window),
        "scheme_name": record.scheme_name,
        "gate_timesteps": list(record.gate_timesteps),
        "remaining_phase": record.remaining_phase,
        "phase_reduction": record.phase_reduction,
    }
    optional_values = {
        "residual_phase_at_window_end_before": record.residual_phase_at_window_end_before,
        "residual_phase_at_window_end_after": record.residual_phase_at_window_end_after,
        "residual_phase_at_window_end_reduction": record.residual_phase_at_window_end_reduction,
    }
    data.update({key: value for key, value in optional_values.items() if value is not None})
    return data


def _global_dd_record_to_dict(record: GlobalDDRecord) -> dict[str, object]:
    return {
        "scheme_name": record.scheme_name,
        "pulse_timesteps": list(record.pulse_timesteps),
        "spacing": record.spacing,
        "sum_abs_residual_phase": record.sum_abs_residual_phase,
        "sum_squared_residual_phase": record.sum_squared_residual_phase,
        "max_abs_residual_phase": record.max_abs_residual_phase,
    }


def _dd_insertion_from_dict(data: object) -> DDInsertionRecord:
    mapping = _require_mapping(data, "each dd_insertion")
    window = _require_int_list(mapping, "window", expected_length=2)
    timestep_key = "gate_timesteps" if "gate_timesteps" in mapping else "pulse_timesteps"
    remaining_phase_key = "remaining_phase" if "remaining_phase" in mapping else "mismatch"
    return DDInsertionRecord(
        ion=_require_int(mapping, "ion"),
        window=(window[0], window[1]),
        scheme_name=_require_str(mapping, "scheme_name"),
        gate_timesteps=tuple(_require_int_list(mapping, timestep_key)),
        remaining_phase=_require_number_with_default(mapping, remaining_phase_key, 0.0),
        phase_reduction=_require_number_with_default(mapping, "phase_reduction", 0.0),
        residual_phase_at_window_end_before=_require_optional_number(
            mapping,
            "residual_phase_at_window_end_before",
        ),
        residual_phase_at_window_end_after=_require_optional_number(
            mapping,
            "residual_phase_at_window_end_after",
        ),
        residual_phase_at_window_end_reduction=_require_optional_number(
            mapping,
            "residual_phase_at_window_end_reduction",
        ),
    )


def _global_dd_record_from_dict(data: object) -> GlobalDDRecord:
    mapping = _require_mapping(data, "each global_dd_record")
    return GlobalDDRecord(
        scheme_name=_require_str(mapping, "scheme_name"),
        pulse_timesteps=tuple(_require_int_list(mapping, "pulse_timesteps")),
        spacing=_require_int(mapping, "spacing"),
        sum_abs_residual_phase=_require_number_with_default(mapping, "sum_abs_residual_phase", 0.0),
        sum_squared_residual_phase=_require_number_with_default(
            mapping,
            "sum_squared_residual_phase",
            0.0,
        ),
        max_abs_residual_phase=_require_number_with_default(mapping, "max_abs_residual_phase", 0.0),
    )


def _architecture_from_metadata(metadata: object) -> Architecture | None:
    if not isinstance(metadata, dict):
        return None
    architecture = metadata.get("architecture")
    return Architecture.from_dict(architecture) if isinstance(architecture, dict) else None


def _initial_state_from_metadata(
    metadata: object,
    architecture: Architecture | None,
) -> State | None:
    if architecture is None or not isinstance(metadata, dict):
        return None
    raw_initial_state = metadata.get("initial_state")
    if not isinstance(raw_initial_state, dict):
        return None
    occupancy = raw_initial_state.get("site_occupancy")
    if not isinstance(occupancy, list):
        return None
    positions: list[tuple[int, int]] = []
    for site, ion in enumerate(occupancy):
        if ion is None:
            continue
        if isinstance(ion, bool) or not isinstance(ion, int):
            msg = "initial_state.site_occupancy must contain integers or null"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
        positions.append((ion, site))
    ion_ids = sorted(ion for ion, _ in positions)
    return State(
        positions=tuple(positions),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=tuple((ion, 0) for ion in ion_ids),
        pzs_busy_until=architecture.initial_pzs_busy_until(),
        time=0,
    )


def _require_mapping(data: object, label: str) -> dict[str, object]:
    if not isinstance(data, dict):
        msg = f"{label} must be a JSON object"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    return cast("dict[str, object]", data)


def _require_list(data: dict[str, object], key: str, *, default: list[object]) -> list[object]:
    value = data.get(key, default)
    if not isinstance(value, list):
        msg = f"{key} must be a list"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    return cast("list[object]", value)


def _require_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    return value


def _require_int_with_default(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    return value


def _require_number_with_default(
    data: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key} must be numeric"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    return float(value)


def _require_optional_number(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key} must be numeric or null"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    return float(value)


def _require_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    return value


def _require_int_list(
    data: Mapping[str, object],
    key: str,
    *,
    expected_length: int | None = None,
) -> list[int]:
    value = data.get(key)
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        msg = f"{key} must be a list of integers"
        raise ValueError(msg)
    result = cast("list[int]", value)
    if expected_length is not None and len(result) != expected_length:
        msg = f"{key} must have length {expected_length}"
        raise ValueError(msg)
    return result


def _require_str_list(
    data: Mapping[str, object],
    key: str,
    *,
    default: list[object],
) -> list[str]:
    value = data.get(key, default)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        msg = f"{key} must be a list of strings"
        raise ValueError(msg)
    return cast("list[str]", value)


__all__ = [
    "ActionDecoder",
    "CompilationResult",
    "CompilationStatus",
    "DDInsertionRecord",
    "GlobalDDRecord",
]
