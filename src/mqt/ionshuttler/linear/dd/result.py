# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared result and report values for Linear dynamical-decoupling passes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, Protocol, TypeVar, cast

from mqt.ionshuttler.linear.schedule import ActionSchedule

from ..._json_utils import (
    require_int,
    require_int_list,
    require_mapping,
    require_number,
    require_optional_number,
    require_str,
)


class DDReport(Protocol):
    """Describe a method-specific report that supports JSON serialization."""

    report_type: ClassVar[str]

    def to_dict(self) -> dict[str, object]:
        """Return the report using JSON-compatible values."""

    @classmethod
    def from_dict(cls, data: object) -> DDReport:
        """Restore a report from JSON-compatible values."""


ReportT = TypeVar("ReportT", bound=DDReport)


@dataclass(frozen=True)
class LocalDDSequence:
    """Describe one ion-local pulse sequence in a DD method report."""

    ion: int
    window: tuple[int, int]
    scheme_name: str
    pulse_timesteps: tuple[int, ...]
    action_ids: tuple[int, ...]
    remaining_phase: float = 0.0
    phase_reduction: float = 0.0
    residual_phase_at_window_end_before: float | None = None
    residual_phase_at_window_end_after: float | None = None
    residual_phase_at_window_end_reduction: float | None = None

    def __post_init__(self) -> None:
        """Validate pulse positions and their associated idle window.

        Raises:
            TypeError: If a sequence field has the wrong type.
            ValueError: If the window or pulse positions are inconsistent.
        """
        if isinstance(self.ion, bool) or not isinstance(self.ion, int):
            msg = "ion must be an integer"
            raise TypeError(msg)
        if len(self.window) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in self.window):
            msg = "window must contain two integers"
            raise TypeError(msg)
        if self.window[0] < 0 or self.window[1] < self.window[0]:
            msg = "window must be an ordered non-negative interval"
            raise ValueError(msg)
        if not isinstance(self.scheme_name, str):
            msg = "scheme_name must be a string"
            raise TypeError(msg)
        if not self.scheme_name:
            msg = "scheme_name must be non-empty"
            raise ValueError(msg)
        pulse_timesteps = tuple(self.pulse_timesteps)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in pulse_timesteps):
            msg = "pulse_timesteps must contain integers"
            raise TypeError(msg)
        if tuple(sorted(set(pulse_timesteps))) != pulse_timesteps:
            msg = "pulse_timesteps must be strictly increasing"
            raise ValueError(msg)
        if any(not self.window[0] <= value <= self.window[1] for value in pulse_timesteps):
            msg = "pulse timesteps must lie within the sequence window"
            raise ValueError(msg)
        action_ids = tuple(self.action_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in action_ids):
            msg = "action_ids must contain integers"
            raise TypeError(msg)
        if any(value < 0 for value in action_ids):
            msg = "action_ids must be non-negative"
            raise ValueError(msg)
        if len(set(action_ids)) != len(action_ids):
            msg = "action_ids must not contain duplicates"
            raise ValueError(msg)
        if len(action_ids) != len(pulse_timesteps):
            msg = "action_ids must contain one identifier for every pulse timestep"
            raise ValueError(msg)
        object.__setattr__(self, "pulse_timesteps", pulse_timesteps)
        object.__setattr__(self, "action_ids", action_ids)

    def to_dict(self) -> dict[str, object]:
        """Return this local sequence using JSON-compatible values."""
        result: dict[str, object] = {
            "ion": self.ion,
            "window": list(self.window),
            "scheme_name": self.scheme_name,
            "pulse_timesteps": list(self.pulse_timesteps),
            "action_ids": list(self.action_ids),
            "remaining_phase": self.remaining_phase,
            "phase_reduction": self.phase_reduction,
        }
        optional_values = {
            "residual_phase_at_window_end_before": self.residual_phase_at_window_end_before,
            "residual_phase_at_window_end_after": self.residual_phase_at_window_end_after,
            "residual_phase_at_window_end_reduction": self.residual_phase_at_window_end_reduction,
        }
        result.update({key: value for key, value in optional_values.items() if value is not None})
        return result

    @classmethod
    def from_dict(cls, data: object) -> LocalDDSequence:
        """Restore a local sequence from JSON-compatible values.

        Returns:
            The restored local DD sequence.

        Raises:
            ValueError: If the serialized sequence is malformed.
        """
        mapping = require_mapping(data, "local DD sequence")
        window = require_int_list(mapping, "window")
        if len(window) != 2:
            msg = "window must contain two integers"
            raise ValueError(msg)
        return cls(
            ion=require_int(mapping, "ion"),
            window=(window[0], window[1]),
            scheme_name=require_str(mapping, "scheme_name"),
            pulse_timesteps=tuple(require_int_list(mapping, "pulse_timesteps")),
            action_ids=tuple(require_int_list(mapping, "action_ids")),
            remaining_phase=require_number(mapping, "remaining_phase", default=0.0),
            phase_reduction=require_number(mapping, "phase_reduction", default=0.0),
            residual_phase_at_window_end_before=require_optional_number(mapping, "residual_phase_at_window_end_before"),
            residual_phase_at_window_end_after=require_optional_number(mapping, "residual_phase_at_window_end_after"),
            residual_phase_at_window_end_reduction=require_optional_number(
                mapping, "residual_phase_at_window_end_reduction"
            ),
        )


@dataclass(frozen=True)
class DDPassResult(Generic[ReportT]):
    """Contain an augmented schedule and diagnostics from one DD pass."""

    schedule: ActionSchedule
    report: ReportT
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the schedule, report, and optional explanation.

        Raises:
            TypeError: If the schedule or report violates the pass-result protocol.
            ValueError: If an unavailable reason is empty.
        """
        if not isinstance(self.schedule, ActionSchedule):
            msg = "schedule must be an ActionSchedule"
            raise TypeError(msg)
        if not isinstance(getattr(self.report, "report_type", None), str):
            msg = "report must define a string report_type"
            raise TypeError(msg)
        if not callable(getattr(self.report, "to_dict", None)):
            msg = "report must define to_dict"
            raise TypeError(msg)
        if self.unavailable_reason is not None and not self.unavailable_reason:
            msg = "unavailable_reason must be non-empty or None"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return the DD-pass result using JSON-compatible values."""
        return {
            "schedule": self.schedule.to_dict(),
            "report_type": self.report.report_type,
            "report": self.report.to_dict(),
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        report_class: type[ReportT],
    ) -> DDPassResult[ReportT]:
        """Restore a DD-pass result with an explicit report class.

        Returns:
            The restored DD-pass result.

        Raises:
            ValueError: If the serialized envelope or report type is malformed.
        """
        mapping = require_mapping(data, "serialized DD-pass result")
        if mapping.get("report_type") != report_class.report_type:
            msg = f"expected report_type {report_class.report_type!r}"
            raise ValueError(msg)
        unavailable_reason = mapping.get("unavailable_reason")
        if unavailable_reason is not None and not isinstance(unavailable_reason, str):
            msg = "unavailable_reason must be a string or null"
            raise ValueError(msg)
        return cls(
            schedule=ActionSchedule.from_dict(mapping.get("schedule")),
            report=cast("ReportT", report_class.from_dict(mapping.get("report"))),
            unavailable_reason=unavailable_reason,
        )

    def to_json(self) -> str:
        """Serialize this DD-pass result as JSON text.

        Returns:
            The JSON document.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str, report_class: type[ReportT]) -> DDPassResult[ReportT]:
        """Restore a DD-pass result from JSON text.

        Returns:
            The restored DD-pass result.
        """
        return cls.from_dict(json.loads(raw), report_class)

    def save(self, filename: str | Path) -> Path:
        """Write this DD-pass result to an explicit UTF-8 JSON file.

        Returns:
            The path written.
        """
        output_path = Path(filename)
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")
        return output_path


__all__ = ["DDPassResult", "DDReport", "LocalDDSequence"]
