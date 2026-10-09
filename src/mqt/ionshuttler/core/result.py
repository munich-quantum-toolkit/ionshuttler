# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared compilation outcomes and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.schedule import MachineState, Schedule

from .._json_utils import require_mapping, require_number

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

RESULT_SCHEMA = "mqt.ionshuttler.compilation_result"
RESULT_VERSION = 1


class CompilationStatus(StrEnum):
    """Describe how compilation ended."""

    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class Architecture(Protocol):
    """Describe result operations supplied by each concrete architecture."""

    def to_dict(self) -> dict[str, object]:
        """Return this architecture using JSON-compatible values."""

    def replay_schedule(self, schedule: Schedule) -> MachineState:
        """Replay a schedule and return its final machine state."""


class Diagnostics(Protocol):
    """Describe typed compiler-specific diagnostics."""

    def to_dict(self) -> dict[str, object]:
        """Return these diagnostics using JSON-compatible values."""


ArchitectureT = TypeVar("ArchitectureT", bound=Architecture)
ActionT = TypeVar("ActionT", bound=Action)
MachineStateT = TypeVar("MachineStateT", bound=MachineState)
DiagnosticsT = TypeVar("DiagnosticsT", bound=Diagnostics)


@dataclass(frozen=True)
class CompilationResult(Generic[ArchitectureT, ActionT, MachineStateT, DiagnosticsT]):
    """Contain a schedule, its hardware boundary states, and diagnostics.

    Equality ignores ``wall_clock_s``, so two compilations with the same
    outcome compare equal.
    """

    status: CompilationStatus
    schedule: Schedule[ActionT, MachineStateT]
    architecture: ArchitectureT
    final_state: MachineStateT
    wall_clock_s: float = field(default=0.0, compare=False)
    diagnostics: DiagnosticsT | None = None

    def __post_init__(self) -> None:
        """Validate common result fields.

        Raises:
            TypeError: If a result field has the wrong type.
            ValueError: If wall-clock time is invalid.
        """
        if not isinstance(self.status, CompilationStatus):
            msg = "status must be a CompilationStatus"
            raise TypeError(msg)
        if not isinstance(self.schedule, Schedule):
            msg = "schedule must be a Schedule"
            raise TypeError(msg)
        if isinstance(self.wall_clock_s, bool) or not isinstance(self.wall_clock_s, int | float):
            msg = "wall_clock_s must be numeric"
            raise TypeError(msg)
        if self.wall_clock_s < 0.0 or not isfinite(self.wall_clock_s):
            msg = "wall_clock_s must be finite and non-negative"
            raise ValueError(msg)

    @property
    def path(self) -> list[ActionT]:
        """A mutable copy of scheduled operations in timeline order."""
        return list(self.schedule.path)

    @property
    def end_time(self) -> int:
        """Absolute schedule completion time."""
        return self.schedule.end_time

    @property
    def start_time(self) -> int:
        """Absolute schedule start time."""
        return self.schedule.start_time

    @property
    def duration(self) -> int:
        """Number of elapsed schedule timesteps."""
        return self.schedule.duration

    @property
    def initial_state(self) -> MachineStateT:
        """Resolved initial machine state."""
        return self.schedule.initial_state

    def validate(self) -> None:
        """Validate and replay this compilation artifact.

        Raises:
            ValueError: If schedule replay does not produce ``final_state``.
        """
        replayed = self.architecture.replay_schedule(self.schedule)
        if replayed != self.final_state:
            msg = "schedule replay does not match final_state"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return the current versioned result representation."""
        diagnostics = self.diagnostics
        return {
            "schema": RESULT_SCHEMA,
            "version": RESULT_VERSION,
            "status": self.status.value,
            "wall_clock_s": self.wall_clock_s,
            "architecture": self.architecture.to_dict(),
            "schedule": self.schedule.to_dict(),
            "final_state": self.final_state.to_dict(),
            "diagnostics": None if diagnostics is None else diagnostics.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        decode_architecture: Callable[[object], ArchitectureT],
        decode_action: Callable[[Mapping[str, object]], ActionT],
        decode_state: Callable[[object], MachineStateT],
        decode_diagnostics: Callable[[object], DiagnosticsT],
    ) -> CompilationResult[ArchitectureT, ActionT, MachineStateT, DiagnosticsT]:
        """Restore a result from the current versioned representation.

        Each architecture supplies the decoders for its own values, for
        example through :func:`mqt.ionshuttler.linear.result_from_dict`.

        Args:
            data: Serialized compilation result.
            decode_architecture: Restores the serialized architecture.
            decode_action: Restores one serialized scheduled action.
            decode_state: Restores a serialized machine state.
            decode_diagnostics: Restores serialized method diagnostics.

        Returns:
            The restored result.

        Raises:
            ValueError: If the schema, version, or status is invalid.
        """
        mapping = require_mapping(data, "serialized compilation result")
        if mapping.get("schema") != RESULT_SCHEMA or mapping.get("version") != RESULT_VERSION:
            msg = "unsupported compilation result schema or version"
            raise ValueError(msg)
        try:
            status = CompilationStatus(mapping.get("status"))
        except ValueError as error:
            msg = f"unknown compilation status: {mapping.get('status')!r}"
            raise ValueError(msg) from error
        raw_diagnostics = mapping.get("diagnostics")
        return cls(
            status=status,
            architecture=decode_architecture(mapping.get("architecture")),
            schedule=Schedule.from_dict(
                mapping.get("schedule"),
                decode_action=decode_action,
                decode_state=decode_state,
            ),
            final_state=decode_state(mapping.get("final_state")),
            wall_clock_s=require_number(mapping, "wall_clock_s"),
            diagnostics=None if raw_diagnostics is None else decode_diagnostics(raw_diagnostics),
        )

    def to_json(self) -> str:
        """Serialize this result as JSON text.

        Returns:
            The JSON document.
        """
        return json.dumps(self.to_dict())

    def save(self, filename: str | Path) -> Path:
        """Write this result to a UTF-8 JSON file.

        Returns:
            The path written.
        """
        output_path = Path(filename)
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")
        return output_path


__all__ = [
    "RESULT_SCHEMA",
    "RESULT_VERSION",
    "Architecture",
    "CompilationResult",
    "CompilationStatus",
    "Diagnostics",
]
