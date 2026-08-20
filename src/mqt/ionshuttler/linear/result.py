# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Linear compilation outcomes and their JSON representation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.schedule import ActionDecoder, ActionDecoders, ActionSchedule
from mqt.ionshuttler.linear.state import State

from .._json_utils import (
    require_int,
    require_int_list,
    require_int_pairs,
    require_mapping,
    require_number,
    require_optional_int,
    require_str_int_pairs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.linear.actions import Action
    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.schedule import MachineState


class CompilationStatus(str, Enum):
    """Describe how compilation ended."""

    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class CompilationResult:
    """Contain an executable schedule and compiler diagnostics.

    The action schedule owns hardware and execution information. This result
    owns search completion, cost, runtime, explored-node count, and the compiler
    state reached at termination.
    """

    status: CompilationStatus
    schedule: ActionSchedule
    wall_clock_s: float = 0.0
    score: int | None = None
    final_state: State | None = None
    explored_nodes: int | None = None

    def __post_init__(self) -> None:
        """Validate compiler diagnostics and the schedule boundary.

        Raises:
            TypeError: If a result field has the wrong type.
            ValueError: If a numeric diagnostic is outside its valid range.
        """
        if not isinstance(self.status, CompilationStatus):
            msg = "status must be a CompilationStatus"
            raise TypeError(msg)
        if not isinstance(self.schedule, ActionSchedule):
            msg = "schedule must be an ActionSchedule"
            raise TypeError(msg)
        if isinstance(self.wall_clock_s, bool) or not isinstance(self.wall_clock_s, int | float):
            msg = "wall_clock_s must be numeric"
            raise TypeError(msg)
        if self.wall_clock_s < 0.0 or not isfinite(self.wall_clock_s):
            msg = "wall_clock_s must be finite and non-negative"
            raise ValueError(msg)
        if self.score is not None and (isinstance(self.score, bool) or not isinstance(self.score, int)):
            msg = "score must be an integer or None"
            raise TypeError(msg)
        if self.final_state is not None and not isinstance(self.final_state, State):
            msg = "final_state must be a State or None"
            raise TypeError(msg)
        if self.explored_nodes is not None:
            if isinstance(self.explored_nodes, bool) or not isinstance(self.explored_nodes, int):
                msg = "explored_nodes must be an integer or None"
                raise TypeError(msg)
            if self.explored_nodes < 0:
                msg = "explored_nodes must be non-negative"
                raise ValueError(msg)

    @property
    def path(self) -> list[Action]:
        """A mutable copy of the schedule's ordered executable operations."""
        return list(self.schedule.path)

    @property
    def num_timesteps(self) -> int:
        """The schedule's makespan."""
        return self.schedule.num_timesteps

    @property
    def architecture(self) -> Architecture:
        """The schedule's Linear hardware model."""
        return self.schedule.architecture

    @property
    def initial_state(self) -> MachineState:
        """The schedule's machine-only initial state."""
        return self.schedule.initial_state

    @property
    def action_types(self) -> tuple[str, ...]:
        """The schedule's serialized hardware capability names."""
        return self.schedule.action_types

    def to_dict(self) -> dict[str, object]:
        """Return the compilation result using JSON-compatible values."""
        return {
            "schedule": self.schedule.to_dict(),
            "diagnostics": {
                "status": self.status.value,
                "wall_clock_s": self.wall_clock_s,
                "score": self.score,
                "final_state": None if self.final_state is None else _state_to_dict(self.final_state),
                "explored_nodes": self.explored_nodes,
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        action_types: Sequence[type[Action]] | None = None,
        action_decoders: ActionDecoders | None = None,
    ) -> CompilationResult:
        """Restore a result from its JSON-compatible representation.

        Returns:
            The restored compilation result.

        Raises:
            ValueError: If the document is malformed.
        """
        mapping = require_mapping(data, "serialized compilation result")
        diagnostics = require_mapping(mapping.get("diagnostics"), "diagnostics")
        status_raw = diagnostics.get("status")
        try:
            status = CompilationStatus(status_raw)
        except ValueError as error:
            msg = f"unknown compilation status: {status_raw!r}"
            raise ValueError(msg) from error
        final_state_data = diagnostics.get("final_state")
        return cls(
            status=status,
            schedule=ActionSchedule.from_dict(
                mapping.get("schedule"),
                action_types=action_types,
                action_decoders=action_decoders,
            ),
            wall_clock_s=require_number(diagnostics, "wall_clock_s"),
            score=require_optional_int(diagnostics, "score"),
            final_state=None if final_state_data is None else _state_from_dict(final_state_data),
            explored_nodes=require_optional_int(diagnostics, "explored_nodes"),
        )

    def to_json(self) -> str:
        """Serialize this compilation result as JSON text.

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
    ) -> CompilationResult:
        """Restore a result from JSON text.

        Returns:
            The restored compilation result.
        """
        return cls.from_dict(json.loads(raw), action_types=action_types, action_decoders=action_decoders)

    def save(
        self,
        filename: str | Path,
        directory: str | Path = "outputs/results/json",
    ) -> Path:
        """Write this result to a UTF-8 JSON file.

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

    @classmethod
    def load(
        cls,
        filename: str | Path,
        *,
        action_types: Sequence[type[Action]] | None = None,
        action_decoders: ActionDecoders | None = None,
    ) -> CompilationResult:
        """Load a result from a UTF-8 JSON file.

        Returns:
            The restored compilation result.
        """
        return cls.from_json(
            Path(filename).read_text(encoding="utf-8"),
            action_types=action_types,
            action_decoders=action_decoders,
        )


def _state_to_dict(state: State) -> dict[str, object]:
    return {
        "positions": [list(item) for item in state.positions],
        "completed_gates": sorted(state.completed_gates),
        "in_progress_gates": [list(item) for item in state.in_progress_gates],
        "ions_busy_until": [list(item) for item in state.ions_busy_until],
        "pzs_busy_until": [list(item) for item in state.pzs_busy_until],
        "time": state.time,
    }


def _state_from_dict(data: object) -> State:
    mapping = require_mapping(data, "final_state")
    return State(
        positions=tuple(require_int_pairs(mapping, "positions")),
        completed_gates=frozenset(require_int_list(mapping, "completed_gates")),
        in_progress_gates=tuple(require_int_pairs(mapping, "in_progress_gates")),
        ions_busy_until=tuple(require_int_pairs(mapping, "ions_busy_until")),
        pzs_busy_until=tuple(require_str_int_pairs(mapping, "pzs_busy_until")),
        time=require_int(mapping, "time"),
    )


__all__ = [
    "ActionDecoder",
    "CompilationResult",
    "CompilationStatus",
]
