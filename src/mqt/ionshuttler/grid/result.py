# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Grid diagnostics, compilation results, and result persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.result import CompilationResult as _CompilationResult
from mqt.ionshuttler.grid.actions import decode_grid_action
from mqt.ionshuttler.grid.architecture import GridArchitecture
from mqt.ionshuttler.grid.planning_trace import PlanningSnapshot
from mqt.ionshuttler.grid.state import GridMachineState

from .._json_utils import require_int, require_list, require_mapping, require_str


class GridStopReason(StrEnum):
    """State why the Grid compiler stopped scheduling.

    Attributes:
        COMPLETED: All gates and the optional target placement completed.
        NO_PROGRESS: The compiler found no gate and no movement to select,
            and no busy resource becomes free later.
        ITERATION_LIMIT: The compiler reached ``max_iterations``.
    """

    COMPLETED = "completed"
    NO_PROGRESS = "no_progress"
    ITERATION_LIMIT = "iteration_limit"


@dataclass(frozen=True)
class GridDiagnostics:
    """Store statistics from Grid compilation.

    Attributes:
        iterations: Number of scheduling iterations.
        explored_routing_states: Number of routing states and cycle-search steps
            that the router evaluated.
        junction_moves: Number of junction moves, including the moves inside
            cycles.
        cycles: Number of cycles.
        gates: Number of completed gates.
        stop_reason: Why scheduling stopped.
        planning_trace: One snapshot per scheduling decision, in decision
            order. ``None`` unless
            :attr:`~mqt.ionshuttler.grid.GridCompilerConfig.record_planning_trace`
            is set.
    """

    iterations: int
    explored_routing_states: int
    junction_moves: int
    cycles: int
    gates: int
    stop_reason: GridStopReason
    planning_trace: tuple[PlanningSnapshot, ...] | None = None

    def __post_init__(self) -> None:
        """Validate diagnostic counters, the stop reason, and the trace.

        Raises:
            TypeError: If a counter is not an integer, the stop reason is not
                a :class:`GridStopReason`, or the trace contains another value.
            ValueError: If a counter is negative.
        """
        for name in ("iterations", "explored_routing_states", "junction_moves", "cycles", "gates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                msg = f"{name} must be an integer"
                raise TypeError(msg)
            if value < 0:
                msg = f"{name} must be non-negative"
                raise ValueError(msg)
        if not isinstance(self.stop_reason, GridStopReason):
            msg = "stop_reason must be a GridStopReason"
            raise TypeError(msg)
        if self.planning_trace is not None:
            trace = tuple(self.planning_trace)
            if any(not isinstance(snapshot, PlanningSnapshot) for snapshot in trace):
                msg = "planning_trace must contain PlanningSnapshot values"
                raise TypeError(msg)
            object.__setattr__(self, "planning_trace", trace)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible diagnostics.

        The planning trace appears only when it was recorded.
        """
        data: dict[str, object] = {
            "iterations": self.iterations,
            "explored_routing_states": self.explored_routing_states,
            "junction_moves": self.junction_moves,
            "cycles": self.cycles,
            "gates": self.gates,
            "stop_reason": self.stop_reason.value,
        }
        if self.planning_trace is not None:
            data["planning_trace"] = [snapshot.to_dict() for snapshot in self.planning_trace]
        return data

    @classmethod
    def from_dict(cls, data: object) -> GridDiagnostics:
        """Restore Grid diagnostics.

        Returns:
            The restored diagnostics.

        Raises:
            ValueError: If the stop reason is unknown.
        """
        mapping = require_mapping(data, "Grid diagnostics")
        stop_reason = require_str(mapping, "stop_reason")
        try:
            reason = GridStopReason(stop_reason)
        except ValueError as error:
            msg = f"stop_reason has unknown value {stop_reason!r}"
            raise ValueError(msg) from error
        return cls(
            iterations=require_int(mapping, "iterations"),
            explored_routing_states=require_int(mapping, "explored_routing_states"),
            junction_moves=require_int(mapping, "junction_moves"),
            cycles=require_int(mapping, "cycles"),
            gates=require_int(mapping, "gates"),
            stop_reason=reason,
            planning_trace=(
                None
                if mapping.get("planning_trace") is None
                else tuple(PlanningSnapshot.from_dict(item) for item in require_list(mapping, "planning_trace"))
            ),
        )


GridCompilationResult: TypeAlias = _CompilationResult[GridArchitecture, Action, GridMachineState, GridDiagnostics]


def result_from_dict(data: object) -> GridCompilationResult:
    """Restore a Grid compilation result.

    Returns:
        The restored result.
    """
    return _CompilationResult.from_dict(
        data,
        decode_architecture=GridArchitecture.from_dict,
        decode_action=decode_grid_action,
        decode_state=GridMachineState.from_dict,
        decode_diagnostics=GridDiagnostics.from_dict,
    )


def result_from_json(raw: str) -> GridCompilationResult:
    """Restore a Grid compilation result from JSON text.

    Returns:
        The restored result.
    """
    return result_from_dict(json.loads(raw))


def load_result(filename: str | Path) -> GridCompilationResult:
    """Load a Grid compilation result from a UTF-8 JSON file.

    Returns:
        The restored result.
    """
    return result_from_json(Path(filename).read_text(encoding="utf-8"))


__all__ = [
    "GridCompilationResult",
    "GridDiagnostics",
    "GridStopReason",
    "load_result",
    "result_from_dict",
    "result_from_json",
]
