# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Linear diagnostics, compilation results, and result persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.actions import decode_linear_action
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.schedule import LinearMachineState

from .._json_utils import require_int, require_list, require_mapping


@dataclass(frozen=True)
class LinearDiagnostics:
    """Store statistics specific to Linear search."""

    score: int
    explored_nodes: int
    preferred_gate_zones: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate search statistics.

        Raises:
            TypeError: If a statistic is not an integer.
            ValueError: If a statistic is negative.
        """
        for name, value in (("score", self.score), ("explored_nodes", self.explored_nodes)):
            if isinstance(value, bool) or not isinstance(value, int):
                msg = f"{name} must be an integer"
                raise TypeError(msg)
            if value < 0:
                msg = f"{name} must be non-negative"
                raise ValueError(msg)
        preferred_gate_zones = tuple(sorted(self.preferred_gate_zones))
        gate_ids: set[int] = set()
        for gate_id, processing_zone_id in preferred_gate_zones:
            if isinstance(gate_id, bool) or not isinstance(gate_id, int):
                msg = "preferred gate IDs must be integers"
                raise TypeError(msg)
            if gate_id < 0:
                msg = "preferred gate IDs must be non-negative"
                raise ValueError(msg)
            if not isinstance(processing_zone_id, str):
                msg = "preferred processing-zone IDs must be strings"
                raise TypeError(msg)
            if not processing_zone_id:
                msg = "preferred processing-zone IDs must be non-empty"
                raise ValueError(msg)
            if gate_id in gate_ids:
                msg = "preferred gate IDs must be unique"
                raise ValueError(msg)
            gate_ids.add(gate_id)
        object.__setattr__(self, "preferred_gate_zones", preferred_gate_zones)

    def to_dict(self) -> dict[str, object]:
        """Return these diagnostics using JSON-compatible values."""
        return {
            "score": self.score,
            "explored_nodes": self.explored_nodes,
            "preferred_gate_zones": [list(assignment) for assignment in self.preferred_gate_zones],
        }

    @classmethod
    def from_dict(cls, data: object) -> LinearDiagnostics:
        """Restore Linear diagnostics.

        Returns:
            The restored diagnostics.

        Raises:
            ValueError: If a preferred gate-zone assignment is malformed.
        """
        mapping = require_mapping(data, "linear diagnostics")
        preferred_gate_zones: list[tuple[int, str]] = []
        for raw_assignment in require_list(mapping, "preferred_gate_zones"):
            if not isinstance(raw_assignment, list) or len(raw_assignment) != 2:
                msg = "preferred_gate_zones entries must be [gate_id, processing_zone_id] pairs"
                raise ValueError(msg)
            gate_id, processing_zone_id = raw_assignment
            if isinstance(gate_id, bool) or not isinstance(gate_id, int):
                msg = "preferred gate IDs must be integers"
                raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
            if not isinstance(processing_zone_id, str):
                msg = "preferred processing-zone IDs must be strings"
                raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
            preferred_gate_zones.append((gate_id, processing_zone_id))
        return cls(
            score=require_int(mapping, "score"),
            explored_nodes=require_int(mapping, "explored_nodes"),
            preferred_gate_zones=tuple(preferred_gate_zones),
        )


LinearCompilationResult: TypeAlias = CompilationResult[
    LinearArchitecture, Action, LinearMachineState, LinearDiagnostics
]


def result_from_dict(data: object) -> LinearCompilationResult:
    """Restore a Linear compilation result from its versioned JSON representation.

    Returns:
        The restored compilation result.
    """
    return CompilationResult.from_dict(
        data,
        decode_architecture=LinearArchitecture.from_dict,
        decode_action=decode_linear_action,
        decode_state=LinearMachineState.from_dict,
        decode_diagnostics=LinearDiagnostics.from_dict,
    )


def result_from_json(raw: str) -> LinearCompilationResult:
    """Restore a Linear compilation result from JSON text.

    Returns:
        The restored compilation result.
    """
    return result_from_dict(json.loads(raw))


def load_result(filename: str | Path) -> LinearCompilationResult:
    """Load a Linear compilation result from a UTF-8 JSON file.

    Returns:
        The restored compilation result.
    """
    return result_from_json(Path(filename).read_text(encoding="utf-8"))


__all__ = [
    "CompilationResult",
    "CompilationStatus",
    "LinearCompilationResult",
    "LinearDiagnostics",
    "load_result",
    "result_from_dict",
    "result_from_json",
]
