# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Segment-graph hardware models and actions."""

from typing import TYPE_CHECKING

from mqt.ionshuttler.core.gates import GateTiming
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction

from .actions import Cycle, JunctionMove
from .architecture import GridArchitecture
from .config import GridCompilerConfig, GridCompilerStrategy, GridDependencyMode
from .model import (
    Junction,
    ProcessingZone,
    Segment,
    SegmentEndpoint,
    SegmentOccupancy,
    TransportTiming,
)
from .result import (
    GridCompilationResult,
    GridDiagnostics,
    GridStopReason,
    load_result,
    result_from_dict,
    result_from_json,
)
from .schedule import load_schedule, schedule_from_dict, schedule_from_json
from .state import GridMachineState

if TYPE_CHECKING:
    from .compiler import GridCompiler


def __getattr__(name: str) -> object:
    """Resolve compiler entry points only when requested.

    Returns:
        The requested public object.

    Raises:
        AttributeError: If ``name`` is not a deferred public object.
    """
    if name == "GridCompiler":
        from .compiler import GridCompiler  # ruff: ignore[import-outside-top-level] - Keep the root import light.

        return GridCompiler
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "CompilationResult",
    "CompilationStatus",
    "Cycle",
    "GateTiming",
    "GridArchitecture",
    "GridCompilationResult",
    "GridCompiler",
    "GridCompilerConfig",
    "GridCompilerStrategy",
    "GridDependencyMode",
    "GridDiagnostics",
    "GridMachineState",
    "GridStopReason",
    "Junction",
    "JunctionMove",
    "ProcessingZone",
    "Schedule",
    "ScheduledAction",
    "Segment",
    "SegmentEndpoint",
    "SegmentOccupancy",
    "TransportTiming",
    "load_result",
    "load_schedule",
    "result_from_dict",
    "result_from_json",
    "schedule_from_dict",
    "schedule_from_json",
]
