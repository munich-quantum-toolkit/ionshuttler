# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compile circuits for a linear ion-shuttling architecture."""

from typing import TYPE_CHECKING

from mqt.ionshuttler.core.gates import GateTiming
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES, TransportTiming
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.config import LinearCompilerConfig, SearchConfig
from mqt.ionshuttler.linear.cost import HeuristicFn, zero_heuristic
from mqt.ionshuttler.linear.result import (
    LinearCompilationResult,
    LinearDiagnostics,
    load_result,
    result_from_dict,
    result_from_json,
)
from mqt.ionshuttler.linear.schedule import (
    LinearMachineState,
    load_schedule,
    schedule_from_dict,
    schedule_from_json,
)

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.compiler import LinearCompiler


def __getattr__(name: str) -> object:
    """Resolve compiler entry points only when requested.

    Returns:
        The requested public object.

    Raises:
        AttributeError: If ``name`` is not a deferred public object.
    """
    if name == "LinearCompiler":
        from mqt.ionshuttler.linear.compiler import (  # ruff: ignore[import-outside-top-level] - Deferred to keep the root import backend-neutral.
            LinearCompiler,
        )

        return LinearCompiler
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "DEFAULT_ACTION_TYPES",
    "CompilationResult",
    "CompilationStatus",
    "GateTiming",
    "HeuristicFn",
    "LinearArchitecture",
    "LinearCompilationResult",
    "LinearCompiler",
    "LinearCompilerConfig",
    "LinearDiagnostics",
    "LinearMachineState",
    "Schedule",
    "ScheduledAction",
    "SearchConfig",
    "TransportTiming",
    "load_result",
    "load_schedule",
    "result_from_dict",
    "result_from_json",
    "schedule_from_dict",
    "schedule_from_json",
    "zero_heuristic",
]
