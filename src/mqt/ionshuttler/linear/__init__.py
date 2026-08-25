# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compile circuits for a linear ion-shuttling architecture."""

from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.config import (
    GateTiming,
    HardwareTiming,
    LinearCompilerConfig,
    SearchConfig,
    TransportTiming,
)
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.schedule import ActionSchedule, MachineState, ScheduledAction

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
    "ActionSchedule",
    "Architecture",
    "CompilationResult",
    "CompilationStatus",
    "GateTiming",
    "HardwareTiming",
    "LinearCompiler",
    "LinearCompilerConfig",
    "MachineState",
    "ScheduledAction",
    "SearchConfig",
    "TransportTiming",
]
