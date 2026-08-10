# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Outcomes returned by the Linear compiler search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.actions import Action
    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.state import State


class CompilationStatus(str, Enum):
    """Describe how compilation ended."""

    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class CompilationResult:
    """Contain a schedule and the state reached by the search."""

    status: CompilationStatus
    path: list[Action]
    num_timesteps: int
    wall_clock_s: float = 0.0
    score: int | None = None
    final_state: State | None = None
    architecture: Architecture | None = None
    initial_state: State | None = None
    explored_nodes: int | None = None


__all__ = ["CompilationResult", "CompilationStatus"]
