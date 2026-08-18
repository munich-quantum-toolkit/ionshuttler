# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared result shape for Linear dynamical-decoupling passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ProgramT = TypeVar("ProgramT")
ReportT = TypeVar("ReportT")


@dataclass(frozen=True)
class DDPassResult(Generic[ProgramT, ReportT]):
    """Contain an augmented program and diagnostics from one DD pass.

    Args:
        program: New executable program returned by the pass.
        report: Method-specific diagnostics for the pass.
        unavailable_reason: Explanation when an optional backend was unavailable.
    """

    program: ProgramT
    report: ReportT
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the optional unavailability explanation.

        Raises:
            ValueError: If an empty unavailability explanation is supplied.
        """
        if self.unavailable_reason is not None and not self.unavailable_reason:
            msg = "unavailable_reason must be non-empty or None"
            raise ValueError(msg)
