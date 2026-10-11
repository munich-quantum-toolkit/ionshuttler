# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the supported Grid import surface."""

from __future__ import annotations

import importlib


def test_package_exports_the_hardware_model() -> None:
    """Keep the package-level Grid API small and intentional."""
    package = importlib.import_module("mqt.ionshuttler.grid")

    assert package.__all__ == [
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

    assert package.GridCompiler.__name__ == "GridCompiler"
