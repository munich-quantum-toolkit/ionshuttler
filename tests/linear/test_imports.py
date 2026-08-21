# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the supported Linear import surface."""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys


def test_linear_module_imports() -> None:
    """Import every Linear module without optional downstream dependencies."""
    package = importlib.import_module("mqt.ionshuttler.linear")
    module_names = [module.name for module in pkgutil.iter_modules(package.__path__, prefix=f"{package.__name__}.")]

    assert module_names
    assert all(importlib.import_module(module_name) is not None for module_name in module_names)


def test_package_exports_only_the_supported_facade() -> None:
    """Keep the package-level API small and intentional."""
    package = importlib.import_module("mqt.ionshuttler.linear")

    assert package.__all__ == [
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


def test_action_schedule_import_does_not_load_compiler_search() -> None:
    """Keep the execution boundary independent of compiler implementation modules."""
    command = (
        "import sys; "
        "from mqt.ionshuttler.linear.schedule import ActionSchedule; "
        "assert 'mqt.ionshuttler.linear.search' not in sys.modules"
    )
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - Fixed interpreter command.
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
