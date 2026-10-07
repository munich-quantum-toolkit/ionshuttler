# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the supported Linear import surface."""

from __future__ import annotations

import importlib
import importlib.util
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
    assert not hasattr(package, "MachineState")


def test_linear_timing_types_belong_to_their_owners() -> None:
    """Gate timing is shared; transport timing lives beside Linear transport actions."""
    from mqt.ionshuttler.core.gates import GateTiming as SharedGateTiming
    from mqt.ionshuttler.linear import GateTiming, TransportTiming
    from mqt.ionshuttler.linear.actions import TransportTiming as ActionTransportTiming

    assert GateTiming is SharedGateTiming
    assert TransportTiming is ActionTransportTiming


def test_linear_architecture_has_only_its_explicit_name() -> None:
    """Expose the Linear architecture without a level-ambiguous alias."""
    from mqt.ionshuttler.linear import LinearArchitecture

    assert LinearArchitecture.__name__ == "LinearArchitecture"
    assert not hasattr(importlib.import_module("mqt.ionshuttler.linear"), "Architecture")


def test_removed_validation_and_timeline_forwarders_are_absent() -> None:
    """Do not retain duplicate validation entry points or the old timeline module."""
    from mqt.ionshuttler.linear import LinearArchitecture

    replay = importlib.import_module("mqt.ionshuttler.linear.replay")

    assert not hasattr(replay, "validate_schedule")
    assert not hasattr(LinearArchitecture, "validate_schedule")
    assert importlib.util.find_spec("mqt.ionshuttler.linear.dd.timeline") is None


def test_schedule_import_does_not_load_compiler_search() -> None:
    """Keep the execution boundary independent of compiler implementation modules."""
    command = (
        "import sys; "
        "from mqt.ionshuttler.linear.schedule import Schedule; "
        "assert 'mqt.ionshuttler.linear.search' not in sys.modules"
    )
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - Fixed interpreter command.
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
