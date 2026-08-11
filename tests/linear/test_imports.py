# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the supported Linear import surface."""

from __future__ import annotations

import importlib
import pkgutil


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
        "Architecture",
        "CompilationResult",
        "CompilationStatus",
        "DDInsertionRecord",
        "GateTiming",
        "GlobalDDRecord",
        "HardwareTiming",
        "LinearCompiler",
        "LinearCompilerConfig",
        "SearchConfig",
        "TransportTiming",
    ]
