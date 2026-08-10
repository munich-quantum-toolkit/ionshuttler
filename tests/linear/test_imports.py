# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the supported Linear import surface."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "mqt.ionshuttler.linear",
        "mqt.ionshuttler.linear.actions",
        "mqt.ionshuttler.linear.architecture",
        "mqt.ionshuttler.linear.compiler",
        "mqt.ionshuttler.linear.config",
        "mqt.ionshuttler.linear.cost",
        "mqt.ionshuttler.linear.expand",
        "mqt.ionshuttler.linear.field_profile",
        "mqt.ionshuttler.linear.parser",
        "mqt.ionshuttler.linear.result",
        "mqt.ionshuttler.linear.search",
        "mqt.ionshuttler.linear.state",
        "mqt.ionshuttler.linear.validation",
    ],
)
def test_linear_module_imports(module_name: str) -> None:
    """Import every Linear module without optional downstream dependencies."""
    assert importlib.import_module(module_name) is not None


def test_package_exports_only_the_supported_facade() -> None:
    """Keep the package-level API small and intentional."""
    package = importlib.import_module("mqt.ionshuttler.linear")

    assert package.__all__ == [
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
