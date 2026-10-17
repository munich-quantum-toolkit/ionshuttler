# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Topology acceptance tests for supported Grid compiler configurations."""

from __future__ import annotations

import pytest

from mqt.ionshuttler.grid import GridCompiler, GridCompilerConfig

from .topology_cases import TopologyCase, topology_cases

_SUPPORTED_CONFIGURATIONS = (pytest.param(GridCompilerConfig(), id="greedy"),)


@pytest.mark.parametrize("config", _SUPPORTED_CONFIGURATIONS)
@pytest.mark.parametrize("case", topology_cases(), ids=lambda case: case.name)
def test_supported_compiler_configuration_terminates_on_topology_case(
    case: TopologyCase,
    config: GridCompilerConfig,
) -> None:
    """Require supported configurations to return a valid result for each topology case."""
    result = GridCompiler(case.architecture, config).compile(
        case.qasm,
        initial_placement=case.initial_placement,
        pre_partition=False,
    )

    assert result.status is case.expected_status
    assert result.diagnostics is not None
    assert result.diagnostics.stop_reason is case.expected_stop_reason
    assert result.diagnostics.iterations <= config.max_iterations
    result.validate()
