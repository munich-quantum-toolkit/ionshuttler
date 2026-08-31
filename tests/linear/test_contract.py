# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for stable Linear compiler contracts."""

from __future__ import annotations

import pytest

from mqt.ionshuttler.linear import Architecture, LinearCompiler, LinearCompilerConfig
from mqt.ionshuttler.linear.actions import AdvanceTime
from mqt.ionshuttler.linear.result import CompilationStatus


def test_production_defaults_are_explicit() -> None:
    """Keep the documented production search policy available without fixtures."""
    config = LinearCompilerConfig()

    assert config.search.horizon == 3
    assert config.search.committed_gates == 2
    assert config.search.iterative_diving_search
    assert not config.search.informed_action_prioritization
    assert config.search.max_frontier_size == 1000
    assert config.search.max_compile_time == pytest.approx(1800.0)


def test_compilation_stops_advancing_after_pending_work_finishes() -> None:
    """Advance time only while an operation or dependency remains pending."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [2, 3]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.1) q[0];\n'

    result = LinearCompiler(architecture).compile(qasm)

    assert result.status is CompilationStatus.SUCCESS
    assert isinstance(result.path[-1], AdvanceTime)
    assert sum(isinstance(action, AdvanceTime) for action in result.path) == result.num_timesteps
    assert result.final_state is not None
    assert result.final_state.time == result.num_timesteps
    assert all(free_time <= result.num_timesteps for _ion, free_time in result.final_state.ions_busy_until)
