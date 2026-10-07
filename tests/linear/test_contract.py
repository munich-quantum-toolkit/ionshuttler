# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for stable Linear compiler contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mqt.ionshuttler.linear import LinearArchitecture, LinearCompiler, LinearCompilerConfig
from mqt.ionshuttler.linear.actions import Rx
from mqt.ionshuttler.linear.replay import replay_schedule
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
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2, 3]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.1) q[0];\n'

    result = LinearCompiler(architecture).compile(qasm)

    assert result.status is CompilationStatus.SUCCESS
    assert result.schedule.scheduled_actions[-1].end_time == result.end_time
    assert result.final_state is not None
    assert result.final_state.time == result.end_time
    assert all(free_time <= result.end_time for _ion, free_time in result.final_state.ions_busy_until)
    result.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("duration", 7, "scheduled duration"),
        ("processing_zone_id", "other", "scheduled processing zone"),
        ("processing_zone_id", None, "scheduled processing zone"),
    ],
)
def test_schedule_validation_checks_explicit_execution_choices(field: str, value: object, message: str) -> None:
    """Reject schedule metadata that disagrees with Linear execution."""
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.1) q[0];\n'
    result = LinearCompiler(architecture).compile(qasm, initial_placement=[0])
    gate_index = next(
        index for index, item in enumerate(result.schedule.scheduled_actions) if isinstance(item.action, Rx)
    )
    changed = replace(result.schedule.scheduled_actions[gate_index], **{field: value})
    scheduled_actions = list(result.schedule.scheduled_actions)
    scheduled_actions[gate_index] = changed
    invalid_schedule = replace(
        result.schedule,
        scheduled_actions=tuple(scheduled_actions),
        end_time=max(result.end_time, changed.end_time),
    )

    with pytest.raises(ValueError, match=message):
        replay_schedule(invalid_schedule, architecture)


def test_schedule_validation_rejects_a_gate_on_an_absent_ion() -> None:
    """Report an unschedulable action before inspecting its metadata."""
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.1) q[0];\n'
    result = LinearCompiler(architecture).compile(qasm, initial_placement=[0])
    gate_index = next(
        index for index, item in enumerate(result.schedule.scheduled_actions) if isinstance(item.action, Rx)
    )
    scheduled_actions = list(result.schedule.scheduled_actions)
    scheduled_actions[gate_index] = replace(scheduled_actions[gate_index], action=Rx(ion=7, theta=0.1))
    invalid_schedule = replace(result.schedule, scheduled_actions=tuple(scheduled_actions))

    with pytest.raises(ValueError, match="is not valid at time"):
        replay_schedule(invalid_schedule, architecture)
