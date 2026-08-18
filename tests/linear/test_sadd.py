# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused orchestration and validation tests for SADD."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import AdvanceTime, Rx, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd import SADDMethod, run_sadd
from mqt.ionshuttler.linear.dd import sadd as sadd_module
from mqt.ionshuttler.linear.dd import sadd_solver as sadd_solver_module
from mqt.ionshuttler.linear.dd.sadd_solver import (
    SADDProblem,
    SADDSolution,
    build_sadd_problem,
)
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.state import create_initial_state


def _idle_result(
    architecture: Architecture,
    *,
    initial_positions: list[int],
    timesteps: int,
) -> CompilationResult:
    return CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[AdvanceTime() for _ in range(timesteps)],
        num_timesteps=timesteps,
        architecture=architecture,
        initial_state=create_initial_state(
            len(initial_positions),
            architecture,
            initial_positions=initial_positions,
        ),
    )


def _unsolved(problem_status: str) -> SADDSolution:
    return SADDSolution(
        status=problem_status,
        objective_before=1.0,
        objective_after=None,
        trajectories={},
        pulse_timesteps={},
        transport_actions=(),
        result=None,
        validation_status="not_solved",
        validation_error=None,
        runtime_s=0.0,
    )


def test_sadd_returns_unchanged_noop_without_an_eligible_window() -> None:
    """Avoid loading the optional solver when no opportunity exists."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    result = _idle_result(architecture, initial_positions=[0], timesteps=1)

    output = run_sadd(result, SADDMethod.PULSE_ONLY)

    assert output.program is result
    assert output.report.opportunities == ()
    assert output.unavailable_reason is None


def test_sadd_reports_unavailable_solver_without_mutating_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return dependency guidance as structured pass diagnostics."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    result = _idle_result(architecture, initial_positions=[1], timesteps=3)

    def unavailable(*args: object, **kwargs: object) -> SADDSolution:
        del args, kwargs
        msg = "install the dd extra"
        raise ImportError(msg)

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", unavailable)
    output = run_sadd(result, SADDMethod.FULL)

    assert output.program is result
    assert output.report.opportunities == ()
    assert output.unavailable_reason == "install the dd extra"


@pytest.mark.parametrize("status", ["INFEASIBLE", "UNKNOWN"])
def test_sadd_records_unsolved_opportunity(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve infeasible and timeout-like solver outcomes without acceptance."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    result = _idle_result(architecture, initial_positions=[1], timesteps=3)

    def unsolved(*args: object, **kwargs: object) -> SADDSolution:
        del args, kwargs
        return _unsolved(status)

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", unsolved)

    output = run_sadd(result, SADDMethod.PULSE_ONLY)

    assert output.program is result
    assert len(output.report.opportunities) == 1
    record = output.report.opportunities[0]
    assert record.status == status
    assert record.validation_status == "not_solved"
    assert not record.accepted


def test_sadd_rejects_replay_valid_non_improvement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require a strict objective improvement before replacing the schedule."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    result = _idle_result(architecture, initial_positions=[1], timesteps=3)

    def unchanged(problem: SADDProblem, **kwargs: object) -> SADDSolution:
        del kwargs
        objective = problem.objective_before
        program = problem.result
        return SADDSolution(
            status="OPTIMAL",
            objective_before=objective,
            objective_after=objective,
            trajectories={0: (1, 1, 1)},
            pulse_timesteps={0: ()},
            transport_actions=(),
            result=program,
            validation_status="valid",
            validation_error=None,
            runtime_s=0.0,
        )

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", unchanged)
    output = run_sadd(result, SADDMethod.PULSE_ONLY)

    assert output.program is result
    assert not output.report.opportunities[0].accepted


def test_sadd_method_is_the_only_transport_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass method transport policy unchanged to the shared backend."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    result = _idle_result(architecture, initial_positions=[1], timesteps=3)
    observed: list[bool] = []

    def capture(*args: object, allow_transport: bool, **kwargs: object) -> SADDSolution:
        del args, kwargs
        observed.append(allow_transport)
        return _unsolved("UNKNOWN")

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", capture)
    run_sadd(result, SADDMethod.PULSE_ONLY)
    run_sadd(result, SADDMethod.FULL)

    assert observed == [False, True]


def test_problem_validation_and_final_slot_constraints() -> None:
    """Validate problem bounds and pin the closing placement obligation."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    result = _idle_result(architecture, initial_positions=[0, 2], timesteps=4)
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=1,
        t_end=3,
        participating_ions=(0, 1),
    )

    assert problem.fixed_positions[0, 2] == 0
    assert problem.fixed_positions[1, 2] == 2
    assert problem.objective_before > 0.0
    with pytest.raises(ValueError, match="unknown processing zone"):
        build_sadd_problem(result, architecture, target_pz="missing", t_start=0, t_end=2, participating_ions=(0,))
    with pytest.raises(ValueError, match="participating_ions"):
        build_sadd_problem(result, architecture, target_pz="pz", t_start=0, t_end=2, participating_ions=())


def test_invalid_materialization_reports_replay_failure() -> None:
    """Reject a decoded schedule that conflicts with an algorithmic gate."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[AdvanceTime(), Rx(ion=0, theta=pi), AdvanceTime(), AdvanceTime()],
        num_timesteps=3,
        architecture=architecture,
        initial_state=create_initial_state(1, architecture, initial_positions=[1]),
    )
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=3,
        participating_ions=(0,),
    )

    materialized, validation_status, validation_error = sadd_solver_module._materialize_solution(
        problem,
        ((0, Shuttle(ion=0, src=1, dst=0)), (2, Shuttle(ion=0, src=0, dst=1))),
        {},
    )

    assert materialized is None
    assert validation_status == "invalid"
    assert validation_error == "decoded solution fails full schedule replay validation"
