# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused orchestration and validation tests for SADD."""

from __future__ import annotations

import importlib.util
from math import pi

import pytest

from mqt.ionshuttler.linear.actions import AdvanceTime, Rx, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd import SADDConfig, SADDMethod, run_sadd
from mqt.ionshuttler.linear.dd import sadd as sadd_module
from mqt.ionshuttler.linear.dd import sadd_solver as sadd_solver_module
from mqt.ionshuttler.linear.dd.sadd_solver import (
    SADDProblem,
    SADDSolution,
    build_sadd_problem,
)
from mqt.ionshuttler.linear.dd.schedule_transform import insert_action_at_time
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state


def _idle_result(
    architecture: Architecture,
    *,
    initial_positions: list[int],
    timesteps: int,
) -> ActionSchedule:
    return ActionSchedule.from_actions(
        [AdvanceTime() for _ in range(timesteps)],
        create_initial_state(
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
        pulse_action_ids={},
        transport_actions=(),
        schedule=None,
        validation_status="not_solved",
        validation_error=None,
        runtime_s=0.0,
    )


def test_sadd_returns_unchanged_noop_without_an_eligible_window() -> None:
    """Avoid loading the optional solver when no opportunity exists."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    result = _idle_result(architecture, initial_positions=[0], timesteps=1)

    output = run_sadd(result, architecture, SADDMethod.PULSE_ONLY)

    assert output.schedule is result
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
    output = run_sadd(result, architecture, SADDMethod.FULL)

    assert output.schedule is result
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

    output = run_sadd(result, architecture, SADDMethod.PULSE_ONLY)

    assert output.schedule is result
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
        program = problem.schedule
        return SADDSolution(
            status="OPTIMAL",
            objective_before=objective,
            objective_after=objective,
            trajectories={0: (1, 1, 1)},
            pulse_timesteps={0: ()},
            pulse_action_ids={0: ()},
            transport_actions=(),
            schedule=program,
            validation_status="valid",
            validation_error=None,
            runtime_s=0.0,
        )

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", unchanged)
    output = run_sadd(result, architecture, SADDMethod.PULSE_ONLY)

    assert output.schedule is result
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
    run_sadd(result, architecture, SADDMethod.PULSE_ONLY)
    run_sadd(result, architecture, SADDMethod.FULL)

    assert observed == [False, True]


def test_participant_selection_reports_busy_ions_without_disqualifying_them() -> None:
    """Keep an ion that is busy for part of its window available to the solver.

    Occupancy is a per-timestep property. The control and operation-duration
    constraints already forbid a pulse while the ion is busy, so excluding the
    ion outright would discard the window's remaining usable boundaries.
    """
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    schedule = ActionSchedule.from_actions(
        [Rx(ion=0, theta=pi, duration=2), AdvanceTime(), AdvanceTime(), AdvanceTime(), AdvanceTime()],
        create_initial_state(1, architecture),
    )

    selection = sadd_module._select_participating_ions(
        schedule,
        architecture,
        "pz",
        (0, 4),
        SADDConfig(),
        frozenset(),
    )

    assert selection.busy_ions == (0,)
    assert selection.eligible_ions == (0,)
    assert selection.selected_ions == (0,)


@pytest.mark.skipif(importlib.util.find_spec("ortools") is None, reason="OR-Tools not installed")
def test_sadd_optimizes_a_window_in_which_every_ion_is_partly_busy() -> None:
    """Keep SADD effective on schedules whose ions all carry gates or transport."""
    architecture = Architecture(
        num_sites=3,
        processing_zones={"pz": [1]},
        field_profile=FieldProfile(num_sites=3, site_field=((0, 4.0), (1, 1.0), (2, 4.0))),
    )
    schedule = ActionSchedule.from_actions(
        [
            AdvanceTime(),
            Shuttle(ion=0, src=0, dst=1, duration=2),
            *(AdvanceTime() for _ in range(5)),
        ],
        create_initial_state(1, architecture, initial_positions=[0]),
    )

    output = run_sadd(schedule, architecture, SADDMethod.FULL, SADDConfig(max_accepted_windows=1))

    assert output.report.opportunities
    opportunity = output.report.opportunities[0]
    assert 0 in opportunity.rejected_busy_ions
    assert 0 in opportunity.participating_ions
    assert opportunity.accepted
    assert opportunity.pulse_timesteps is not None
    assert opportunity.pulse_timesteps[0]


def test_sadd_reports_transport_changes_by_action_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Describe schedule transport changes rather than only synthesized actions."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    schedule = _idle_result(architecture, initial_positions=[0], timesteps=3)

    def add_shuttle(problem: SADDProblem, **_kwargs: object) -> SADDSolution:
        updated = insert_action_at_time(
            problem.schedule,
            problem.architecture,
            problem.t_start,
            Shuttle(ion=0, src=0, dst=1),
        )
        return SADDSolution(
            status="OPTIMAL",
            objective_before=problem.objective_before,
            objective_after=max(0.0, problem.objective_before - 1.0),
            trajectories={0: (1, 1, 1)},
            pulse_timesteps={0: ()},
            pulse_action_ids={0: ()},
            transport_actions=((problem.t_start, Shuttle(ion=0, src=0, dst=1)),),
            schedule=updated,
            validation_status="valid",
            validation_error=None,
            runtime_s=0.0,
        )

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", add_shuttle)
    output = run_sadd(schedule, architecture, SADDMethod.FULL)

    assert output.report.opportunities[0].transport_delta == {"Shuttle": 1}


def test_sadd_threads_accepted_pulse_identity_into_later_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat pulses accepted in an earlier window as DD during later analysis."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    schedule = _idle_result(architecture, initial_positions=[0], timesteps=4)
    observed_prior_ids: list[frozenset[int]] = []

    def insert_one_pulse(problem: SADDProblem, **_kwargs: object) -> SADDSolution:
        observed_prior_ids.append(problem.local_pulse_action_ids)
        updated = insert_action_at_time(
            problem.schedule,
            problem.architecture,
            problem.t_start,
            Rx(ion=0, theta=pi),
        )
        pulse_id = updated.next_action_id - 1
        return SADDSolution(
            status="OPTIMAL",
            objective_before=problem.objective_before,
            objective_after=max(0.0, problem.objective_before - 1.0),
            trajectories={0: tuple(0 for _ in range(problem.duration))},
            pulse_timesteps={0: (problem.t_start,)},
            pulse_action_ids={0: (pulse_id,)},
            transport_actions=(),
            schedule=updated,
            validation_status="valid",
            validation_error=None,
            runtime_s=0.0,
        )

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", insert_one_pulse)
    output = run_sadd(
        schedule,
        architecture,
        SADDMethod.PULSE_ONLY,
        SADDConfig(min_window_length=2, max_window_length=2),
    )

    assert observed_prior_ids[0] == frozenset()
    first_pulse_action_ids = output.report.opportunities[0].pulse_action_ids
    assert first_pulse_action_ids is not None
    assert observed_prior_ids[1] == frozenset(first_pulse_action_ids[0])


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
        build_sadd_problem(
            result,
            architecture,
            target_pz="missing",
            t_start=0,
            t_end=2,
            participating_ions=(0,),
        )
    with pytest.raises(ValueError, match="participating_ions"):
        build_sadd_problem(result, architecture, target_pz="pz", t_start=0, t_end=2, participating_ions=())


def test_invalid_materialization_reports_replay_failure() -> None:
    """Reject a decoded schedule that conflicts with an algorithmic gate."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    result = ActionSchedule.from_actions(
        [AdvanceTime(), Rx(ion=0, theta=pi), AdvanceTime(), AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[1]),
    )
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=3,
        participating_ions=(0,),
    )

    materialized, pulse_action_ids, validation_status, validation_error = sadd_solver_module._materialize_solution(
        problem,
        ((0, Shuttle(ion=0, src=1, dst=0)), (2, Shuttle(ion=0, src=0, dst=1))),
        {},
    )

    assert materialized is None
    assert pulse_action_ids == {}
    assert validation_status == "invalid"
    assert validation_error == "decoded solution fails full schedule replay validation"
