# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused orchestration and validation tests for SADD."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from math import pi
from typing import TYPE_CHECKING

import pytest

from mqt.ionshuttler import visualize
from mqt.ionshuttler.linear import GateTiming, TransportTiming
from mqt.ionshuttler.linear.actions import Rx, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.dd import SADDConfig, SADDMethod, SADDReport, SADDResult, run_sadd
from mqt.ionshuttler.linear.dd import sadd as sadd_module
from mqt.ionshuttler.linear.dd import sadd_solver as sadd_solver_module
from mqt.ionshuttler.linear.dd.sadd_solver import (
    SADDProblem,
    SADDSolution,
    build_sadd_problem,
)
from mqt.ionshuttler.linear.dd.schedule_transform import insert_action_at_time
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.schedule import Schedule, schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state

if TYPE_CHECKING:
    from pathlib import Path


def _idle_result(
    architecture: LinearArchitecture,
    *,
    initial_positions: list[int],
    timesteps: int,
) -> Schedule:
    return schedule_from_path(
        [AdvanceTime() for _ in range(timesteps)],
        create_initial_state(
            len(initial_positions),
            architecture,
            initial_positions=initial_positions,
        ),
        architecture,
    )


def _compilation(schedule: Schedule, architecture: LinearArchitecture) -> CompilationResult:
    return CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=schedule,
        architecture=architecture,
        final_state=architecture.replay_schedule(schedule),
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
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    result = _idle_result(architecture, initial_positions=[0], timesteps=1)

    compilation = _compilation(result, architecture)
    output = run_sadd(compilation, SADDMethod.PULSE_ONLY)

    assert output.result is compilation
    assert output.report.opportunities == ()
    assert output.unavailable_reason is None


def test_sadd_result_round_trips_the_transformed_compilation() -> None:
    """Preserve the visualizable compilation artifact with the SADD report."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    compilation = _compilation(_idle_result(architecture, initial_positions=[0], timesteps=1), architecture)

    output = run_sadd(compilation, SADDMethod.PULSE_ONLY)
    restored = SADDResult.from_json(output.to_json())

    assert restored == output
    assert restored.result.schedule == compilation.schedule


def test_sadd_result_validates_fields_json_and_file_persistence(tmp_path: Path) -> None:
    """Reject malformed pass results and persist a valid result with a JSON suffix."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    compilation = _compilation(_idle_result(architecture, initial_positions=[0], timesteps=1), architecture)
    report = SADDReport(SADDMethod.PULSE_ONLY)

    with pytest.raises(TypeError, match="result must be a CompilationResult"):
        SADDResult(object(), report)  # ty: ignore[invalid-argument-type] - Runtime validation test.
    with pytest.raises(TypeError, match="report must be a SADDReport"):
        SADDResult(compilation, object())  # ty: ignore[invalid-argument-type] - Runtime validation test.
    with pytest.raises(ValueError, match="unavailable_reason must be non-empty"):
        SADDResult(compilation, report, "")
    with pytest.raises(ValueError, match="SADD result must be a JSON object"):
        SADDResult.from_dict([])
    with pytest.raises(ValueError, match="unavailable_reason must be a string or null"):
        SADDResult.from_dict({"result": compilation.to_dict(), "report": report.to_dict(), "unavailable_reason": 1})

    output = SADDResult(compilation, report).save(tmp_path / "nested" / "result")
    assert output == tmp_path / "nested" / "result.json"
    assert output.is_file()


def test_run_sadd_rejects_invalid_public_arguments() -> None:
    """Validate the compilation artifact and method before schedule analysis."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    compilation = _compilation(_idle_result(architecture, initial_positions=[0], timesteps=1), architecture)

    with pytest.raises(TypeError, match="result must be a CompilationResult"):
        run_sadd(object(), SADDMethod.PULSE_ONLY)  # ty: ignore[invalid-argument-type] - Runtime validation test.
    with pytest.raises(TypeError, match="method must be a SADDMethod"):
        run_sadd(compilation, "pulse_only_sadd")  # ty: ignore[invalid-argument-type] - Runtime validation test.


def test_sadd_result_is_directly_visualizable() -> None:
    """Expose the transformed compilation through the standard result view."""
    pyplot = pytest.importorskip("matplotlib.pyplot")
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    compilation = _compilation(_idle_result(architecture, initial_positions=[0], timesteps=1), architecture)

    output = run_sadd(compilation, SADDMethod.PULSE_ONLY)
    figure = visualize(output.result)

    assert figure is not None
    pyplot.close(figure)


def test_sadd_result_owns_an_explicit_architecture_override() -> None:
    """Carry a compatible DD architecture into the transformed result."""
    source_architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    dd_architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    compilation = _compilation(
        _idle_result(source_architecture, initial_positions=[0], timesteps=1),
        source_architecture,
    )

    output = run_sadd(compilation, SADDMethod.PULSE_ONLY, architecture=dd_architecture)

    assert output.result is not compilation
    assert output.result.architecture is dd_architecture
    output.result.validate()


def test_sadd_reports_unavailable_solver_without_mutating_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return dependency guidance as structured pass diagnostics."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1]})
    result = _idle_result(architecture, initial_positions=[1], timesteps=3)

    def unavailable(*args: object, **kwargs: object) -> SADDSolution:
        del args, kwargs
        msg = "install the dd extra"
        raise ImportError(msg)

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", unavailable)
    compilation = _compilation(result, architecture)
    output = run_sadd(compilation, SADDMethod.FULL)

    assert output.result is compilation
    assert output.report.opportunities == ()
    assert output.unavailable_reason == "install the dd extra"


@pytest.mark.parametrize("status", ["INFEASIBLE", "UNKNOWN"])
def test_sadd_records_unsolved_opportunity(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve infeasible and timeout-like solver outcomes without acceptance."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1]})
    result = _idle_result(architecture, initial_positions=[1], timesteps=3)

    def unsolved(*args: object, **kwargs: object) -> SADDSolution:
        del args, kwargs
        return _unsolved(status)

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", unsolved)

    compilation = _compilation(result, architecture)
    output = run_sadd(compilation, SADDMethod.PULSE_ONLY)

    assert output.result is compilation
    assert len(output.report.opportunities) == 1
    record = output.report.opportunities[0]
    assert record.status == status
    assert record.validation_status == "not_solved"
    assert not record.accepted


def test_sadd_rejects_replay_valid_non_improvement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require a strict objective improvement before replacing the schedule."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1]})
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
    compilation = _compilation(result, architecture)
    output = run_sadd(compilation, SADDMethod.PULSE_ONLY)

    assert output.result is compilation
    assert not output.report.opportunities[0].accepted


def test_sadd_method_is_the_only_transport_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass method transport policy unchanged to the shared backend."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1]})
    result = _idle_result(architecture, initial_positions=[1], timesteps=3)
    observed: list[bool] = []

    def capture(*args: object, allow_transport: bool, **kwargs: object) -> SADDSolution:
        del args, kwargs
        observed.append(allow_transport)
        return _unsolved("UNKNOWN")

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", capture)
    compilation = _compilation(result, architecture)
    run_sadd(compilation, SADDMethod.PULSE_ONLY)
    run_sadd(compilation, SADDMethod.FULL)

    assert observed == [False, True]


def test_sadd_synthesizes_operations_with_architecture_durations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give synthesized transport and pulses the durations defined by the architecture."""
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [1]},
        gate_timing=GateTiming(rx=2),
        transport_timing=TransportTiming(shuttle=2, swap=4),
    )
    schedule = schedule_from_path(
        [Shuttle(ion=0, src=0, dst=1), *(AdvanceTime() for _ in range(4))],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )
    observed: list[tuple[int, int, int]] = []

    def capture(problem: SADDProblem, **_kwargs: object) -> SADDSolution:
        observed.append((problem.shuttle_duration, problem.swap_duration, problem.pulse_duration))
        return _unsolved("UNKNOWN")

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", capture)

    run_sadd(_compilation(schedule, architecture), SADDMethod.FULL)

    assert observed == [(2, 4, 2)]


def test_sadd_requires_a_physical_rx_pulse() -> None:
    """Reject an architecture whose Rx rotation cannot act as a timed decoupling pulse."""
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [1]},
        gate_timing=GateTiming(rx=0, virtual_single_qubit_gates=frozenset({"rx", "rz"})),
    )
    schedule = _idle_result(architecture, initial_positions=[1], timesteps=3)

    with pytest.raises(ValueError, match="physical Rx pulse"):
        build_sadd_problem(schedule, architecture, target_pz="pz", t_start=0, t_end=3, participating_ions=(0,))


def test_participant_selection_reports_busy_ions_without_disqualifying_them() -> None:
    """Keep an ion that is busy for part of its window available to the solver.

    Occupancy is a per-timestep property. The control and operation-duration
    constraints already forbid a pulse while the ion is busy, so excluding the
    ion outright would discard the window's remaining usable boundaries.
    """
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        gate_timing=GateTiming(rx=2),
    )
    schedule = schedule_from_path(
        [Rx(ion=0, theta=pi), AdvanceTime(), AdvanceTime(), AdvanceTime(), AdvanceTime()],
        create_initial_state(1, architecture),
        architecture,
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
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [1]},
        field_profile=FieldProfile(num_sites=3, site_field=((0, 4.0), (1, 1.0), (2, 4.0))),
        transport_timing=TransportTiming(shuttle=2),
    )
    schedule = schedule_from_path(
        [
            AdvanceTime(),
            Shuttle(ion=0, src=0, dst=1),
            *(AdvanceTime() for _ in range(5)),
        ],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )

    output = run_sadd(
        _compilation(schedule, architecture),
        SADDMethod.FULL,
        SADDConfig(max_accepted_windows=1),
    )

    assert output.report.opportunities
    opportunity = output.report.opportunities[0]
    assert 0 in opportunity.busy_ions
    assert 0 in opportunity.participating_ions
    assert opportunity.accepted
    assert opportunity.pulse_timesteps is not None
    assert opportunity.pulse_timesteps[0]
    output.result.validate()
    assert output.result.schedule is not schedule


def test_sadd_reports_transport_changes_by_action_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Describe schedule transport changes rather than only synthesized actions."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1]})
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
    output = run_sadd(_compilation(schedule, architecture), SADDMethod.FULL)

    assert output.report.opportunities[0].transport_delta == {"Shuttle": 1}


def test_sadd_threads_accepted_pulse_identity_into_later_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat pulses accepted in an earlier window as DD during later analysis."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
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
        _compilation(schedule, architecture),
        SADDMethod.PULSE_ONLY,
        SADDConfig(min_window_length=2, max_window_length=2),
    )

    assert observed_prior_ids[0] == frozenset()
    first_pulse_action_ids = output.report.opportunities[0].pulse_action_ids
    assert first_pulse_action_ids is not None
    assert observed_prior_ids[1] == frozenset(first_pulse_action_ids[0])


def test_sadd_control_windows_start_at_the_absolute_schedule_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass absolute control-window bounds to each SADD problem."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    base = _idle_result(architecture, initial_positions=[0], timesteps=4)
    offset = 3
    shifted = Schedule(
        base.scheduled_actions,
        base.end_time + offset,
        replace(
            base.initial_state,
            time=offset,
            ions_busy_until=((0, offset),),
            pzs_busy_until=(("pz", offset),),
        ),
    )
    observed_windows: list[tuple[int, int]] = []

    def capture(problem: SADDProblem, **_kwargs: object) -> SADDSolution:
        observed_windows.append((problem.t_start, problem.t_end))
        return _unsolved("UNKNOWN")

    monkeypatch.setattr(sadd_module, "solve_sadd_problem", capture)

    run_sadd(
        _compilation(shifted, architecture),
        SADDMethod.PULSE_ONLY,
        SADDConfig(min_window_length=2, max_window_length=2),
    )

    assert observed_windows == [(3, 5), (5, 7)]


def test_problem_validation_and_final_slot_constraints() -> None:
    """Validate problem bounds and pin the closing placement obligation."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1]})
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
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1]})
    result = schedule_from_path(
        [AdvanceTime(), Rx(ion=0, theta=pi), AdvanceTime(), AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[1]),
        architecture,
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
