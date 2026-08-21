# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optional-extra tests for the SADD CP-SAT model."""

from __future__ import annotations

import pytest

from mqt.ionshuttler.linear.actions import AdvanceTime, PhysicalSwap, Rx, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd import OperationDurations
from mqt.ionshuttler.linear.dd import sadd_solver as sadd_solver_module
from mqt.ionshuttler.linear.dd.sadd_solver import (
    SolverObject,
    build_sadd_problem,
    solve_sadd_problem,
)
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state

cp_model = pytest.importorskip("ortools.sat.python.cp_model")


def _idle_result(
    architecture: Architecture,
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


def test_transport_decoder_preserves_configured_durations() -> None:
    """Materialize reciprocal moves as swaps and remaining moves as shuttles."""
    actions = sadd_solver_module._ordered_transport_actions_for_transition(
        2,
        {0: 0, 1: 1, 2: 3},
        {0: 1, 1: 0, 2: 2},
        shuttle_duration=2,
        swap_duration=3,
    )
    assert actions == [
        (2, PhysicalSwap(0, 1, 0, 1, duration=3)),
        (2, Shuttle(2, 3, 2, duration=2)),
    ]


def test_solver_applies_terminal_parity_to_downstream_phase() -> None:
    """Keep the integer CP-SAT phase objective aligned with full replay."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    result = _idle_result(architecture, [0], 5)
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=1,
        t_end=3,
        participating_ions=(0,),
    )

    solution = solve_sadd_problem(problem, timeout_s=1.0, allow_transport=False)

    assert solution.status in {"OPTIMAL", "FEASIBLE"}
    assert solution.pulse_timesteps == {0: (2,)}
    assert solution.objective_before == pytest.approx(25.0)
    assert solution.objective_after == pytest.approx(1.0)
    assert solution.materialized


def test_equal_primary_optima_remain_semantically_equivalent() -> None:
    """Accept either primary optimum when symmetric pulse positions tie."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    result = _idle_result(architecture, [1], 3)
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=3,
        participating_ions=(0,),
    )

    solutions = [solve_sadd_problem(problem, allow_transport=False) for _ in range(10)]

    assert {solution.pulse_timesteps[0] for solution in solutions} <= {(1,), (2,)}
    assert all(solution.objective_after == pytest.approx(1.0) for solution in solutions)


def test_nonunit_operation_intervals_prevent_overlapping_starts() -> None:
    """Reserve shuttle, swap, and control resources for their full durations."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [0, 1, 2]})

    shuttle_model, shuttle_move, shuttles, _swaps, _controls = _duration_model(
        architecture,
        [0, 2],
        OperationDurations(shuttle=3, swap=3, one_qubit_gate=2),
    )
    shuttle_model.Add(shuttles[0, 0] == 1)
    shuttle_model.Add(shuttle_move[0, 1] == 1)
    assert cp_model.CpSolver().Solve(shuttle_model) == cp_model.INFEASIBLE

    swap_model, swap_move, _shuttles, swaps, _controls = _duration_model(
        architecture,
        [0, 1],
        OperationDurations(swap=3),
    )
    swap_model.Add(swaps[0, 1, 0] == 1)
    swap_model.Add(swap_move[0, 1] == 1)
    assert cp_model.CpSolver().Solve(swap_model) == cp_model.INFEASIBLE

    control_model, _moves, _shuttles, _swaps, controls = _duration_model(
        architecture,
        [1],
        OperationDurations(one_qubit_gate=2),
    )
    control_model.Add(controls[0, 0] == 1)
    control_model.Add(controls[0, 1] == 1)
    assert cp_model.CpSolver().Solve(control_model) == cp_model.INFEASIBLE


def _duration_model(
    architecture: Architecture,
    initial_positions: list[int],
    durations: OperationDurations,
) -> tuple[
    SolverObject,
    dict[tuple[int, int], SolverObject],
    dict[tuple[int, int], SolverObject],
    dict[tuple[int, int, int], SolverObject],
    dict[tuple[int, int], SolverObject],
]:
    result = _idle_result(architecture, initial_positions, 4)
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=4,
        participating_ions=tuple(range(len(initial_positions))),
        operation_durations=durations,
    )
    timeline = build_timeline(result, architecture)
    model = cp_model.CpModel()
    x = {}
    for ion in problem.participating_ions:
        for rel_t in range(problem.duration):
            occupancy = []
            for site in range(architecture.num_sites):
                variable = model.NewBoolVar(f"x_i{ion}_s{site}_t{rel_t}")
                x[ion, rel_t, site] = variable
                occupancy.append(variable)
            model.AddExactlyOne(occupancy)
    move = sadd_solver_module._add_position_and_movement_constraints(model, problem, timeline, x, allow_transport=True)
    shuttle, swap = sadd_solver_module._add_transport_kind_constraints(model, problem, x, move)
    control = sadd_solver_module._add_control_constraints(model, problem, timeline, x, move, allow_pulses=True)
    sadd_solver_module._add_operation_duration_constraints(model, problem, timeline, shuttle, swap, control)
    return model, move, shuttle, swap, control


def test_problem_infers_existing_transport_durations() -> None:
    """Use uniform schedule timings when no explicit synthesis timings are given."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    result = ActionSchedule.from_actions(
        [
            Shuttle(0, 0, 1, duration=2),
            AdvanceTime(),
            AdvanceTime(),
            PhysicalSwap(0, 1, 1, 2, duration=3),
            AdvanceTime(),
            AdvanceTime(),
            AdvanceTime(),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
    )
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=5,
        participating_ions=(0, 1),
    )
    assert problem.shuttle_duration == 2
    assert problem.swap_duration == 3


def test_materialized_pulse_uses_configured_duration() -> None:
    """Carry configured single-qubit duration into the augmented schedule."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    result = _idle_result(architecture, [0], 3)
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=3,
        participating_ions=(0,),
        operation_durations=OperationDurations(one_qubit_gate=2),
    )
    solution = solve_sadd_problem(problem, timeout_s=1.0, allow_transport=False)
    assert solution.schedule is not None
    assert any(isinstance(action, Rx) and action.duration == 2 for action in solution.schedule.path)
