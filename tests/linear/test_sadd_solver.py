# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optional-extra tests for the SADD CP-SAT model."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from mqt.ionshuttler.linear import GateTiming, TransportTiming
from mqt.ionshuttler.linear.actions import PhysicalSwap, Rx, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.dd import sadd_solver as sadd_solver_module
from mqt.ionshuttler.linear.dd.sadd_solver import (
    SolverObject,
    build_sadd_problem,
    solve_sadd_problem,
)
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.schedule import Schedule, schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state
from mqt.ionshuttler.linear.timeline import build_timeline

cp_model = pytest.importorskip("ortools.sat.python.cp_model")


def _idle_result(
    architecture: LinearArchitecture,
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


def test_transport_decoder_materializes_swaps_and_shuttles() -> None:
    """Materialize reciprocal moves as swaps and remaining moves as shuttles."""
    actions = sadd_solver_module._ordered_transport_actions_for_transition(
        2,
        {0: 0, 1: 1, 2: 3},
        {0: 1, 1: 0, 2: 2},
    )
    assert actions == [
        (2, PhysicalSwap(0, 1, 0, 1)),
        (2, Shuttle(2, 3, 2)),
    ]


def test_solver_applies_terminal_parity_to_downstream_phase() -> None:
    """Keep the integer CP-SAT phase objective aligned with full replay."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
    )
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


def test_solver_is_invariant_under_absolute_time_translation() -> None:
    """Translate solver windows and decoded pulses with the schedule clock."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    base = _idle_result(architecture, [0], 5)
    offset = 3
    shifted_state = replace(
        base.initial_state,
        time=offset,
        ions_busy_until=((0, offset),),
        pzs_busy_until=(("pz", offset),),
    )
    shifted = Schedule(base.scheduled_actions, base.end_time + offset, shifted_state)
    base_problem = build_sadd_problem(
        base,
        architecture,
        target_pz="pz",
        t_start=1,
        t_end=3,
        participating_ions=(0,),
    )
    shifted_problem = build_sadd_problem(
        shifted,
        architecture,
        target_pz="pz",
        t_start=1 + offset,
        t_end=3 + offset,
        participating_ions=(0,),
    )

    base_solution = solve_sadd_problem(base_problem, timeout_s=1.0, allow_transport=False)
    shifted_solution = solve_sadd_problem(shifted_problem, timeout_s=1.0, allow_transport=False)

    assert shifted_solution.status == base_solution.status
    assert shifted_solution.objective_before == pytest.approx(base_solution.objective_before)
    assert shifted_solution.objective_after == pytest.approx(base_solution.objective_after)
    assert shifted_solution.pulse_timesteps == {
        ion: tuple(timestep + offset for timestep in timesteps)
        for ion, timesteps in base_solution.pulse_timesteps.items()
    }
    assert shifted_solution.schedule is not None
    assert shifted_solution.schedule.start_time == offset
    assert shifted_solution.validation_status == "valid"


def test_phase_bounds_cover_the_full_critical_segment() -> None:
    """Include fixed phase on both sides of the optimized opportunity."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    result = _idle_result(architecture, [0], 5)
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=1,
        t_end=3,
        participating_ions=(0,),
    )

    (term_data,) = sadd_solver_module._phase_term_data(problem)

    assert term_data.phase_before_overlap == 1_000
    assert term_data.phase_after_opportunity == 2_000
    assert term_data.phase_bound == 5_000


def test_solver_rejects_objective_outside_cp_sat_integer_domain() -> None:
    """Reject an oversized scaled objective before constructing phase variables."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    result = _idle_result(architecture, [0], 5)
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=1,
        t_end=3,
        participating_ions=(0,),
        scale=10**10,
    )

    with pytest.raises(ValueError, match="int64"):
        solve_sadd_problem(problem, timeout_s=1.0, allow_transport=False)


def test_equal_primary_optima_remain_semantically_equivalent() -> None:
    """Accept either primary optimum when symmetric pulse positions tie."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1]})
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


@pytest.mark.parametrize(
    ("t_end", "participating_ions"),
    [(3, (0,)), (2, (0, 1))],
)
def test_retained_transport_admits_the_unchanged_schedule(
    t_end: int,
    participating_ions: tuple[int, ...],
) -> None:
    """Keep a schedule whose transport the window cannot rewrite representable."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [2]})
    schedule = schedule_from_path(
        [
            PhysicalSwap(ion_a=0, ion_b=1, pos_a=0, pos_b=1),
            AdvanceTime(),
            AdvanceTime(),
            AdvanceTime(),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 1]),
        architecture,
    )
    problem = build_sadd_problem(
        schedule,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=t_end,
        participating_ions=participating_ions,
    )

    solution = solve_sadd_problem(problem, timeout_s=1.0, allow_transport=True, allow_pulses=False)

    assert solution.status == "OPTIMAL"
    assert solution.validation_status == "valid"
    assert solution.schedule is not None
    assert solution.schedule.path == schedule.path


def test_full_segment_phase_range_admits_the_unchanged_schedule() -> None:
    """Bound phase by its own segment so a long tail cannot clip the domain."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    schedule = _idle_result(architecture, [0], 100)
    problem = build_sadd_problem(
        schedule,
        architecture,
        target_pz="pz",
        t_start=47,
        t_end=49,
        participating_ions=(0,),
    )

    solution = solve_sadd_problem(problem, timeout_s=1.0, allow_transport=False, allow_pulses=False)

    assert solution.status == "OPTIMAL"
    assert solution.validation_status == "valid"
    assert solution.objective_after == pytest.approx(solution.objective_before)


def test_position_model_rejects_traversing_a_retained_shuttle_edge() -> None:
    """Forbid the reciprocal exchange that transport-layer replay rejects."""
    architecture = LinearArchitecture(num_sites=4, processing_zones={"pz": [0]})
    schedule = schedule_from_path(
        [Shuttle(ion=1, src=1, dst=2), Shuttle(ion=0, src=2, dst=3), AdvanceTime()],
        create_initial_state(2, architecture, initial_positions=[2, 1]),
        architecture,
    )
    problem = build_sadd_problem(
        schedule,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=1,
        participating_ions=(0,),
    )
    problem = replace(problem, fixed_positions={})
    timeline = build_timeline(schedule, architecture)
    model = cp_model.CpModel()
    x = {}
    occupancy = []
    for site in range(architecture.num_sites):
        variable = model.NewBoolVar(f"x_i0_s{site}_t0")
        x[0, 0, site] = variable
        occupancy.append(variable)
    model.AddExactlyOne(occupancy)
    sadd_solver_module._add_position_and_movement_constraints(model, problem, timeline, x, allow_transport=True)
    model.Add(x[0, 0, 1] == 1)

    assert cp_model.CpSolver().Solve(model) == cp_model.INFEASIBLE


def test_synthesized_transport_is_not_mistaken_for_a_local_pulse() -> None:
    """Score a solution by its inserted pulses, never by every new action id."""
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [0]},
        field_profile=FieldProfile(num_sites=3, site_field=((0, 3.0), (1, 1.0), (2, 0.5))),
    )
    schedule = _idle_result(architecture, [1], 4)
    problem = build_sadd_problem(
        schedule,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=4,
        participating_ions=(0,),
    )

    solution = solve_sadd_problem(problem, timeout_s=5.0, allow_transport=True)

    assert solution.materialized
    assert solution.objective_after is not None
    assert any(isinstance(action, (Shuttle, PhysicalSwap)) for _timestep, action in solution.transport_actions)
    inserted_pulses = frozenset(
        action_id for action_ids in solution.pulse_action_ids.values() for action_id in action_ids
    )
    assert all(
        isinstance(item.action, Rx)
        for item in cast("Schedule", solution.schedule).scheduled_actions
        if item.action_id in inserted_pulses
    )


def test_unchanged_schedule_hint_leaves_decisions_unchanged() -> None:
    """Seed the search without altering the feasible set or the objective."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    schedule = _idle_result(architecture, [0], 5)
    problem = build_sadd_problem(
        schedule,
        architecture,
        target_pz="pz",
        t_start=1,
        t_end=3,
        participating_ions=(0,),
    )

    solution = solve_sadd_problem(problem, timeout_s=1.0, allow_transport=False)

    assert solution.objective_before == pytest.approx(25.0)
    assert solution.objective_after == pytest.approx(1.0)
    assert solution.pulse_timesteps == {0: (2,)}


def test_nonunit_operation_intervals_prevent_overlapping_starts() -> None:
    """Reserve shuttle, swap, and control resources for their full durations."""
    shuttle_model, shuttle_move, shuttles, _swaps, _controls = _duration_model(
        _duration_architecture(gate_timing=GateTiming(rx=2), transport_timing=TransportTiming(shuttle=3, swap=3)),
        [0, 2],
    )
    shuttle_model.Add(shuttles[0, 0] == 1)
    shuttle_model.Add(shuttle_move[0, 1] == 1)
    assert cp_model.CpSolver().Solve(shuttle_model) == cp_model.INFEASIBLE

    swap_model, swap_move, _shuttles, swaps, _controls = _duration_model(
        _duration_architecture(transport_timing=TransportTiming(swap=3)),
        [0, 1],
    )
    swap_model.Add(swaps[0, 1, 0] == 1)
    swap_model.Add(swap_move[0, 1] == 1)
    assert cp_model.CpSolver().Solve(swap_model) == cp_model.INFEASIBLE

    control_model, _moves, _shuttles, _swaps, controls = _duration_model(
        _duration_architecture(gate_timing=GateTiming(rx=2)),
        [1],
    )
    control_model.Add(controls[0, 0] == 1)
    control_model.Add(controls[0, 1] == 1)
    assert cp_model.CpSolver().Solve(control_model) == cp_model.INFEASIBLE


def _duration_architecture(
    *,
    gate_timing: GateTiming | None = None,
    transport_timing: TransportTiming | None = None,
) -> LinearArchitecture:
    return LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [0, 1, 2]},
        gate_timing=gate_timing or GateTiming(),
        transport_timing=transport_timing or TransportTiming(),
    )


def _duration_model(
    architecture: LinearArchitecture,
    initial_positions: list[int],
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


def test_problem_takes_operation_durations_from_the_architecture() -> None:
    """Use the architecture's transport and Rx durations for synthesized operations."""
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [1]},
        transport_timing=TransportTiming(shuttle=2, swap=3),
    )
    result = schedule_from_path(
        [
            Shuttle(0, 0, 1),
            AdvanceTime(),
            AdvanceTime(),
            PhysicalSwap(0, 1, 1, 2),
            AdvanceTime(),
            AdvanceTime(),
            AdvanceTime(),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
        architecture,
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
    assert problem.pulse_duration == 1


def test_materialized_pulse_uses_the_architecture_rx_duration() -> None:
    """Carry the architecture's single-qubit duration into the augmented schedule."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        gate_timing=GateTiming(rx=2),
    )
    result = _idle_result(architecture, [0], 3)
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=3,
        participating_ions=(0,),
    )
    solution = solve_sadd_problem(problem, timeout_s=1.0, allow_transport=False)
    assert solution.schedule is not None
    assert any(isinstance(item.action, Rx) and item.duration == 2 for item in solution.schedule.scheduled_actions)


def test_pulse_zone_follows_the_rewritten_transport() -> None:
    """Derive an inserted pulse's zone from the rewritten ion position."""
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [1]})
    result = _idle_result(architecture, [0], 3)
    problem = build_sadd_problem(
        result,
        architecture,
        target_pz="pz",
        t_start=0,
        t_end=3,
        participating_ions=(0,),
    )

    rewritten, pulse_action_ids = sadd_solver_module._rewrite_control_window(
        problem,
        ((0, Shuttle(0, 0, 1)),),
        {0: (1,)},
    )

    (pulse_id,) = pulse_action_ids[0]
    pulse = next(item for item in rewritten.scheduled_actions if item.action_id == pulse_id)
    assert isinstance(pulse.action, Rx)
    assert pulse.processing_zone_id == "pz"
