# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optional OR-Tools integration for shuttling-aware dynamical decoupling (SADD)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from itertools import count
from math import pi
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

from mqt.ionshuttler.linear.actions import (
    Action,
    AdvanceTime,
    PhysicalSwap,
    Rx,
    Shuttle,
    SingleQubitGate,
    TransportAction,
    TwoQubitGate,
)
from mqt.ionshuttler.linear.dd.critical_segments import CriticalSegment, compute_critical_segments
from mqt.ionshuttler.linear.dd.schedule_transform import rebuild_schedule, validate_rebuilt_schedule
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline
from mqt.ionshuttler.linear.schedule import ActionSchedule, ScheduledAction

if TYPE_CHECKING:
    from types import ModuleType

    from mqt.ionshuttler.linear.architecture import Architecture

_MISSING_OR_TOOLS_MESSAGE = "OR-Tools is required for SADD optimization; install IonShuttler with the 'dd' extra"
# Solver expression types cannot be imported while the optional extra is absent.
SolverObject: TypeAlias = Any


class _HasDuration(Protocol):
    duration: int


class _OperationDurations(Protocol):
    @property
    def shuttle(self) -> int: ...

    @property
    def swap(self) -> int: ...

    @property
    def one_qubit_gate(self) -> int: ...


@dataclass(frozen=True)
class _InferredDurations:
    shuttle: int
    swap: int
    one_qubit_gate: int = 1


@dataclass(frozen=True)
class SADDProblem:
    """Contain one bounded control-window CP-SAT problem."""

    schedule: ActionSchedule
    target_pz: str
    t_start: int
    t_end: int
    participating_ions: tuple[int, ...]
    start_positions: dict[int, int]
    fixed_positions: dict[tuple[int, int], int]
    fixed_transport_timesteps: frozenset[tuple[int, int]]
    phase_segments: tuple[CriticalSegment, ...]
    sensitivity_profile: tuple[float, ...]
    objective_before: float
    scale: int = 1000
    shuttle_duration: int = 1
    swap_duration: int = 1
    pulse_duration: int = 1
    num_search_workers: int = 8
    local_pulse_action_ids: frozenset[int] = frozenset()

    @property
    def duration(self) -> int:
        """Number of discrete intervals in the opportunity."""
        return self.t_end - self.t_start

    @property
    def architecture(self) -> Architecture:
        """Hardware model owned by the input schedule."""
        return self.schedule.architecture


@dataclass(frozen=True)
class SADDSolution:
    """Contain a decoded CP-SAT solution and materialization diagnostics."""

    status: str
    objective_before: float
    objective_after: float | None
    trajectories: dict[int, tuple[int, ...]]
    pulse_timesteps: dict[int, tuple[int, ...]]
    pulse_action_ids: dict[int, tuple[int, ...]]
    transport_actions: tuple[tuple[int, Action], ...]
    schedule: ActionSchedule | None
    validation_status: str
    validation_error: str | None
    runtime_s: float
    solver_objective: int | None = None
    model_num_variables: int = 0
    model_num_constraints: int = 0

    @property
    def materialized(self) -> bool:
        """Whether the solution produced a replay-valid schedule."""
        return self.schedule is not None and self.validation_status == "valid"


def build_sadd_problem(
    schedule: ActionSchedule,
    *,
    target_pz: str,
    t_start: int,
    t_end: int,
    participating_ions: tuple[int, ...],
    scale: int = 1000,
    operation_durations: _OperationDurations | None = None,
    num_search_workers: int = 8,
    local_pulse_action_ids: frozenset[int] = frozenset(),
) -> SADDProblem:
    """Build one shuttling-aware dynamical decoupling optimization problem.

    Returns:
        The bounded optimization problem.

    Raises:
        ValueError: If schedule metadata or problem bounds are invalid.
    """
    architecture = schedule.architecture
    if target_pz not in (architecture.processing_zones or {}):
        msg = f"unknown processing zone: {target_pz!r}"
        raise ValueError(msg)
    if not 0 <= t_start < t_end <= schedule.num_timesteps:
        msg = f"expected 0 <= t_start < t_end <= {schedule.num_timesteps}"
        raise ValueError(msg)
    if not participating_ions:
        msg = "participating_ions must not be empty"
        raise ValueError(msg)
    if scale < 1:
        msg = "scale must be >= 1"
        raise ValueError(msg)
    if num_search_workers < 1:
        msg = "num_search_workers must be >= 1"
        raise ValueError(msg)

    durations = operation_durations or _infer_operation_durations(schedule)
    timeline = build_timeline(schedule)
    positions_before = _positions_before_timesteps(schedule)
    fixed_positions, fixed_transport_timesteps = _fixed_positions_for_interval(
        timeline,
        positions_before,
        participating_ions,
        t_start,
        t_end,
    )
    trace = compute_critical_segments(schedule, local_pulse_action_ids=local_pulse_action_ids)
    phase_segments = tuple(
        segment
        for segment in trace.segments
        if segment.ion in participating_ions and segment.start < t_end and t_start < segment.end
    )
    return SADDProblem(
        schedule=schedule,
        target_pz=target_pz,
        t_start=t_start,
        t_end=t_end,
        participating_ions=tuple(sorted(participating_ions)),
        start_positions={ion: positions_before[t_start][ion] for ion in participating_ions},
        fixed_positions=fixed_positions,
        fixed_transport_timesteps=fixed_transport_timesteps,
        phase_segments=phase_segments,
        sensitivity_profile=trace.sensitivity_profile,
        objective_before=sum(segment.squared_phase for segment in phase_segments),
        scale=scale,
        shuttle_duration=durations.shuttle,
        swap_duration=durations.swap,
        pulse_duration=durations.one_qubit_gate,
        num_search_workers=num_search_workers,
        local_pulse_action_ids=local_pulse_action_ids,
    )


def solve_sadd_problem(
    problem: SADDProblem,
    *,
    timeout_s: float = 10.0,
    allow_transport: bool = True,
    allow_pulses: bool = True,
) -> SADDSolution:
    """Solve and materialize one SADD control-window problem.

    Returns:
        The decoded solver outcome and validation diagnostics.
    """
    cp_model = _load_cp_model()
    started = time.perf_counter()
    timeline = build_timeline(problem.schedule)
    model = cp_model.CpModel()
    rel_times = range(problem.duration)

    x = {}
    for ion in problem.participating_ions:
        for rel_t in rel_times:
            occupancy = []
            for site in range(problem.architecture.num_sites):
                variable = model.NewBoolVar(f"x_i{ion}_s{site}_t{rel_t}")
                x[ion, rel_t, site] = variable
                occupancy.append(variable)
            model.AddExactlyOne(occupancy)

    move = _add_position_and_movement_constraints(model, problem, timeline, x, allow_transport=allow_transport)
    shuttle, swap = _add_transport_kind_constraints(model, problem, x, move)
    control = _add_control_constraints(model, problem, timeline, x, move, allow_pulses=allow_pulses)
    _add_operation_duration_constraints(model, problem, timeline, shuttle, swap, control)
    parity = _add_parity_constraints(model, problem, control)
    square_terms = _add_phase_objective_terms(model, problem, x, parity)
    objective = sum(square_terms) * 1_000_000 + sum(control.values()) * 1_000 + sum(move.values())
    model.Minimize(objective)
    model_proto = model.Proto()
    model_num_variables = len(model_proto.variables)
    model_num_constraints = len(model_proto.constraints)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_search_workers = problem.num_search_workers
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)
    if status not in {"OPTIMAL", "FEASIBLE"}:
        return SADDSolution(
            status=status,
            objective_before=problem.objective_before,
            objective_after=None,
            trajectories={},
            pulse_timesteps={},
            pulse_action_ids={},
            transport_actions=(),
            schedule=None,
            validation_status="not_solved",
            validation_error=None,
            runtime_s=time.perf_counter() - started,
            model_num_variables=model_num_variables,
            model_num_constraints=model_num_constraints,
        )

    primary_objective_value = round(solver.ObjectiveValue())
    runtime_s = time.perf_counter() - started

    trajectories = {
        ion: tuple(_selected_site(solver, x, ion, rel_t, problem.architecture.num_sites) for rel_t in rel_times)
        for ion in problem.participating_ions
    }
    pulse_timesteps = {
        ion: tuple(problem.t_start + rel_t for rel_t in rel_times if solver.Value(control[ion, rel_t]))
        for ion in problem.participating_ions
    }
    transport_actions = _decode_transport_actions(problem, trajectories)
    materialized, pulse_action_ids, validation_status, validation_error = _materialize_solution(
        problem,
        transport_actions,
        pulse_timesteps,
    )
    return SADDSolution(
        status=status,
        objective_before=problem.objective_before,
        objective_after=_objective_for_result(materialized, problem) if materialized is not None else None,
        trajectories=trajectories,
        pulse_timesteps=pulse_timesteps,
        pulse_action_ids=pulse_action_ids,
        transport_actions=transport_actions,
        schedule=materialized,
        validation_status=validation_status,
        validation_error=validation_error,
        runtime_s=runtime_s,
        solver_objective=primary_objective_value,
        model_num_variables=model_num_variables,
        model_num_constraints=model_num_constraints,
    )


def _load_cp_model() -> ModuleType:
    """Load the CP-SAT module at the private solver boundary.

    Returns:
        The ``ortools.sat.python.cp_model`` module.

    Raises:
        ImportError: If the optional OR-Tools package is unavailable.
        ModuleNotFoundError: If an unrelated dependency of OR-Tools is unavailable.
    """
    try:
        return import_module("ortools.sat.python.cp_model")
    except ModuleNotFoundError as error:
        if error.name == "ortools" or (error.name is not None and error.name.startswith("ortools.")):
            raise ImportError(_MISSING_OR_TOOLS_MESSAGE) from error
        raise


def _infer_operation_durations(program: ActionSchedule) -> _InferredDurations:
    shuttle_durations = {action.duration for action in program.path if isinstance(action, Shuttle)}
    swap_durations = {action.duration for action in program.path if isinstance(action, PhysicalSwap)}
    return _InferredDurations(
        shuttle=_single_duration_or_default(shuttle_durations),
        swap=_single_duration_or_default(swap_durations),
    )


def _single_duration_or_default(durations: set[int]) -> int:
    return next(iter(durations)) if len(durations) == 1 else 1


def _fixed_positions_for_interval(
    timeline: CompiledTimeline,
    positions_before: dict[int, dict[int, int]],
    participating_ions: tuple[int, ...],
    t_start: int,
    t_end: int,
) -> tuple[dict[tuple[int, int], int], frozenset[tuple[int, int]]]:
    fixed: dict[tuple[int, int], int] = {}
    fixed_transport: set[tuple[int, int]] = set()
    participant_set = set(participating_ions)
    for ion in participating_ions:
        fixed[ion, t_end - 1] = positions_before[t_end][ion]
    for timestep in range(t_end):
        for action in timeline.action_at(timestep) or ():
            for ion in _gate_ions(action):
                if ion in participant_set and t_start <= timestep < t_end:
                    fixed[ion, timestep] = timeline.ion_position(ion, timestep)
            action_ions = _action_ions(action)
            if not isinstance(action, TransportAction) or not action_ions.intersection(participant_set):
                continue
            duration = cast("_HasDuration", action).duration
            rewritable = action_ions.issubset(participant_set) and t_start <= timestep and timestep + duration <= t_end
            if rewritable:
                continue
            for ion in action_ions.intersection(participant_set):
                if t_start < timestep < t_end:
                    fixed[ion, timestep - 1] = positions_before[timestep][ion]
                if t_start <= timestep < t_end:
                    fixed[ion, timestep] = timeline.ion_position(ion, timestep)
                fixed_transport.update(
                    (ion, busy_t) for busy_t in range(max(timestep, t_start), min(timestep + duration, t_end))
                )
    return fixed, frozenset(fixed_transport)


def _positions_before_timesteps(program: ActionSchedule) -> dict[int, dict[int, int]]:
    positions = dict(program.initial_state.positions)
    positions_before = {0: dict(positions)}
    current_time = 0
    for action in program.path:
        if isinstance(action, AdvanceTime):
            current_time += action.timestep_increment
            positions_before.setdefault(current_time, dict(positions))
        elif isinstance(action, Shuttle):
            positions[action.ion] = action.dst
        elif isinstance(action, PhysicalSwap):
            positions[action.ion_a], positions[action.ion_b] = positions[action.ion_b], positions[action.ion_a]
    return positions_before


def _gate_ions(action: Action) -> tuple[int, ...]:
    if isinstance(action, SingleQubitGate):
        return (action.ion,)
    if isinstance(action, TwoQubitGate):
        return (action.ion_a, action.ion_b)
    return ()


def _action_ions(action: Action) -> set[int]:
    if isinstance(action, Shuttle):
        return {action.ion}
    if isinstance(action, PhysicalSwap):
        return {action.ion_a, action.ion_b}
    if isinstance(action, SingleQubitGate):
        return {action.ion}
    if isinstance(action, TwoQubitGate):
        return {action.ion_a, action.ion_b}
    return set()


def _add_position_and_movement_constraints(
    model: SolverObject,
    problem: SADDProblem,
    timeline: CompiledTimeline,
    x: dict[tuple[int, int, int], SolverObject],
    *,
    allow_transport: bool,
) -> dict[tuple[int, int], SolverObject]:
    participant_set = set(problem.participating_ions)
    for rel_t in range(problem.duration):
        t_abs = problem.t_start + rel_t
        fixed_obstacles = {site for ion, site in timeline.state_at(t_abs).positions if ion not in participant_set}
        for site in range(problem.architecture.num_sites):
            capacity = 0 if site in fixed_obstacles else 1
            model.Add(sum(x[ion, rel_t, site] for ion in problem.participating_ions) <= capacity)
        for ion in problem.participating_ions:
            fixed = problem.fixed_positions.get((ion, t_abs))
            if fixed is not None:
                model.Add(x[ion, rel_t, fixed] == 1)
            if not allow_transport:
                model.Add(x[ion, rel_t, timeline.ion_position(ion, t_abs)] == 1)

    move = {}
    for ion in problem.participating_ions:
        for rel_t in range(problem.duration):
            destination = {site: x[ion, rel_t, site] for site in range(problem.architecture.num_sites)}
            if rel_t == 0:
                move[ion, rel_t] = _add_initial_transition_constraints(
                    model,
                    problem.architecture.num_sites,
                    problem.start_positions[ion],
                    destination,
                    f"move_i{ion}_t{rel_t}",
                )
            else:
                source = {site: x[ion, rel_t - 1, site] for site in range(problem.architecture.num_sites)}
                move[ion, rel_t] = _add_transition_constraints(
                    model,
                    problem.architecture.num_sites,
                    source,
                    destination,
                    f"move_i{ion}_t{rel_t}",
                )
    return move


def _add_initial_transition_constraints(
    model: SolverObject,
    num_sites: int,
    source_site: int,
    destination: dict[int, SolverObject],
    name: str,
) -> SolverObject:
    for dst in range(num_sites):
        if abs(source_site - dst) > 1:
            model.Add(destination[dst] == 0)
    moved = model.NewBoolVar(name)
    model.Add(destination[source_site] == 0).OnlyEnforceIf(moved)
    model.Add(destination[source_site] == 1).OnlyEnforceIf(moved.Not())
    return moved


def _add_transition_constraints(
    model: SolverObject,
    num_sites: int,
    source: dict[int, SolverObject],
    destination: dict[int, SolverObject],
    name: str,
) -> SolverObject:
    for src in range(num_sites):
        for dst in range(num_sites):
            if abs(src - dst) > 1:
                model.Add(source[src] + destination[dst] <= 1)
    same_terms = []
    for site in range(num_sites):
        same = model.NewBoolVar(f"{name}_same_s{site}")
        model.Add(same <= source[site])
        model.Add(same <= destination[site])
        model.Add(same >= source[site] + destination[site] - 1)
        same_terms.append(same)
    moved = model.NewBoolVar(name)
    model.Add(sum(same_terms) == 0).OnlyEnforceIf(moved)
    model.Add(sum(same_terms) == 1).OnlyEnforceIf(moved.Not())
    return moved


def _add_transport_kind_constraints(
    model: SolverObject,
    problem: SADDProblem,
    x: dict[tuple[int, int, int], SolverObject],
    move: dict[tuple[int, int], SolverObject],
) -> tuple[dict[tuple[int, int], SolverObject], dict[tuple[int, int, int], SolverObject]]:
    ions = problem.participating_ions
    num_sites = problem.architecture.num_sites
    shuttle = {}
    swap = {}
    for rel_t in range(problem.duration):
        for index, ion_a in enumerate(ions):
            for ion_b in ions[index + 1 :]:
                variable = model.NewBoolVar(f"swap_i{ion_a}_i{ion_b}_t{rel_t}")
                swap[ion_a, ion_b, rel_t] = variable
                patterns = []
                if rel_t == 0:
                    src_a = problem.start_positions[ion_a]
                    src_b = problem.start_positions[ion_b]
                    if abs(src_a - src_b) == 1:
                        patterns.append(
                            _add_bool_and(
                                model,
                                (x[ion_a, rel_t, src_b], x[ion_b, rel_t, src_a]),
                                f"swap_pattern_i{ion_a}_i{ion_b}_t{rel_t}",
                            )
                        )
                else:
                    for left in range(num_sites - 1):
                        right = left + 1
                        for src_a, src_b in ((left, right), (right, left)):
                            patterns.append(
                                _add_bool_and(
                                    model,
                                    (
                                        x[ion_a, rel_t - 1, src_a],
                                        x[ion_b, rel_t - 1, src_b],
                                        x[ion_a, rel_t, src_b],
                                        x[ion_b, rel_t, src_a],
                                    ),
                                    f"swap_pattern_i{ion_a}_i{ion_b}_s{src_a}_t{rel_t}",
                                )
                            )
                model.Add(variable == sum(patterns))

        for ion in ions:
            ion_swaps = [
                variable
                for (ion_a, ion_b, timestep), variable in swap.items()
                if timestep == rel_t and ion in {ion_a, ion_b}
            ]
            variable = model.NewBoolVar(f"shuttle_i{ion}_t{rel_t}")
            shuttle[ion, rel_t] = variable
            model.Add(variable + sum(ion_swaps) == move[ion, rel_t])
    return shuttle, swap


def _add_bool_and(
    model: SolverObject,
    literals: tuple[SolverObject, ...],
    name: str,
) -> SolverObject:
    result = model.NewBoolVar(name)
    for literal in literals:
        model.Add(result <= literal)
    model.Add(result >= sum(literals) - len(literals) + 1)
    return result


def _add_control_constraints(
    model: SolverObject,
    problem: SADDProblem,
    timeline: CompiledTimeline,
    x: dict[tuple[int, int, int], SolverObject],
    move: dict[tuple[int, int], SolverObject],
    *,
    allow_pulses: bool,
) -> dict[tuple[int, int], SolverObject]:
    processing_zones = problem.architecture.processing_zones or {}
    zone_sites = processing_zones[problem.target_pz]
    control = {}
    for ion in problem.participating_ions:
        for rel_t in range(problem.duration):
            t_abs = problem.t_start + rel_t
            variable = model.NewBoolVar(f"control_x_i{ion}_t{rel_t}")
            control[ion, rel_t] = variable
            if (
                not allow_pulses
                or timeline.ion_gate_busy(ion, t_abs)
                or (ion, t_abs) in problem.fixed_transport_timesteps
            ):
                model.Add(variable == 0)
                continue
            model.Add(sum(x[ion, rel_t, site] for site in zone_sites) >= 1).OnlyEnforceIf(variable)
            model.Add(move[ion, rel_t] == 0).OnlyEnforceIf(variable)
    for rel_t in range(problem.duration):
        if timeline.pz_busy(problem.target_pz, problem.t_start + rel_t):
            for ion in problem.participating_ions:
                model.Add(control[ion, rel_t] == 0)
        model.Add(sum(control[ion, rel_t] for ion in problem.participating_ions) <= 1)
    return control


def _add_operation_duration_constraints(
    model: SolverObject,
    problem: SADDProblem,
    timeline: CompiledTimeline,
    shuttle: dict[tuple[int, int], SolverObject],
    swap: dict[tuple[int, int, int], SolverObject],
    control: dict[tuple[int, int], SolverObject],
) -> None:
    horizon = problem.duration
    for (_ion, start), variable in shuttle.items():
        if start + problem.shuttle_duration > horizon:
            model.Add(variable == 0)
    for (_ion_a, _ion_b, start), variable in swap.items():
        if start + problem.swap_duration > horizon:
            model.Add(variable == 0)
    for (_ion, start), variable in control.items():
        if start + problem.pulse_duration > horizon:
            model.Add(variable == 0)

    for ion in problem.participating_ions:
        for rel_t in range(horizon):
            active = [
                variable
                for (candidate_ion, start), variable in shuttle.items()
                if candidate_ion == ion and start <= rel_t < start + problem.shuttle_duration
            ]
            active.extend(
                variable
                for (ion_a, ion_b, start), variable in swap.items()
                if ion in {ion_a, ion_b} and start <= rel_t < start + problem.swap_duration
            )
            active.extend(
                variable
                for (candidate_ion, start), variable in control.items()
                if candidate_ion == ion and start <= rel_t < start + problem.pulse_duration
            )
            model.Add(sum(active) <= 1)
            t_abs = problem.t_start + rel_t
            if timeline.ion_gate_busy(ion, t_abs) or (ion, t_abs) in problem.fixed_transport_timesteps:
                model.Add(sum(active) == 0)

    for rel_t in range(horizon):
        active_controls = [
            variable for (_ion, start), variable in control.items() if start <= rel_t < start + problem.pulse_duration
        ]
        model.Add(sum(active_controls) <= 1)
        if timeline.pz_busy(problem.target_pz, problem.t_start + rel_t):
            model.Add(sum(active_controls) == 0)


def _add_parity_constraints(
    model: SolverObject,
    problem: SADDProblem,
    control: dict[tuple[int, int], SolverObject],
) -> dict[tuple[int, int], SolverObject]:
    parity = {}
    allowed = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
    for ion in problem.participating_ions:
        for rel_t in range(problem.duration):
            parity[ion, rel_t] = model.NewBoolVar(f"parity_i{ion}_t{rel_t}")
            if rel_t == 0:
                model.Add(parity[ion, rel_t] == control[ion, rel_t])
            else:
                model.AddAllowedAssignments(
                    [parity[ion, rel_t - 1], control[ion, rel_t], parity[ion, rel_t]],
                    allowed,
                )
    return parity


def _add_phase_objective_terms(
    model: SolverObject,
    problem: SADDProblem,
    x: dict[tuple[int, int, int], SolverObject],
    parity: dict[tuple[int, int], SolverObject],
) -> list[SolverObject]:
    terms = []
    phase_bound = problem.scale * problem.duration * max(1, problem.architecture.num_sites) * 4
    for term_index, segment in enumerate(problem.phase_segments):
        overlap_start = max(segment.start, problem.t_start)
        overlap_end = min(segment.end, problem.t_end)
        phase_before_overlap = 0
        phase_after_opportunity = 0
        expression_terms = []
        for t_abs in range(segment.start, overlap_start):
            offset = t_abs - segment.start
            phase_before_overlap += round(
                problem.scale * segment.toggling_signs[offset] * segment.sensitivities[offset]
            )
        for t_abs in range(overlap_start, overlap_end):
            offset = t_abs - segment.start
            rel_t = t_abs - problem.t_start
            base_sign = segment.toggling_signs[offset]
            for site in range(problem.architecture.num_sites):
                weight = round(problem.scale * base_sign * problem.sensitivity_profile[site])
                expression_terms.append(weight * x[segment.ion, rel_t, site])
                toggled = model.NewBoolVar(f"toggle_x_i{segment.ion}_s{site}_t{rel_t}_seg{term_index}")
                model.Add(toggled <= x[segment.ion, rel_t, site])
                model.Add(toggled <= parity[segment.ion, rel_t])
                model.Add(toggled >= x[segment.ion, rel_t, site] + parity[segment.ion, rel_t] - 1)
                expression_terms.append(-2 * weight * toggled)
        for t_abs in range(max(overlap_end, problem.t_end), segment.end):
            offset = t_abs - segment.start
            phase_after_opportunity += round(
                problem.scale * segment.toggling_signs[offset] * segment.sensitivities[offset]
            )
        if phase_after_opportunity:
            terminal_parity = parity[segment.ion, problem.duration - 1]
            expression_terms.extend((phase_after_opportunity, -2 * phase_after_opportunity * terminal_parity))
        phase = model.NewIntVar(-phase_bound, phase_bound, f"phase_seg{term_index}")
        model.Add(phase == phase_before_overlap + sum(expression_terms))
        square = model.NewIntVar(0, phase_bound * phase_bound, f"phase_square_seg{term_index}")
        model.AddMultiplicationEquality(square, [phase, phase])
        terms.append(square)
    return terms


def _selected_site(
    solver: SolverObject,
    x: dict[tuple[int, int, int], SolverObject],
    ion: int,
    rel_t: int,
    num_sites: int,
) -> int:
    for site in range(num_sites):
        if solver.Value(x[ion, rel_t, site]):
            return site
    msg = "solver returned no selected site for an occupied ion"
    raise ValueError(msg)


def _decode_transport_actions(
    problem: SADDProblem,
    trajectories: dict[int, tuple[int, ...]],
) -> tuple[tuple[int, Action], ...]:
    actions: list[tuple[int, Action]] = []
    previous = dict(problem.start_positions)
    for rel_t in range(problem.duration):
        t_action = problem.t_start + rel_t
        target = {ion: trajectory[rel_t] for ion, trajectory in trajectories.items()}
        fixed_transport_ions = {
            ion for ion in problem.participating_ions if (ion, t_action) in problem.fixed_transport_timesteps
        }
        actions.extend(
            _ordered_transport_actions_for_transition(
                t_action,
                {ion: site for ion, site in previous.items() if ion not in fixed_transport_ions},
                {ion: site for ion, site in target.items() if ion not in fixed_transport_ions},
                shuttle_duration=problem.shuttle_duration,
                swap_duration=problem.swap_duration,
            )
        )
        previous = dict(target)
    return tuple(actions)


def _ordered_transport_actions_for_transition(
    t_action: int,
    previous: dict[int, int],
    target: dict[int, int],
    *,
    shuttle_duration: int = 1,
    swap_duration: int = 1,
) -> list[tuple[int, Action]]:
    current = dict(previous)
    moves = {ion: (previous[ion], target[ion]) for ion in previous if previous[ion] != target[ion]}
    actions: list[tuple[int, Action]] = []
    while moves:
        swap = _find_swap(moves)
        if swap is not None:
            ion, other = swap
            src, dst = moves.pop(ion)
            moves.pop(other)
            actions.append((
                t_action,
                PhysicalSwap(ion_a=ion, ion_b=other, pos_a=src, pos_b=dst, duration=swap_duration),
            ))
            current[ion], current[other] = current[other], current[ion]
            continue
        occupied = set(current.values())
        shuttle_ion = next((ion for ion, (_src, dst) in sorted(moves.items()) if dst not in occupied), None)
        if shuttle_ion is None:
            msg = (
                f"could not order synchronous transport layer at t={t_action}: previous={previous!r} target={target!r}"
            )
            raise ValueError(msg)
        src, dst = moves.pop(shuttle_ion)
        actions.append((t_action, Shuttle(shuttle_ion, src, dst, duration=shuttle_duration)))
        current[shuttle_ion] = dst
    return actions


def _find_swap(moves: dict[int, tuple[int, int]]) -> tuple[int, int] | None:
    for ion, (src, dst) in sorted(moves.items()):
        for other, (other_src, other_dst) in sorted(moves.items()):
            if ion < other and src == other_dst and dst == other_src:
                return ion, other
    return None


def _materialize_solution(
    problem: SADDProblem,
    transport_actions: tuple[tuple[int, Action], ...],
    pulse_timesteps: dict[int, tuple[int, ...]],
) -> tuple[ActionSchedule | None, dict[int, tuple[int, ...]], str, str | None]:
    try:
        updated, pulse_action_ids = _materialize_and_validate(problem, transport_actions, pulse_timesteps)
    except Exception as error:  # ruff: ignore[blind-except] - Validation details are part of the solver result.
        return None, {}, "invalid", str(error)
    return updated, pulse_action_ids, "valid", None


def _materialize_and_validate(
    problem: SADDProblem,
    transport_actions: tuple[tuple[int, Action], ...],
    pulse_timesteps: dict[int, tuple[int, ...]],
) -> tuple[ActionSchedule, dict[int, tuple[int, ...]]]:
    updated, pulse_action_ids = _rewrite_control_window(problem, transport_actions, pulse_timesteps)
    if not validate_rebuilt_schedule(updated):
        msg = "decoded solution fails full schedule replay validation"
        raise ValueError(msg)
    updated_timeline = build_timeline(updated)
    original_timeline = build_timeline(problem.schedule)
    if (
        updated_timeline.state_at(problem.schedule.num_timesteps).positions
        != original_timeline.state_at(problem.schedule.num_timesteps).positions
    ):
        msg = "decoded solution changes final ion positions"
        raise ValueError(msg)
    return updated, pulse_action_ids


def _rewrite_control_window(
    problem: SADDProblem,
    transport_actions: tuple[tuple[int, Action], ...],
    pulse_timesteps: dict[int, tuple[int, ...]],
) -> tuple[ActionSchedule, dict[int, tuple[int, ...]]]:
    timeline = build_timeline(problem.schedule)
    participant_set = set(problem.participating_ions)
    decoded_transport_by_time: dict[int, list[Action]] = {}
    for timestep, action in transport_actions:
        decoded_transport_by_time.setdefault(timestep, []).append(action)
    pulses_by_time: dict[int, list[Rx]] = {}
    for ion, timesteps in pulse_timesteps.items():
        for timestep in timesteps:
            pulses_by_time.setdefault(timestep, []).append(Rx(ion=ion, theta=pi, duration=problem.pulse_duration))

    action_ids = count(problem.schedule.next_action_id)
    pulse_action_ids: dict[int, list[int]] = {}
    new_actions: list[ScheduledAction] = []
    for timestep in range(problem.schedule.num_timesteps + 1):
        if problem.t_start <= timestep < problem.t_end:
            new_actions.extend(
                ScheduledAction(next(action_ids), action) for action in decoded_transport_by_time.get(timestep, ())
            )
            for action in pulses_by_time.get(timestep, ()):
                action_id = next(action_ids)
                new_actions.append(ScheduledAction(action_id, action))
                pulse_action_ids.setdefault(action.ion, []).append(action_id)
        original_at_time = timeline.scheduled_action_at(timestep) or ()
        for item in original_at_time:
            action = item.action
            if isinstance(action, AdvanceTime):
                continue
            if (
                problem.t_start <= timestep < problem.t_end
                and isinstance(action, TransportAction)
                and _action_ions(action).intersection(participant_set)
            ):
                duration = cast("_HasDuration", action).duration
                if _action_ions(action).issubset(participant_set) and timestep + duration <= problem.t_end:
                    continue
            new_actions.append(item)
        new_actions.extend(item for item in original_at_time if isinstance(item.action, AdvanceTime))
    return rebuild_schedule(problem.schedule, new_actions), {
        ion: tuple(pulse_action_ids.get(ion, ())) for ion in pulse_timesteps
    }


def _objective_for_result(program: ActionSchedule, problem: SADDProblem) -> float:
    inserted_action_ids = frozenset(item.action_id for item in program.scheduled_actions).difference(
        item.action_id for item in problem.schedule.scheduled_actions
    )
    trace = compute_critical_segments(
        program,
        local_pulse_action_ids=problem.local_pulse_action_ids.union(inserted_action_ids),
    )
    segment_keys = {(segment.ion, segment.index, segment.start, segment.end) for segment in problem.phase_segments}
    return sum(
        segment.squared_phase
        for segment in trace.segments
        if (segment.ion, segment.index, segment.start, segment.end) in segment_keys
    )


__all__ = ["SADDProblem", "SADDSolution", "build_sadd_problem", "solve_sadd_problem"]
