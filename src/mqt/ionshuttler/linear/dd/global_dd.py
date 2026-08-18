# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Periodic global dynamical decoupling for compiled Linear schedules."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from math import isfinite, pi
from typing import TYPE_CHECKING, Literal

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, GlobalPulse
from mqt.ionshuttler.linear.dd.frame_replay import frame_operation_for_gate_spec, global_pulse_timesteps
from mqt.ionshuttler.linear.dd.metrics import ResidualPhaseSummary, summarize_residual_phases
from mqt.ionshuttler.linear.dd.result import DDPassResult
from mqt.ionshuttler.linear.dd.schedule_transform import rebuild_result
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline
from mqt.ionshuttler.linear.result import GlobalDDRecord

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.result import CompilationResult

ShiftObjective = Literal["sum_abs", "sum_squared"]
_SHIFT_OBJECTIVES = frozenset({"sum_abs", "sum_squared"})
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalDDConfig:
    """Configure a periodic sequence of schedule-wide global X pulses."""

    spacing: int
    shift_range: int = 0
    shift_objective: ShiftObjective = "sum_abs"
    half_first_window: bool = True
    pulse: GateSpec = field(default_factory=lambda: GateSpec("Rx", theta=pi))
    scheme_name: str = "periodic_x"

    def __post_init__(self) -> None:
        """Validate periodic placement and optional shift optimization.

        Raises:
            TypeError: If a configuration value has the wrong type.
            ValueError: If a value or pulse lies outside the supported domain.
        """
        _require_int(self.spacing, "spacing", minimum=1)
        _require_int(self.shift_range, "shift_range", minimum=0)
        if not isinstance(self.shift_objective, str):
            msg = "shift_objective must be a string"
            raise TypeError(msg)
        if self.shift_objective not in _SHIFT_OBJECTIVES:
            available = ", ".join(sorted(_SHIFT_OBJECTIVES))
            msg = f"shift_objective must be one of {{{available}}}"
            raise ValueError(msg)
        if not isinstance(self.half_first_window, bool):
            msg = "half_first_window must be a Boolean"
            raise TypeError(msg)
        if not isinstance(self.pulse, GateSpec):
            msg = "pulse must be a GateSpec"
            raise TypeError(msg)
        operation = frame_operation_for_gate_spec(self.pulse)
        if operation.label != "X":
            msg = "periodic global refocusing currently supports only global X pulses"
            raise ValueError(msg)
        if not isinstance(self.scheme_name, str):
            msg = "scheme_name must be a string"
            raise TypeError(msg)
        if not self.scheme_name:
            msg = "scheme_name must be non-empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class GlobalDDReport:
    """Summarize the selected global pulse sequence and residual phases."""

    scheme_name: str
    pulse_timesteps: tuple[int, ...]
    spacing: int
    sum_absolute_residual_phase: float
    sum_squared_residual_phase: float
    max_absolute_residual_phase: float

    def __post_init__(self) -> None:
        """Freeze pulse positions and validate their ordering.

        Raises:
            TypeError: If a report value has the wrong type.
            ValueError: If a report value is outside its supported range.
        """
        if not isinstance(self.scheme_name, str):
            msg = "scheme_name must be a string"
            raise TypeError(msg)
        if not self.scheme_name:
            msg = "scheme_name must be non-empty"
            raise ValueError(msg)
        _require_int(self.spacing, "spacing", minimum=1)
        pulse_timesteps = tuple(self.pulse_timesteps)
        for timestep in pulse_timesteps:
            _require_int(timestep, "each pulse timestep", minimum=0)
        if tuple(sorted(set(pulse_timesteps))) != pulse_timesteps:
            msg = "pulse_timesteps must be strictly increasing"
            raise ValueError(msg)
        for name in (
            "sum_absolute_residual_phase",
            "sum_squared_residual_phase",
            "max_absolute_residual_phase",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = f"{name} must be numeric"
                raise TypeError(msg)
            if value < 0.0 or not isfinite(value):
                msg = f"{name} must be finite and non-negative"
                raise ValueError(msg)
        object.__setattr__(self, "pulse_timesteps", pulse_timesteps)


def apply_periodic_global_dd(
    result: CompilationResult,
    config: GlobalDDConfig,
    architecture: Architecture | None = None,
) -> DDPassResult[CompilationResult, GlobalDDReport]:
    """Insert periodic global X pulses, optionally shifting them by phase cost.

    Global pulses are placed before all existing actions at the same schedule
    boundary and may overlap local gates or transport under the global-frame
    abstraction.

    Returns:
        The transformed schedule and selected pulse/phase summary.

    Raises:
        ValueError: If replay metadata is absent or global pulses already exist.
    """
    resolved_architecture = architecture or result.architecture
    if resolved_architecture is None:
        msg = "architecture is required for global DD"
        raise ValueError(msg)
    if result.initial_state is None:
        msg = "result.initial_state is required for global DD"
        raise ValueError(msg)

    timeline = build_timeline(result, resolved_architecture)
    existing_pulse_times = global_pulse_timesteps(timeline)
    if existing_pulse_times:
        msg = (
            "periodic global DD expects an original path without global pulses; "
            f"existing global pulse timesteps: {list(existing_pulse_times)}"
        )
        raise ValueError(msg)

    candidate_times = _periodic_pulse_timesteps(
        result.num_timesteps,
        config.spacing,
        half_first_window=config.half_first_window,
    )
    if not candidate_times:
        report = _global_dd_report(config, (), summarize_residual_phases(result, timeline=timeline))
        return DDPassResult(program=result, report=report)

    pulse_timesteps, updated, summary = _optimize_global_pulse_timesteps(
        result,
        resolved_architecture,
        config,
        candidate_times,
    )
    report = _global_dd_report(config, pulse_timesteps, summary)
    record = GlobalDDRecord(
        scheme_name=config.scheme_name,
        pulse_timesteps=pulse_timesteps,
        spacing=config.spacing,
        sum_abs_residual_phase=summary.sum_absolute_residual_phase,
        sum_squared_residual_phase=summary.sum_squared_residual_phase,
        max_abs_residual_phase=summary.max_absolute_residual_phase,
    )
    logger.info(
        "inserted periodic global DD: scheme=%s spacing=%s shift_range=%s shift_objective=%s pulse_timesteps=%s",
        config.scheme_name,
        config.spacing,
        config.shift_range,
        config.shift_objective,
        pulse_timesteps,
    )
    return DDPassResult(
        program=replace(updated, global_dd_records=(*result.global_dd_records, record)),
        report=report,
    )


def _optimize_global_pulse_timesteps(
    result: CompilationResult,
    architecture: Architecture,
    config: GlobalDDConfig,
    base_pulse_timesteps: tuple[int, ...],
) -> tuple[tuple[int, ...], CompilationResult, ResidualPhaseSummary]:
    best_timesteps = base_pulse_timesteps
    best_result, _timeline, best_summary = _materialize_candidate(
        result,
        architecture,
        base_pulse_timesteps,
        config.pulse,
    )
    best_score = _residual_phase_objective(best_summary, config.shift_objective)
    if config.shift_range == 0:
        return best_timesteps, best_result, best_summary

    for pulse_index in range(len(best_timesteps)):
        pulse_best_timesteps = best_timesteps
        pulse_best_result = best_result
        pulse_best_summary = best_summary
        pulse_best_score = best_score
        for candidate_timesteps in _shifted_pulse_timestep_candidates(
            best_timesteps,
            pulse_index,
            result.num_timesteps,
            config.shift_range,
        ):
            candidate_result, _timeline, candidate_summary = _materialize_candidate(
                result,
                architecture,
                candidate_timesteps,
                config.pulse,
            )
            candidate_score = _residual_phase_objective(candidate_summary, config.shift_objective)
            if candidate_score < pulse_best_score:
                pulse_best_timesteps = candidate_timesteps
                pulse_best_result = candidate_result
                pulse_best_summary = candidate_summary
                pulse_best_score = candidate_score
        best_timesteps = pulse_best_timesteps
        best_result = pulse_best_result
        best_summary = pulse_best_summary
        best_score = pulse_best_score
    return best_timesteps, best_result, best_summary


def _materialize_candidate(
    result: CompilationResult,
    architecture: Architecture,
    pulse_timesteps: tuple[int, ...],
    pulse_spec: GateSpec,
) -> tuple[CompilationResult, CompiledTimeline, ResidualPhaseSummary]:
    path = _path_with_inserted_global_pulses(result.path, pulse_timesteps, pulse_spec)
    updated = rebuild_result(result, path, architecture)
    timeline = build_timeline(updated, architecture)
    return updated, timeline, summarize_residual_phases(updated, timeline=timeline)


def _residual_phase_objective(summary: ResidualPhaseSummary, objective: ShiftObjective) -> float:
    if objective == "sum_abs":
        return summary.sum_absolute_residual_phase
    return summary.sum_squared_residual_phase


def _shifted_pulse_timestep_candidates(
    pulse_timesteps: tuple[int, ...],
    pulse_index: int,
    num_timesteps: int,
    shift_range: int,
) -> tuple[tuple[int, ...], ...]:
    current_timestep = pulse_timesteps[pulse_index]
    minimum = 0 if pulse_index == 0 else pulse_timesteps[pulse_index - 1] + 1
    maximum = num_timesteps - 1 if pulse_index == len(pulse_timesteps) - 1 else pulse_timesteps[pulse_index + 1] - 1
    candidates: list[tuple[int, ...]] = []
    for delta in range(-shift_range, shift_range + 1):
        candidate_timestep = current_timestep + delta
        if delta == 0 or not minimum <= candidate_timestep <= maximum:
            continue
        candidate = list(pulse_timesteps)
        candidate[pulse_index] = candidate_timestep
        candidates.append(tuple(candidate))
    return tuple(candidates)


def _path_with_inserted_global_pulses(
    path: list[Action],
    pulse_timesteps: tuple[int, ...],
    pulse_spec: GateSpec,
) -> list[Action]:
    pulses_by_time = {timestep: [GlobalPulse(gate=pulse_spec)] for timestep in pulse_timesteps}
    updated: list[Action] = []
    current_time = 0
    inserted_at_current_time = False
    for action in path:
        if not inserted_at_current_time and current_time in pulses_by_time:
            updated.extend(pulses_by_time[current_time])
            inserted_at_current_time = True
        updated.append(action)
        if isinstance(action, AdvanceTime):
            current_time += action.timestep_increment
            inserted_at_current_time = False
    if not inserted_at_current_time and current_time in pulses_by_time:
        updated.extend(pulses_by_time[current_time])
    return updated


def _periodic_pulse_timesteps(
    num_timesteps: int,
    spacing: int,
    *,
    half_first_window: bool,
) -> tuple[int, ...]:
    first_timestep = spacing // 2 if half_first_window else spacing - 1
    return tuple(range(first_timestep, num_timesteps, spacing))


def _global_dd_report(
    config: GlobalDDConfig,
    pulse_timesteps: tuple[int, ...],
    summary: ResidualPhaseSummary,
) -> GlobalDDReport:
    return GlobalDDReport(
        scheme_name=config.scheme_name,
        pulse_timesteps=pulse_timesteps,
        spacing=config.spacing,
        sum_absolute_residual_phase=summary.sum_absolute_residual_phase,
        sum_squared_residual_phase=summary.sum_squared_residual_phase,
        max_absolute_residual_phase=summary.max_absolute_residual_phase,
    )


def _require_int(value: object, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if value < minimum:
        msg = f"{name} must be >= {minimum}"
        raise ValueError(msg)


__all__ = ["GlobalDDConfig", "GlobalDDReport", "ShiftObjective", "apply_periodic_global_dd"]
