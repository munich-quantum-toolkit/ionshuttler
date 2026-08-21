# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Periodic global dynamical decoupling for compiled Linear schedules."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import count
from math import isfinite, pi
from typing import TYPE_CHECKING, ClassVar, cast

from mqt.ionshuttler.linear.actions import AdvanceTime, GateSpec, GlobalPulse
from mqt.ionshuttler.linear.dd.critical_segments import CriticalSegmentResult, compute_critical_segments
from mqt.ionshuttler.linear.dd.frame_replay import frame_operation_for_gate_spec, global_pulse_timesteps
from mqt.ionshuttler.linear.dd.result import DDPassResult
from mqt.ionshuttler.linear.dd.schedule_transform import rebuild_schedule, validate_schedule_compatibility
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.schedule import ActionSchedule, ScheduledAction

from ..._json_utils import require_int, require_int_list, require_number, require_str

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalDDConfig:
    """Configure a periodic sequence of schedule-wide global X pulses.

    Shift optimization minimizes the logic-aware critical-segment phase
    cost shared with SADD.
    """

    spacing: int
    shift_range: int = 0
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
    """Summarize the selected global pulse sequence and phase cost."""

    report_type: ClassVar[str] = "periodic_global_dd"

    scheme_name: str
    pulse_timesteps: tuple[int, ...]
    spacing: int
    phase_cost: float

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
        if isinstance(self.phase_cost, bool) or not isinstance(self.phase_cost, int | float):
            msg = "phase_cost must be numeric"
            raise TypeError(msg)
        if self.phase_cost < 0.0 or not isfinite(self.phase_cost):
            msg = "phase_cost must be finite and non-negative"
            raise ValueError(msg)
        object.__setattr__(self, "pulse_timesteps", pulse_timesteps)

    def to_dict(self) -> dict[str, object]:
        """Return this report using JSON-compatible values."""
        return {
            "scheme_name": self.scheme_name,
            "pulse_timesteps": list(self.pulse_timesteps),
            "spacing": self.spacing,
            "phase_cost": self.phase_cost,
        }

    @classmethod
    def from_dict(cls, data: object) -> GlobalDDReport:
        """Restore a global-DD report from JSON-compatible values.

        Returns:
            The restored global-DD report.

        Raises:
            ValueError: If the serialized report is malformed.
        """
        if not isinstance(data, dict):
            msg = "global DD report must be a JSON object"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
        mapping = cast("dict[str, object]", data)
        return cls(
            scheme_name=require_str(mapping, "scheme_name"),
            pulse_timesteps=tuple(require_int_list(mapping, "pulse_timesteps")),
            spacing=require_int(mapping, "spacing"),
            phase_cost=require_number(mapping, "phase_cost"),
        )


def apply_periodic_global_dd(
    schedule: ActionSchedule,
    architecture: Architecture,
    config: GlobalDDConfig,
) -> DDPassResult[GlobalDDReport]:
    """Insert periodic global X pulses, optionally shifting them by phase cost.

    Global pulses are placed before all existing actions at the same schedule
    boundary and may overlap local gates or transport under the global-frame
    abstraction.

    Returns:
        The transformed schedule and selected pulse/phase summary.

    Raises:
        ValueError: If the architecture does not support global pulses, replay
            metadata is absent, or global pulses already exist.
    """
    validate_schedule_compatibility(schedule, architecture)
    if not architecture.supports(GlobalPulse):
        msg = "periodic global DD requires an architecture with global-pulse support"
        raise ValueError(msg)
    timeline = build_timeline(schedule, architecture)
    existing_pulse_times = global_pulse_timesteps(timeline)
    if existing_pulse_times:
        msg = (
            "periodic global DD expects an original path without global pulses; "
            f"existing global pulse timesteps: {list(existing_pulse_times)}"
        )
        raise ValueError(msg)

    candidate_times = _periodic_pulse_timesteps(
        schedule.num_timesteps,
        config.spacing,
        half_first_window=config.half_first_window,
    )
    if not candidate_times:
        report = _global_dd_report(config, (), compute_critical_segments(schedule, architecture))
        return DDPassResult(schedule=schedule, architecture=architecture, report=report)

    pulse_timesteps, updated, summary = _optimize_global_pulse_timesteps(
        schedule,
        architecture,
        config,
        candidate_times,
    )
    report = _global_dd_report(config, pulse_timesteps, summary)
    logger.info(
        "inserted periodic global DD: scheme=%s spacing=%s shift_range=%s pulse_timesteps=%s",
        config.scheme_name,
        config.spacing,
        config.shift_range,
        pulse_timesteps,
    )
    return DDPassResult(schedule=updated, architecture=architecture, report=report)


def _optimize_global_pulse_timesteps(
    program: ActionSchedule,
    architecture: Architecture,
    config: GlobalDDConfig,
    base_pulse_timesteps: tuple[int, ...],
) -> tuple[tuple[int, ...], ActionSchedule, CriticalSegmentResult]:
    best_timesteps = base_pulse_timesteps
    best_result, best_summary = _materialize_candidate(
        program,
        architecture,
        base_pulse_timesteps,
        config.pulse,
    )
    best_score = best_summary.phase_cost
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
            program.num_timesteps,
            config.shift_range,
        ):
            candidate_result, candidate_summary = _materialize_candidate(
                program,
                architecture,
                candidate_timesteps,
                config.pulse,
            )
            candidate_score = candidate_summary.phase_cost
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
    program: ActionSchedule,
    architecture: Architecture,
    pulse_timesteps: tuple[int, ...],
    pulse_spec: GateSpec,
) -> tuple[ActionSchedule, CriticalSegmentResult]:
    scheduled_actions = _path_with_inserted_global_pulses(program, pulse_timesteps, pulse_spec)
    updated = rebuild_schedule(program, scheduled_actions)
    return updated, compute_critical_segments(updated, architecture)


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
    program: ActionSchedule,
    pulse_timesteps: tuple[int, ...],
    pulse_spec: GateSpec,
) -> tuple[ScheduledAction, ...]:
    pulses_by_time = {timestep: [GlobalPulse(gate=pulse_spec)] for timestep in pulse_timesteps}
    updated: list[ScheduledAction] = []
    action_ids = count(program.next_action_id)
    current_time = 0
    inserted_at_current_time = False
    for item in program.scheduled_actions:
        if not inserted_at_current_time and current_time in pulses_by_time:
            updated.extend(ScheduledAction(next(action_ids), action) for action in pulses_by_time[current_time])
            inserted_at_current_time = True
        updated.append(item)
        if isinstance(item.action, AdvanceTime):
            current_time += item.action.timestep_increment
            inserted_at_current_time = False
    if not inserted_at_current_time and current_time in pulses_by_time:
        updated.extend(ScheduledAction(next(action_ids), action) for action in pulses_by_time[current_time])
    return tuple(updated)


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
    summary: CriticalSegmentResult,
) -> GlobalDDReport:
    return GlobalDDReport(
        scheme_name=config.scheme_name,
        pulse_timesteps=pulse_timesteps,
        spacing=config.spacing,
        phase_cost=summary.phase_cost,
    )


def _require_int(value: object, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if value < minimum:
        msg = f"{name} must be >= {minimum}"
        raise ValueError(msg)


__all__ = ["GlobalDDConfig", "GlobalDDReport", "apply_periodic_global_dd"]
