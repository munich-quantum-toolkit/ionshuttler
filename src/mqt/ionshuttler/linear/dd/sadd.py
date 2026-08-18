# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public types for the shuttling-aware dynamical decoupling (SADD) pass."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from mqt.ionshuttler.linear.dd.critical_segments import CriticalSegment, compute_critical_segments
from mqt.ionshuttler.linear.dd.result import DDPassResult
from mqt.ionshuttler.linear.dd.sadd_solver import build_sadd_problem, solve_sadd_problem
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.result import CompilationResult

IonFloatMapping = Mapping[int, float]
IonTimestepsMapping = Mapping[int, tuple[int, ...]]


class SADDMethod(str, Enum):
    """Select a shuttling-aware dynamical decoupling (SADD) method."""

    PULSE_ONLY = "pulse_only_sadd"
    FULL = "full_sadd"

    @property
    def allow_transport(self) -> bool:
        """Whether this method permits transport synthesis."""
        return self is SADDMethod.FULL


@dataclass(frozen=True)
class OperationDurations:
    """Define durations of operations that SADD may synthesize."""

    shuttle: int = 1
    swap: int = 3
    one_qubit_gate: int = 1

    def __post_init__(self) -> None:
        """Validate that every synthesized operation occupies positive time.

        Raises:
            TypeError: If a duration is not an integer.
            ValueError: If a duration is not positive.
        """
        for name in ("shuttle", "swap", "one_qubit_gate"):
            duration = getattr(self, name)
            if isinstance(duration, bool) or not isinstance(duration, int):
                msg = f"{name} must be an integer"
                raise TypeError(msg)
            if duration < 1:
                msg = f"{name} must be >= 1"
                raise ValueError(msg)


@dataclass(frozen=True)
class SADDConfig:
    """Configure the shared pulse-only and transport-enabled SADD backend.

    SADD denotes shuttling-aware dynamical decoupling.
    """

    min_window_length: int = 2
    max_window_length: int = 16
    max_participating_ions: int = 5
    timeout_s: float = 10.0
    ion_preselection: Literal["phase", "distance"] = "phase"
    opportunity_order: Literal["chronological", "reverse_chronological"] = "chronological"
    max_accepted_windows: int | None = None
    improvement_tolerance: float = 1e-12
    allow_pulses: bool = True
    scale: int = 1000
    num_search_workers: int = 8
    operation_durations: OperationDurations = field(default_factory=OperationDurations)

    def __post_init__(self) -> None:
        """Validate optimization and problem parameters.

        Raises:
            TypeError: If a parameter has the wrong type.
            ValueError: If a parameter is outside its supported range.
        """
        _require_positive_int(self.min_window_length, "min_window_length")
        _require_positive_int(self.max_window_length, "max_window_length")
        if self.max_window_length < self.min_window_length:
            msg = "max_window_length must be >= min_window_length"
            raise ValueError(msg)
        _require_positive_int(self.max_participating_ions, "max_participating_ions")
        _require_positive_finite(self.timeout_s, "timeout_s")
        if self.ion_preselection not in {"phase", "distance"}:
            msg = "ion_preselection must be 'phase' or 'distance'"
            raise ValueError(msg)
        if self.opportunity_order not in {"chronological", "reverse_chronological"}:
            msg = "opportunity_order must be 'chronological' or 'reverse_chronological'"
            raise ValueError(msg)
        if self.max_accepted_windows is not None:
            _require_nonnegative_int(self.max_accepted_windows, "max_accepted_windows")
        _require_nonnegative_finite(self.improvement_tolerance, "improvement_tolerance")
        if not isinstance(self.allow_pulses, bool):
            msg = "allow_pulses must be a Boolean"
            raise TypeError(msg)
        _require_positive_int(self.scale, "scale")
        _require_positive_int(self.num_search_workers, "num_search_workers")
        if not isinstance(self.operation_durations, OperationDurations):
            msg = "operation_durations must be an OperationDurations instance"
            raise TypeError(msg)


@dataclass(frozen=True)
class SADDOpportunityRecord:
    """Describe one ordered SADD optimization opportunity and its outcome."""

    target_pz: str
    window: tuple[int, int]
    participating_ions: tuple[int, ...]
    status: str
    validation_status: str
    objective_before: float
    objective_after: float | None
    accepted: bool
    pulse_count: int
    transport_action_count: int
    runtime_s: float
    message: str | None = None
    eligible_ions: tuple[int, ...] = ()
    rejected_busy_ions: tuple[int, ...] = ()
    rejected_no_active_segment_ions: tuple[int, ...] = ()
    rejected_unreachable_ions: tuple[int, ...] = ()
    selection_scores: tuple[tuple[int, float, int], ...] = ()
    phase_before_by_ion: IonFloatMapping | None = None
    phase_after_by_ion: IonFloatMapping | None = None
    pulse_timesteps: IonTimestepsMapping | None = None
    transport_actions: tuple[str, ...] = ()
    trajectories: IonTimestepsMapping | None = None
    model_num_variables: int = 0
    model_num_constraints: int = 0

    def __post_init__(self) -> None:
        """Validate and freeze nested opportunity diagnostics.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field contains an invalid value.
        """
        if not self.target_pz:
            msg = "target_pz must be non-empty"
            raise ValueError(msg)
        _validate_window(self.window)
        for name in (
            "participating_ions",
            "eligible_ions",
            "rejected_busy_ions",
            "rejected_no_active_segment_ions",
            "rejected_unreachable_ions",
        ):
            object.__setattr__(self, name, _freeze_ion_sequence(getattr(self, name), name))
        if not self.status:
            msg = "status must be non-empty"
            raise ValueError(msg)
        if not self.validation_status:
            msg = "validation_status must be non-empty"
            raise ValueError(msg)
        _require_nonnegative_finite(self.objective_before, "objective_before")
        if self.objective_after is not None:
            _require_nonnegative_finite(self.objective_after, "objective_after")
        if not isinstance(self.accepted, bool):
            msg = "accepted must be a Boolean"
            raise TypeError(msg)
        if self.accepted and self.objective_after is None:
            msg = "accepted opportunities require objective_after"
            raise ValueError(msg)
        _require_nonnegative_int(self.pulse_count, "pulse_count")
        _require_nonnegative_int(self.transport_action_count, "transport_action_count")
        _require_nonnegative_finite(self.runtime_s, "runtime_s")
        object.__setattr__(self, "selection_scores", _freeze_selection_scores(self.selection_scores))
        object.__setattr__(
            self,
            "phase_before_by_ion",
            _freeze_float_mapping(self.phase_before_by_ion, "phase_before_by_ion"),
        )
        object.__setattr__(
            self,
            "phase_after_by_ion",
            _freeze_float_mapping(self.phase_after_by_ion, "phase_after_by_ion"),
        )
        object.__setattr__(
            self,
            "pulse_timesteps",
            _freeze_timesteps_mapping(self.pulse_timesteps, "pulse_timesteps"),
        )
        object.__setattr__(self, "transport_actions", tuple(self.transport_actions))
        object.__setattr__(
            self,
            "trajectories",
            _freeze_timesteps_mapping(self.trajectories, "trajectories"),
        )
        _require_nonnegative_int(self.model_num_variables, "model_num_variables")
        _require_nonnegative_int(self.model_num_constraints, "model_num_constraints")
        if self.pulse_timesteps is not None and self.pulse_count != sum(map(len, self.pulse_timesteps.values())):
            msg = "pulse_count must match pulse_timesteps"
            raise ValueError(msg)
        if self.transport_action_count != len(self.transport_actions):
            msg = "transport_action_count must match transport_actions"
            raise ValueError(msg)


@dataclass(frozen=True)
class SADDReport:
    """Collect ordered opportunity records produced by one SADD pass."""

    method: SADDMethod
    opportunities: tuple[SADDOpportunityRecord, ...] = ()

    def __post_init__(self) -> None:
        """Normalize opportunity records to an immutable tuple.

        Raises:
            TypeError: If the method or an opportunity has the wrong type.
        """
        if not isinstance(self.method, SADDMethod):
            msg = "method must be a SADDMethod"
            raise TypeError(msg)
        opportunities = tuple(self.opportunities)
        if any(not isinstance(opportunity, SADDOpportunityRecord) for opportunity in opportunities):
            msg = "opportunities must contain SADDOpportunityRecord values"
            raise TypeError(msg)
        object.__setattr__(self, "opportunities", opportunities)


@dataclass(frozen=True)
class _ParticipantSelection:
    selected_ions: tuple[int, ...]
    eligible_ions: tuple[int, ...]
    busy_ions: tuple[int, ...]
    rejected_no_active_segment_ions: tuple[int, ...]
    rejected_unreachable_ions: tuple[int, ...]
    selection_scores: tuple[tuple[int, float, int], ...]


def run_sadd(
    result: CompilationResult,
    method: SADDMethod,
    architecture: Architecture | None = None,
    config: SADDConfig | None = None,
) -> DDPassResult[CompilationResult, SADDReport]:
    """Apply shuttling-aware dynamical decoupling to a compiled schedule.

    Pulse-only and transport-enabled SADD use the same optimization backend;
    ``method`` controls whether the solver may synthesize transport.

    Returns:
        The transformed schedule and an ordered SADD report.

    Raises:
        TypeError: If ``method`` is not a :class:`SADDMethod`.
        ValueError: If required schedule metadata is absent.
    """
    if not isinstance(method, SADDMethod):
        msg = "method must be a SADDMethod"
        raise TypeError(msg)
    resolved_config = config or SADDConfig()
    resolved_architecture = architecture or result.architecture
    if resolved_architecture is None:
        msg = "architecture is required for SADD"
        raise ValueError(msg)
    if result.initial_state is None:
        msg = "result.initial_state is required for SADD"
        raise ValueError(msg)

    updated = result
    records: list[SADDOpportunityRecord] = []
    accepted_count = 0
    for target_pz, window in _iter_control_windows(updated, resolved_architecture, resolved_config):
        if resolved_config.max_accepted_windows is not None and accepted_count >= resolved_config.max_accepted_windows:
            break
        selection = _select_participating_ions(
            updated,
            resolved_architecture,
            target_pz,
            window,
            resolved_config,
        )
        if not selection.selected_ions:
            continue
        try:
            problem = build_sadd_problem(
                updated,
                resolved_architecture,
                target_pz=target_pz,
                t_start=window[0],
                t_end=window[1],
                participating_ions=selection.selected_ions,
                scale=resolved_config.scale,
                operation_durations=resolved_config.operation_durations,
                num_search_workers=resolved_config.num_search_workers,
            )
            solution = solve_sadd_problem(
                problem,
                timeout_s=resolved_config.timeout_s,
                allow_transport=method.allow_transport,
                allow_pulses=resolved_config.allow_pulses,
            )
        except ImportError as error:
            return DDPassResult(
                program=updated,
                report=SADDReport(method=method, opportunities=tuple(records)),
                unavailable_reason=str(error),
            )
        accepted = (
            solution.result is not None
            and solution.objective_after is not None
            and solution.validation_status == "valid"
            and solution.objective_after < problem.objective_before - resolved_config.improvement_tolerance
        )
        if accepted and solution.result is not None:
            updated = solution.result
            accepted_count += 1
        records.append(
            SADDOpportunityRecord(
                target_pz=target_pz,
                window=window,
                participating_ions=selection.selected_ions,
                status=solution.status,
                validation_status=solution.validation_status,
                objective_before=solution.objective_before,
                objective_after=solution.objective_after,
                accepted=accepted,
                pulse_count=sum(len(timesteps) for timesteps in solution.pulse_timesteps.values()),
                transport_action_count=len(solution.transport_actions),
                runtime_s=solution.runtime_s,
                message=solution.validation_error,
                eligible_ions=selection.eligible_ions,
                rejected_busy_ions=selection.busy_ions,
                rejected_no_active_segment_ions=selection.rejected_no_active_segment_ions,
                rejected_unreachable_ions=selection.rejected_unreachable_ions,
                selection_scores=selection.selection_scores,
                phase_before_by_ion=_phase_by_ion(problem.phase_segments),
                phase_after_by_ion=(
                    _phase_by_ion_for_result(solution.result, problem.phase_segments)
                    if solution.result is not None
                    else None
                ),
                pulse_timesteps=solution.pulse_timesteps,
                transport_actions=tuple(f"t={timestep}: {action!r}" for timestep, action in solution.transport_actions),
                trajectories=solution.trajectories,
                model_num_variables=solution.model_num_variables,
                model_num_constraints=solution.model_num_constraints,
            )
        )
    return DDPassResult(program=updated, report=SADDReport(method=method, opportunities=tuple(records)))


def _iter_control_windows(
    result: CompilationResult,
    architecture: Architecture,
    config: SADDConfig,
) -> tuple[tuple[str, tuple[int, int]], ...]:
    timeline = build_timeline(result, architecture)
    windows: list[tuple[str, tuple[int, int]]] = []
    for pz_name in sorted(architecture.processing_zones or {}):
        start: int | None = None
        for timestep in range(timeline.makespan):
            if not timeline.pz_busy(pz_name, timestep):
                if start is None:
                    start = timestep
                continue
            if start is not None:
                windows.extend(_split_window(pz_name, start, timestep, config))
                start = None
        if start is not None:
            windows.extend(_split_window(pz_name, start, timeline.makespan, config))
    ordered = sorted(windows, key=lambda item: (item[1][0], item[1][1], item[0]))
    if config.opportunity_order == "reverse_chronological":
        ordered.reverse()
    return tuple(ordered)


def _split_window(
    pz_name: str,
    start: int,
    end: int,
    config: SADDConfig,
) -> list[tuple[str, tuple[int, int]]]:
    windows: list[tuple[str, tuple[int, int]]] = []
    cursor = start
    while cursor + config.min_window_length <= end:
        chunk_end = min(end, cursor + config.max_window_length)
        if chunk_end - cursor >= config.min_window_length:
            windows.append((pz_name, (cursor, chunk_end)))
        cursor = chunk_end
    return windows


def _select_participating_ions(
    result: CompilationResult,
    architecture: Architecture,
    target_pz: str,
    window: tuple[int, int],
    config: SADDConfig,
) -> _ParticipantSelection:
    timeline = build_timeline(result, architecture)
    trace = compute_critical_segments(result, architecture)
    processing_zones = architecture.processing_zones or {}
    zone_sites = processing_zones[target_pz]
    ion_scores: list[tuple[float, int, int]] = []
    busy_ions: list[int] = []
    rejected_no_segment: list[int] = []
    rejected_unreachable: list[int] = []
    for ion, _site in timeline.state_at(0).positions:
        if any(timeline.ion_busy(ion, timestep) for timestep in range(window[0], window[1])):
            busy_ions.append(ion)
        active_segments = [
            segment
            for segment in trace.segments
            if segment.ion == ion and segment.start < window[1] and window[0] < segment.end
        ]
        if not active_segments:
            rejected_no_segment.append(ion)
            continue
        reachable_distance = _min_reachable_control_distance(timeline, ion, zone_sites, window)
        if reachable_distance is None:
            rejected_unreachable.append(ion)
            continue
        ion_scores.append((-sum(segment.squared_phase for segment in active_segments), reachable_distance, ion))
    ranked = (
        sorted(ion_scores)
        if config.ion_preselection == "phase"
        else sorted(ion_scores, key=operator.itemgetter(1, 0, 2))
    )
    return _ParticipantSelection(
        selected_ions=tuple(ion for _score, _distance, ion in ranked[: config.max_participating_ions]),
        eligible_ions=tuple(ion for _score, _distance, ion in ranked),
        busy_ions=tuple(sorted(busy_ions)),
        rejected_no_active_segment_ions=tuple(sorted(rejected_no_segment)),
        rejected_unreachable_ions=tuple(sorted(rejected_unreachable)),
        selection_scores=tuple((ion, -score, distance) for score, distance, ion in ranked),
    )


def _min_reachable_control_distance(
    timeline: CompiledTimeline,
    ion: int,
    zone_sites: Sequence[int],
    window: tuple[int, int],
) -> int | None:
    obligations = _obligations_for_ion(timeline, ion, window)
    best_distance: int | None = None
    for timestep in range(window[0], window[1]):
        if timeline.ion_gate_busy(ion, timestep):
            continue
        previous = max((item for item in obligations if item[0] <= timestep), default=None)
        following = min((item for item in obligations if item[0] >= timestep), default=None)
        if previous is None or following is None:
            continue
        previous_time, previous_site = previous
        following_time, following_site = following
        for site in zone_sites:
            inbound = abs(previous_site - site)
            outbound = abs(site - following_site)
            if inbound <= timestep - previous_time and outbound <= following_time - timestep:
                distance = min(inbound, outbound)
                best_distance = distance if best_distance is None else min(best_distance, distance)
    return best_distance


def _obligations_for_ion(
    timeline: CompiledTimeline,
    ion: int,
    window: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    obligations = {
        (window[0], timeline.ion_position(ion, window[0])),
        (window[1] - 1, timeline.ion_position(ion, window[1] - 1)),
    }
    obligations.update(
        (timestep, timeline.ion_position(ion, timestep))
        for timestep in range(window[0], window[1])
        if timeline.ion_gate_busy(ion, timestep)
    )
    return tuple(sorted(obligations))


def _phase_by_ion(segments: tuple[CriticalSegment, ...]) -> dict[int, float]:
    phase_by_ion: dict[int, float] = {}
    for segment in segments:
        phase_by_ion[segment.ion] = phase_by_ion.get(segment.ion, 0.0) + segment.squared_phase
    return phase_by_ion


def _phase_by_ion_for_result(
    result: CompilationResult,
    source_segments: tuple[CriticalSegment, ...],
) -> dict[int, float]:
    if result.architecture is None:
        return {}
    trace = compute_critical_segments(result, result.architecture)
    segment_keys = {(segment.ion, segment.index, segment.start, segment.end) for segment in source_segments}
    return _phase_by_ion(
        tuple(
            segment
            for segment in trace.segments
            if (segment.ion, segment.index, segment.start, segment.end) in segment_keys
        )
    )


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if value < 1:
        msg = f"{name} must be >= 1"
        raise ValueError(msg)


def _require_nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if value < 0:
        msg = f"{name} must be >= 0"
        raise ValueError(msg)


def _require_positive_finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} must be numeric"
        raise TypeError(msg)
    if not math.isfinite(value) or value <= 0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)


def _require_nonnegative_finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} must be numeric"
        raise TypeError(msg)
    if not math.isfinite(value) or value < 0:
        msg = f"{name} must be finite and non-negative"
        raise ValueError(msg)


def _validate_window(window: object) -> None:
    if not isinstance(window, tuple) or len(window) != 2:
        msg = "window must be a pair of integers"
        raise TypeError(msg)
    start, end = window
    _require_nonnegative_int(start, "window start")
    _require_nonnegative_int(end, "window end")
    if end <= start:
        msg = "window end must be greater than its start"
        raise ValueError(msg)


def _freeze_ion_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    frozen = tuple(values)
    for ion in frozen:
        _require_nonnegative_int(ion, f"{name} ion")
    if len(set(frozen)) != len(frozen):
        msg = f"{name} must not contain duplicate ions"
        raise ValueError(msg)
    return frozen


def _freeze_selection_scores(
    values: Sequence[tuple[int, float, int]],
) -> tuple[tuple[int, float, int], ...]:
    frozen = tuple(tuple(score) for score in values)
    for ion, score, distance in frozen:
        _require_nonnegative_int(ion, "selection score ion")
        _require_nonnegative_finite(score, "selection score phase")
        _require_nonnegative_int(distance, "selection score distance")
    return frozen


def _freeze_float_mapping(values: IonFloatMapping | None, name: str) -> IonFloatMapping | None:
    if values is None:
        return None
    frozen: dict[int, float] = {}
    for ion, value in values.items():
        _require_nonnegative_int(ion, f"{name} ion")
        _require_nonnegative_finite(value, f"{name} value")
        frozen[ion] = float(value)
    return MappingProxyType(frozen)


def _freeze_timesteps_mapping(values: IonTimestepsMapping | None, name: str) -> IonTimestepsMapping | None:
    if values is None:
        return None
    frozen: dict[int, tuple[int, ...]] = {}
    for ion, timesteps in values.items():
        _require_nonnegative_int(ion, f"{name} ion")
        frozen_timesteps = tuple(timesteps)
        for timestep in frozen_timesteps:
            _require_nonnegative_int(timestep, f"{name} timestep")
        frozen[ion] = frozen_timesteps
    return MappingProxyType(frozen)


__all__ = [
    "OperationDurations",
    "SADDConfig",
    "SADDMethod",
    "SADDOpportunityRecord",
    "SADDReport",
    "run_sadd",
]
