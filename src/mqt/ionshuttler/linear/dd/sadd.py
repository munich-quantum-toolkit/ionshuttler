# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public types for the shuttling-aware dynamical decoupling (SADD) pass."""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from mqt.ionshuttler.linear.actions import PhysicalSwap, Rx, Shuttle, TransportAction
from mqt.ionshuttler.linear.dd.critical_segments import CriticalSegment, compute_critical_segments
from mqt.ionshuttler.linear.dd.result import DDPassResult, LocalDDSequence
from mqt.ionshuttler.linear.dd.sadd_solver import build_sadd_problem, solve_sadd_problem
from mqt.ionshuttler.linear.dd.schedule_transform import validate_schedule_compatibility
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.schedule import ActionSchedule

IonFloatMapping = Mapping[int, float]
IonTimestepsMapping = Mapping[int, tuple[int, ...]]


class SADDMethod(StrEnum):
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

    ``timeout_s`` bounds each control window's solve. The unchanged schedule is
    always an admissible solution and is supplied to the solver as a hint, so a
    window that runs out of time yields a weaker improvement or none at all
    rather than an invalid or missing result. Raise it for long windows or many
    participating ions, where proving optimality takes longer.

    When ``operation_durations`` is ``None``, SADD infers uniform transport
    durations already present in the schedule and uses the standard operation
    defaults for transport kinds that are absent.
    """

    min_window_length: int = 2
    max_window_length: int = 16
    max_participating_ions: int = 5
    timeout_s: float = 1.0
    ion_preselection: Literal["phase", "distance"] = "phase"
    opportunity_order: Literal["chronological", "reverse_chronological"] = "chronological"
    max_accepted_windows: int | None = None
    improvement_tolerance: float = 1e-12
    allow_pulses: bool = True
    scale: int = 1000
    num_search_workers: int = 8
    operation_durations: OperationDurations | None = None

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
        if self.operation_durations is not None and not isinstance(self.operation_durations, OperationDurations):
            msg = "operation_durations must be an OperationDurations instance or None"
            raise TypeError(msg)


@dataclass(frozen=True)
class SADDOpportunityRecord:
    """Describe one ordered SADD optimization opportunity and its outcome.

    ``transport_delta`` records the signed change in scheduled transport actions
    by concrete action type between the opportunity input and proposed result.
    """

    target_pz: str
    window: tuple[int, int]
    participating_ions: tuple[int, ...]
    status: str
    validation_status: str
    phase_cost_before: float
    phase_cost_after: float | None
    accepted: bool
    pulse_count: int
    transport_delta: Mapping[str, int]
    runtime_s: float
    message: str | None = None
    eligible_ions: tuple[int, ...] = ()
    busy_ions: tuple[int, ...] = ()
    rejected_no_active_segment_ions: tuple[int, ...] = ()
    rejected_unreachable_ions: tuple[int, ...] = ()
    selection_scores: tuple[tuple[int, float, int], ...] = ()
    phase_before_by_ion: IonFloatMapping | None = None
    phase_after_by_ion: IonFloatMapping | None = None
    pulse_timesteps: IonTimestepsMapping | None = None
    pulse_action_ids: IonTimestepsMapping | None = None
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
            "busy_ions",
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
        _require_nonnegative_finite(self.phase_cost_before, "phase_cost_before")
        if self.phase_cost_after is not None:
            _require_nonnegative_finite(self.phase_cost_after, "phase_cost_after")
        if not isinstance(self.accepted, bool):
            msg = "accepted must be a Boolean"
            raise TypeError(msg)
        if self.accepted and self.phase_cost_after is None:
            msg = "accepted opportunities require phase_cost_after"
            raise ValueError(msg)
        _require_nonnegative_int(self.pulse_count, "pulse_count")
        object.__setattr__(self, "transport_delta", _freeze_transport_delta(self.transport_delta))
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
        object.__setattr__(
            self,
            "pulse_action_ids",
            _freeze_timesteps_mapping(self.pulse_action_ids, "pulse_action_ids"),
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
        if self.pulse_action_ids is not None:
            if self.pulse_timesteps is None or self.pulse_action_ids.keys() != self.pulse_timesteps.keys():
                msg = "pulse_action_ids must contain the same ions as pulse_timesteps"
                raise ValueError(msg)
            if any(
                len(self.pulse_action_ids[ion]) != len(timesteps) for ion, timesteps in self.pulse_timesteps.items()
            ):
                msg = "pulse_action_ids must contain one identifier for every pulse timestep"
                raise ValueError(msg)


@dataclass(frozen=True)
class SADDReport:
    """Collect ordered opportunity records produced by one SADD pass."""

    report_type: ClassVar[str] = "sadd"

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

    @property
    def sequences(self) -> tuple[LocalDDSequence, ...]:
        """Accepted ion-local pulse sequences in opportunity order."""
        return tuple(
            LocalDDSequence(
                ion=ion,
                window=opportunity.window,
                scheme_name="cp_sat_control_centric",
                pulse_timesteps=timesteps,
                action_ids=opportunity.pulse_action_ids[ion],
            )
            for opportunity in self.opportunities
            if (
                opportunity.accepted
                and opportunity.pulse_timesteps is not None
                and opportunity.pulse_action_ids is not None
            )
            for ion, timesteps in opportunity.pulse_timesteps.items()
            if timesteps
        )

    def to_dict(self) -> dict[str, object]:
        """Return this report using JSON-compatible values."""
        return {
            "method": self.method.value,
            "opportunities": [_opportunity_to_dict(opportunity) for opportunity in self.opportunities],
        }

    @classmethod
    def from_dict(cls, data: object) -> SADDReport:
        """Restore a SADD report from JSON-compatible values.

        Returns:
            The restored SADD report.

        Raises:
            ValueError: If the serialized report is malformed.
        """
        if not isinstance(data, dict):
            msg = "SADD report must be a JSON object"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
        raw_opportunities = data.get("opportunities")
        if not isinstance(raw_opportunities, list):
            msg = "SADD report opportunities must be a list"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
        try:
            method = SADDMethod(data.get("method"))
        except ValueError as error:
            msg = f"unknown SADD method: {data.get('method')!r}"
            raise ValueError(msg) from error
        return cls(method, tuple(_opportunity_from_dict(value) for value in raw_opportunities))


@dataclass(frozen=True)
class _ParticipantSelection:
    selected_ions: tuple[int, ...]
    eligible_ions: tuple[int, ...]
    busy_ions: tuple[int, ...]
    rejected_no_active_segment_ions: tuple[int, ...]
    rejected_unreachable_ions: tuple[int, ...]
    selection_scores: tuple[tuple[int, float, int], ...]


def run_sadd(
    schedule: ActionSchedule,
    architecture: Architecture,
    method: SADDMethod,
    config: SADDConfig | None = None,
) -> DDPassResult[SADDReport]:
    """Apply shuttling-aware dynamical decoupling to a compiled schedule.

    Pulse-only and transport-enabled SADD use the same optimization backend;
    ``method`` controls whether the solver may synthesize transport.

    Returns:
        The transformed schedule and an ordered SADD report.

    Raises:
        TypeError: If ``method`` is not a :class:`SADDMethod`.
        ValueError: If the schedule and architecture are incompatible or the
            architecture lacks actions required by the selected method.
    """
    if not isinstance(method, SADDMethod):
        msg = "method must be a SADDMethod"
        raise TypeError(msg)
    validate_schedule_compatibility(schedule, architecture)
    required_actions = (Rx,) if method is SADDMethod.PULSE_ONLY else (Rx, Shuttle, PhysicalSwap)
    unsupported = [action_type.__name__ for action_type in required_actions if not architecture.supports(action_type)]
    if unsupported:
        msg = f"SADD requires actions unsupported by the architecture: {', '.join(unsupported)}"
        raise ValueError(msg)
    resolved_config = config or SADDConfig()
    updated = schedule
    records: list[SADDOpportunityRecord] = []
    local_pulse_action_ids: set[int] = set()
    accepted_count = 0
    for target_pz, window in _iter_control_windows(updated, architecture, resolved_config):
        opportunity_input = updated
        if resolved_config.max_accepted_windows is not None and accepted_count >= resolved_config.max_accepted_windows:
            break
        selection = _select_participating_ions(
            updated,
            architecture,
            target_pz,
            window,
            resolved_config,
            frozenset(local_pulse_action_ids),
        )
        if not selection.selected_ions:
            continue
        try:
            problem = build_sadd_problem(
                updated,
                architecture,
                target_pz=target_pz,
                t_start=window[0],
                t_end=window[1],
                participating_ions=selection.selected_ions,
                scale=resolved_config.scale,
                operation_durations=resolved_config.operation_durations,
                num_search_workers=resolved_config.num_search_workers,
                local_pulse_action_ids=frozenset(local_pulse_action_ids),
            )
            solution = solve_sadd_problem(
                problem,
                timeout_s=resolved_config.timeout_s,
                allow_transport=method.allow_transport,
                allow_pulses=resolved_config.allow_pulses,
            )
        except ImportError as error:
            return DDPassResult(
                schedule=updated,
                architecture=architecture,
                report=SADDReport(method=method, opportunities=tuple(records)),
                unavailable_reason=str(error),
            )
        accepted = (
            solution.schedule is not None
            and solution.objective_after is not None
            and solution.validation_status == "valid"
            and solution.objective_after < problem.objective_before - resolved_config.improvement_tolerance
        )
        solution_local_pulse_action_ids = frozenset(local_pulse_action_ids).union(
            action_id for action_ids in solution.pulse_action_ids.values() for action_id in action_ids
        )
        if accepted and solution.schedule is not None:
            updated = solution.schedule
            local_pulse_action_ids.update(solution_local_pulse_action_ids)
            accepted_count += 1
        records.append(
            SADDOpportunityRecord(
                target_pz=target_pz,
                window=window,
                participating_ions=selection.selected_ions,
                status=solution.status,
                validation_status=solution.validation_status,
                phase_cost_before=solution.objective_before,
                phase_cost_after=solution.objective_after,
                accepted=accepted,
                pulse_count=sum(len(timesteps) for timesteps in solution.pulse_timesteps.values()),
                transport_delta=_transport_delta(opportunity_input, solution.schedule),
                runtime_s=solution.runtime_s,
                message=solution.validation_error,
                eligible_ions=selection.eligible_ions,
                busy_ions=selection.busy_ions,
                rejected_no_active_segment_ions=selection.rejected_no_active_segment_ions,
                rejected_unreachable_ions=selection.rejected_unreachable_ions,
                selection_scores=selection.selection_scores,
                phase_before_by_ion=_phase_by_ion(problem.phase_segments),
                phase_after_by_ion=(
                    _phase_by_ion_for_program(
                        solution.schedule,
                        architecture,
                        problem.phase_segments,
                        solution_local_pulse_action_ids,
                    )
                    if solution.schedule is not None
                    else None
                ),
                pulse_timesteps=solution.pulse_timesteps,
                pulse_action_ids=solution.pulse_action_ids,
                transport_actions=tuple(f"t={timestep}: {action!r}" for timestep, action in solution.transport_actions),
                trajectories=solution.trajectories,
                model_num_variables=solution.model_num_variables,
                model_num_constraints=solution.model_num_constraints,
            )
        )
    return DDPassResult(
        schedule=updated,
        architecture=architecture,
        report=SADDReport(method=method, opportunities=tuple(records)),
    )


def _iter_control_windows(
    program: ActionSchedule,
    architecture: Architecture,
    config: SADDConfig,
) -> tuple[tuple[str, tuple[int, int]], ...]:
    timeline = build_timeline(program, architecture)
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
    program: ActionSchedule,
    architecture: Architecture,
    target_pz: str,
    window: tuple[int, int],
    config: SADDConfig,
    local_pulse_action_ids: frozenset[int],
) -> _ParticipantSelection:
    timeline = build_timeline(program, architecture)
    trace = compute_critical_segments(program, architecture, local_pulse_action_ids=local_pulse_action_ids)
    processing_zones = architecture.processing_zones or {}
    zone_sites = processing_zones[target_pz]
    ion_scores: list[tuple[float, int, int]] = []
    busy_ions: list[int] = []
    rejected_no_segment: list[int] = []
    rejected_unreachable: list[int] = []
    for ion, _site in timeline.state_at(0).positions:
        if any(timeline.ion_busy(ion, timestep) for timestep in range(window[0], window[1])):
            # Being busy for part of the window is recorded but does not disqualify the ion:
            # the solver's control and operation-duration constraints already forbid a pulse
            # at each busy timestep, leaving the window's remaining boundaries usable.
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


def _phase_by_ion_for_program(
    program: ActionSchedule,
    architecture: Architecture,
    source_segments: tuple[CriticalSegment, ...],
    local_pulse_action_ids: frozenset[int],
) -> dict[int, float]:
    trace = compute_critical_segments(program, architecture, local_pulse_action_ids=local_pulse_action_ids)
    segment_keys = {(segment.ion, segment.index, segment.start, segment.end) for segment in source_segments}
    return _phase_by_ion(
        tuple(
            segment
            for segment in trace.segments
            if (segment.ion, segment.index, segment.start, segment.end) in segment_keys
        )
    )


def _transport_delta(
    before: ActionSchedule,
    after: ActionSchedule | None,
) -> dict[str, int]:
    if after is None:
        return {}
    before_counts = Counter(type(action).__name__ for action in before.path if isinstance(action, TransportAction))
    after_counts = Counter(type(action).__name__ for action in after.path if isinstance(action, TransportAction))
    action_types = sorted(before_counts.keys() | after_counts.keys())
    return {
        action_type: delta
        for action_type in action_types
        if (delta := after_counts[action_type] - before_counts[action_type]) != 0
    }


def _opportunity_to_dict(opportunity: SADDOpportunityRecord) -> dict[str, object]:
    return {
        "target_pz": opportunity.target_pz,
        "window": list(opportunity.window),
        "participating_ions": list(opportunity.participating_ions),
        "status": opportunity.status,
        "validation_status": opportunity.validation_status,
        "phase_cost_before": opportunity.phase_cost_before,
        "phase_cost_after": opportunity.phase_cost_after,
        "accepted": opportunity.accepted,
        "pulse_count": opportunity.pulse_count,
        "transport_delta": dict(opportunity.transport_delta),
        "runtime_s": opportunity.runtime_s,
        "message": opportunity.message,
        "eligible_ions": list(opportunity.eligible_ions),
        "busy_ions": list(opportunity.busy_ions),
        "rejected_no_active_segment_ions": list(opportunity.rejected_no_active_segment_ions),
        "rejected_unreachable_ions": list(opportunity.rejected_unreachable_ions),
        "selection_scores": [list(score) for score in opportunity.selection_scores],
        "phase_before_by_ion": _float_mapping_to_list(opportunity.phase_before_by_ion),
        "phase_after_by_ion": _float_mapping_to_list(opportunity.phase_after_by_ion),
        "pulse_timesteps": _timesteps_mapping_to_list(opportunity.pulse_timesteps),
        "pulse_action_ids": _timesteps_mapping_to_list(opportunity.pulse_action_ids),
        "transport_actions": list(opportunity.transport_actions),
        "trajectories": _timesteps_mapping_to_list(opportunity.trajectories),
        "model_num_variables": opportunity.model_num_variables,
        "model_num_constraints": opportunity.model_num_constraints,
    }


def _opportunity_from_dict(data: object) -> SADDOpportunityRecord:
    if not isinstance(data, dict):
        msg = "each SADD opportunity must be a JSON object"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    try:
        window = _json_int_pair(data, "window")
        return SADDOpportunityRecord(
            target_pz=_json_str(data, "target_pz"),
            window=(window[0], window[1]),
            participating_ions=tuple(_json_int_list(data, "participating_ions")),
            status=_json_str(data, "status"),
            validation_status=_json_str(data, "validation_status"),
            phase_cost_before=_json_float(data, "phase_cost_before"),
            phase_cost_after=_json_optional_float(data, "phase_cost_after"),
            accepted=_json_bool(data, "accepted"),
            pulse_count=_json_int(data, "pulse_count"),
            transport_delta=_json_str_int_mapping(data, "transport_delta"),
            runtime_s=_json_float(data, "runtime_s"),
            message=_json_optional_str(data, "message"),
            eligible_ions=tuple(_json_int_list(data, "eligible_ions")),
            busy_ions=tuple(_json_int_list(data, "busy_ions")),
            rejected_no_active_segment_ions=tuple(_json_int_list(data, "rejected_no_active_segment_ions")),
            rejected_unreachable_ions=tuple(_json_int_list(data, "rejected_unreachable_ions")),
            selection_scores=tuple(_json_selection_scores(data, "selection_scores")),
            phase_before_by_ion=_json_float_mapping(data, "phase_before_by_ion"),
            phase_after_by_ion=_json_float_mapping(data, "phase_after_by_ion"),
            pulse_timesteps=_json_timesteps_mapping(data, "pulse_timesteps"),
            pulse_action_ids=_json_timesteps_mapping(data, "pulse_action_ids"),
            transport_actions=tuple(_json_str_list(data, "transport_actions")),
            trajectories=_json_timesteps_mapping(data, "trajectories"),
            model_num_variables=_json_int(data, "model_num_variables"),
            model_num_constraints=_json_int(data, "model_num_constraints"),
        )
    except (KeyError, TypeError, ValueError) as error:
        msg = "malformed SADD opportunity"
        raise ValueError(msg) from error


def _json_int_pair(data: Mapping[str, object], key: str) -> tuple[int, int]:
    values = _json_int_list(data, key)
    if len(values) != 2:
        msg = f"{key} must contain two integers"
        raise ValueError(msg)
    return values[0], values[1]


def _float_mapping_to_list(values: IonFloatMapping | None) -> list[list[int | float]] | None:
    if values is None:
        return None
    return [[ion, value] for ion, value in values.items()]


def _timesteps_mapping_to_list(values: IonTimestepsMapping | None) -> list[list[object]] | None:
    if values is None:
        return None
    return [[ion, list(timesteps)] for ion, timesteps in values.items()]


def _json_int(data: Mapping[str, object], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError
    return value


def _json_float(data: Mapping[str, object], key: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError
    return float(value)


def _json_optional_float(data: Mapping[str, object], key: str) -> float | None:
    value = data[key]
    return None if value is None else _json_float(data, key)


def _json_str(data: Mapping[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise TypeError
    return value


def _json_optional_str(data: Mapping[str, object], key: str) -> str | None:
    value = data[key]
    return None if value is None else _json_str(data, key)


def _json_bool(data: Mapping[str, object], key: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError
    return value


def _json_list(data: Mapping[str, object], key: str) -> list[object]:
    value = data[key]
    if not isinstance(value, list):
        raise TypeError
    return value


def _json_int_list(data: Mapping[str, object], key: str) -> list[int]:
    values = _json_list(data, key)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError
    return cast("list[int]", values)


def _json_str_list(data: Mapping[str, object], key: str) -> list[str]:
    values = _json_list(data, key)
    if any(not isinstance(value, str) for value in values):
        raise TypeError
    return cast("list[str]", values)


def _json_str_int_mapping(data: Mapping[str, object], key: str) -> dict[str, int]:
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError
    result: dict[str, int] = {}
    for name, delta in value.items():
        if not isinstance(name, str) or isinstance(delta, bool) or not isinstance(delta, int):
            raise TypeError
        result[name] = delta
    return result


def _json_selection_scores(data: Mapping[str, object], key: str) -> list[tuple[int, float, int]]:
    result: list[tuple[int, float, int]] = []
    for value in _json_list(data, key):
        if not isinstance(value, list) or len(value) != 3:
            raise TypeError
        ion, score, distance = value
        if (
            isinstance(ion, bool)
            or not isinstance(ion, int)
            or isinstance(score, bool)
            or not isinstance(score, int | float)
            or isinstance(distance, bool)
            or not isinstance(distance, int)
        ):
            raise TypeError
        result.append((ion, float(score), distance))
    return result


def _json_float_mapping(data: Mapping[str, object], key: str) -> dict[int, float] | None:
    values = data[key]
    if values is None:
        return None
    if not isinstance(values, list):
        raise TypeError
    result: dict[int, float] = {}
    for value in values:
        if not isinstance(value, list) or len(value) != 2:
            raise TypeError
        ion, number = value
        if (
            isinstance(ion, bool)
            or not isinstance(ion, int)
            or isinstance(number, bool)
            or not isinstance(number, int | float)
        ):
            raise TypeError
        result[ion] = float(number)
    return result


def _json_timesteps_mapping(data: Mapping[str, object], key: str) -> dict[int, tuple[int, ...]] | None:
    values = data[key]
    if values is None:
        return None
    if not isinstance(values, list):
        raise TypeError
    result: dict[int, tuple[int, ...]] = {}
    for value in values:
        if not isinstance(value, list) or len(value) != 2:
            raise TypeError
        ion, timesteps = value
        if isinstance(ion, bool) or not isinstance(ion, int) or not isinstance(timesteps, list):
            raise TypeError
        if any(isinstance(timestep, bool) or not isinstance(timestep, int) for timestep in timesteps):
            raise TypeError
        result[ion] = tuple(timesteps)
    return result


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


def _freeze_transport_delta(values: Mapping[str, int]) -> Mapping[str, int]:
    frozen: dict[str, int] = {}
    for action_type, delta in values.items():
        if not isinstance(action_type, str):
            msg = "transport_delta action types must be strings"
            raise TypeError(msg)
        if not action_type:
            msg = "transport_delta action types must be non-empty"
            raise ValueError(msg)
        if isinstance(delta, bool) or not isinstance(delta, int):
            msg = "transport_delta values must be integers"
            raise TypeError(msg)
        if delta == 0:
            msg = "transport_delta must omit unchanged action types"
            raise ValueError(msg)
        frozen[action_type] = delta
    return MappingProxyType(frozen)


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
