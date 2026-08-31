# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Nearest-feasible midpoint Hahn comparator for compiled Linear schedules."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from mqt.ionshuttler.linear.actions import GateSpec, Rx
from mqt.ionshuttler.linear.dd.result import DDPassResult, LocalDDSequence
from mqt.ionshuttler.linear.dd.schedule_transform import (
    insert_action_at_time,
    local_gate_for_spec,
    validate_rebuilt_schedule,
    validate_schedule_compatibility,
)
from mqt.ionshuttler.linear.dd.schemes import MIDPOINT_ONLY_HAHN
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.dd.windows import find_idle_windows
from mqt.ionshuttler.linear.schedule import ScheduledAction

from ..._json_utils import require_int, require_int_list, require_list, require_mapping, require_str

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline
    from mqt.ionshuttler.linear.schedule import ActionSchedule

NearestHahnStatus = Literal["exact", "shifted", "skipped"]
NearestHahnSkipReason = Literal[
    "no_processing_zone_access",
    "ion_busy",
    "processing_zone_busy",
    "schedule_validation_failed",
]

_X_PI = GateSpec("Rx", theta=pi)
_STATUSES: frozenset[str] = frozenset({"exact", "shifted", "skipped"})
_SKIP_REASONS: frozenset[str] = frozenset({
    "no_processing_zone_access",
    "ion_busy",
    "processing_zone_busy",
    "schedule_validation_failed",
})


@dataclass(frozen=True)
class NearestHahnConfig:
    """Configure the schedule-preserving nearest-feasible Hahn comparator."""

    min_idle_timesteps: int = 2
    label: str = "NearestHahn"

    def __post_init__(self) -> None:
        """Validate the comparator configuration.

        Raises:
            TypeError: If a configuration value has the wrong type.
            ValueError: If a configuration value is outside its supported range.
        """
        if isinstance(self.min_idle_timesteps, bool) or not isinstance(self.min_idle_timesteps, int):
            msg = "min_idle_timesteps must be an integer"
            raise TypeError(msg)
        if self.min_idle_timesteps < 2:
            msg = "min_idle_timesteps must be >= 2"
            raise ValueError(msg)
        if not isinstance(self.label, str):
            msg = "label must be a string"
            raise TypeError(msg)
        if not self.label:
            msg = "label must be non-empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class NearestHahnOpportunityRecord:
    """Describe the placement outcome for one eligible ion-local idle window."""

    ion: int
    window: tuple[int, int]
    ideal_midpoint: int
    status: NearestHahnStatus
    selected_timestep: int | None = None
    processing_zone: str | None = None
    signed_displacement: int | None = None
    absolute_displacement: int | None = None
    skip_reason: NearestHahnSkipReason | None = None

    def __post_init__(self) -> None:
        """Validate one placement audit entry.

        Raises:
            TypeError: If a record value has the wrong type.
            ValueError: If a record value is outside its supported range.
        """
        if isinstance(self.ion, bool) or not isinstance(self.ion, int):
            msg = "ion must be an integer"
            raise TypeError(msg)
        window = tuple(self.window)
        if len(window) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in window):
            msg = "window must contain two integers"
            raise TypeError(msg)
        if window[0] < 0 or window[1] < window[0]:
            msg = "window must be an ordered non-negative interval"
            raise ValueError(msg)
        object.__setattr__(self, "window", window)
        if self.status not in _STATUSES:
            msg = f"status must be one of {sorted(_STATUSES)}"
            raise ValueError(msg)
        if self.skip_reason is not None and self.skip_reason not in _SKIP_REASONS:
            msg = f"skip_reason must be one of {sorted(_SKIP_REASONS)}"
            raise ValueError(msg)
        if (self.status == "skipped") != (self.skip_reason is not None):
            msg = "skip_reason must be present exactly when the window was skipped"
            raise ValueError(msg)
        if (self.status == "skipped") == (self.selected_timestep is not None):
            msg = "selected_timestep must be present exactly when a pulse was placed"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return this record using JSON-compatible values."""
        return {
            "ion": self.ion,
            "window": list(self.window),
            "ideal_midpoint": self.ideal_midpoint,
            "status": self.status,
            "selected_timestep": self.selected_timestep,
            "processing_zone": self.processing_zone,
            "signed_displacement": self.signed_displacement,
            "absolute_displacement": self.absolute_displacement,
            "skip_reason": self.skip_reason,
        }

    @classmethod
    def from_dict(cls, data: object) -> NearestHahnOpportunityRecord:
        """Restore one placement record from JSON-compatible values.

        Returns:
            The restored record.

        Raises:
            ValueError: If the serialized record is malformed.
        """
        mapping = require_mapping(data, "nearest Hahn opportunity record")
        window = require_int_list(mapping, "window")
        if len(window) != 2:
            msg = "window must contain two integers"
            raise ValueError(msg)
        return cls(
            ion=require_int(mapping, "ion"),
            window=(window[0], window[1]),
            ideal_midpoint=require_int(mapping, "ideal_midpoint"),
            status=cast("NearestHahnStatus", require_str(mapping, "status")),
            selected_timestep=_optional_int(mapping, "selected_timestep"),
            processing_zone=_optional_str(mapping, "processing_zone"),
            signed_displacement=_optional_int(mapping, "signed_displacement"),
            absolute_displacement=_optional_int(mapping, "absolute_displacement"),
            skip_reason=cast("NearestHahnSkipReason | None", _optional_str(mapping, "skip_reason")),
        )


@dataclass(frozen=True)
class NearestHahnReport:
    """Summarize nearest-Hahn placements and the windows it could not use."""

    report_type: ClassVar[str] = "nearest_hahn"

    sequences: tuple[LocalDDSequence, ...] = ()
    opportunities: tuple[NearestHahnOpportunityRecord, ...] = ()

    def __post_init__(self) -> None:
        """Freeze and validate the report collections.

        Raises:
            TypeError: If a collection contains an unexpected value.
        """
        sequences = tuple(self.sequences)
        if any(not isinstance(sequence, LocalDDSequence) for sequence in sequences):
            msg = "sequences must contain LocalDDSequence values"
            raise TypeError(msg)
        opportunities = tuple(self.opportunities)
        if any(not isinstance(record, NearestHahnOpportunityRecord) for record in opportunities):
            msg = "opportunities must contain NearestHahnOpportunityRecord values"
            raise TypeError(msg)
        object.__setattr__(self, "sequences", sequences)
        object.__setattr__(self, "opportunities", opportunities)

    @property
    def placed(self) -> int:
        """The number of windows that received a pulse."""
        return sum(1 for record in self.opportunities if record.status != "skipped")

    def to_dict(self) -> dict[str, object]:
        """Return this report using JSON-compatible values."""
        return {
            "sequences": [sequence.to_dict() for sequence in self.sequences],
            "opportunities": [record.to_dict() for record in self.opportunities],
        }

    @classmethod
    def from_dict(cls, data: object) -> NearestHahnReport:
        """Restore a nearest-Hahn report from JSON-compatible values.

        Returns:
            The restored report.
        """
        mapping = require_mapping(data, "nearest Hahn report")
        return cls(
            sequences=tuple(LocalDDSequence.from_dict(value) for value in require_list(mapping, "sequences")),
            opportunities=tuple(
                NearestHahnOpportunityRecord.from_dict(value) for value in require_list(mapping, "opportunities")
            ),
        )


@dataclass(frozen=True)
class _Opportunity:
    ion: int
    window: tuple[int, int]
    ideal_midpoint: int


def run_nearest_hahn(
    schedule: ActionSchedule,
    architecture: Architecture,
    config: NearestHahnConfig | None = None,
) -> DDPassResult[NearestHahnReport]:
    """Place one X pulse as near each eligible idle-window midpoint as the schedule allows.

    The pass uses only processing-zone access the compiled trajectory already
    provides. It inserts no transport, never extends the makespan, and adds no
    physical recovery pulse, so the resulting Pauli frame persists and is
    discharged virtually by a consumer that corrects the terminal frame. Unlike
    :func:`~mqt.ionshuttler.linear.dd.apply_idealized_hahn`, every returned
    schedule is executable on the given architecture.

    The pass is intended for a base schedule. Applying it to an already
    augmented schedule is permitted, but the returned report then owns only the
    pulses this call inserted.

    Args:
        schedule: Base schedule to augment.
        architecture: Hardware model the schedule runs on.
        config: Optional comparator settings.

    Returns:
        The augmented schedule and its complete eligible-window audit.
    """
    resolved_config = config or NearestHahnConfig()
    validate_schedule_compatibility(schedule, architecture)
    base_timeline = build_timeline(schedule, architecture)
    opportunities = _eligible_opportunities(schedule, base_timeline, resolved_config)

    updated, sequences, records = _place_all(
        schedule,
        base_timeline,
        opportunities,
        architecture,
        label=resolved_config.label,
        expected_makespan=schedule.num_timesteps,
        validate_each=False,
    )
    if updated is not schedule and not validate_rebuilt_schedule(updated, architecture):
        updated, sequences, records = _place_all(
            schedule,
            base_timeline,
            opportunities,
            architecture,
            label=resolved_config.label,
            expected_makespan=schedule.num_timesteps,
            validate_each=True,
        )

    return DDPassResult(
        schedule=updated,
        architecture=architecture,
        report=NearestHahnReport(sequences=tuple(sequences), opportunities=tuple(records)),
    )


def _place_all(
    schedule: ActionSchedule,
    base_timeline: CompiledTimeline,
    opportunities: tuple[_Opportunity, ...],
    architecture: Architecture,
    *,
    label: str,
    expected_makespan: int,
    validate_each: bool,
) -> tuple[ActionSchedule, list[LocalDDSequence], list[NearestHahnOpportunityRecord]]:
    updated = schedule
    timeline = base_timeline
    sequences: list[LocalDDSequence] = []
    records: list[NearestHahnOpportunityRecord] = []
    for opportunity in opportunities:
        updated, timeline, sequence, record = _place_opportunity(
            updated,
            timeline,
            architecture,
            opportunity,
            label=label,
            expected_makespan=expected_makespan,
            validate=validate_each,
        )
        if sequence is not None:
            sequences.append(sequence)
        records.append(record)
    return updated, sequences, records


def _eligible_opportunities(
    schedule: ActionSchedule,
    timeline: CompiledTimeline,
    config: NearestHahnConfig,
) -> tuple[_Opportunity, ...]:
    opportunities = [
        _Opportunity(ion=ion, window=window, ideal_midpoint=_ideal_midpoint(window))
        for ion, _site in sorted(schedule.initial_state.positions)
        for window in find_idle_windows(timeline, ion)
        if window[1] - window[0] >= config.min_idle_timesteps
    ]
    return tuple(sorted(opportunities, key=lambda item: (item.ideal_midpoint, item.ion, item.window)))


def _ideal_midpoint(window: tuple[int, int]) -> int:
    """Return the window's midpoint under the shared boundary quantization.

    Returns:
        The boundary the midpoint-only Hahn sequence would target.
    """
    (timestep, _spec), *_rest = MIDPOINT_ONLY_HAHN.clamped_pulses(window)
    return timestep


def _place_opportunity(
    schedule: ActionSchedule,
    timeline: CompiledTimeline,
    architecture: Architecture,
    opportunity: _Opportunity,
    *,
    label: str,
    expected_makespan: int,
    validate: bool,
) -> tuple[ActionSchedule, CompiledTimeline, LocalDDSequence | None, NearestHahnOpportunityRecord]:
    start, end = opportunity.window
    candidates = sorted(range(start, end), key=lambda timestep: (abs(timestep - opportunity.ideal_midpoint), timestep))

    saw_zone = False
    saw_free_ion = False
    saw_free_zone = False
    for timestep in candidates:
        site = timeline.ion_position(opportunity.ion, timestep)
        zone = architecture.get_processing_zone(site)
        if zone is None:
            continue
        saw_zone = True
        gate = local_gate_for_spec(_X_PI, opportunity.ion)
        duration = cast("Rx", gate).duration
        occupied = range(timestep, timestep + duration)
        if timestep + duration > end or any(timeline.ion_busy(opportunity.ion, busy_t) for busy_t in occupied):
            continue
        saw_free_ion = True
        if any(timeline.pz_busy(zone, busy_t) for busy_t in occupied):
            continue
        saw_free_zone = True
        inserted = ScheduledAction(schedule.next_action_id, gate)
        patched_timeline = timeline.with_inserted_single_qubit_gate(inserted, zone, timestep)
        try:
            patched = insert_action_at_time(schedule, architecture, timestep, gate, timeline=timeline)
        except ValueError:
            continue
        if patched.num_timesteps != expected_makespan:
            continue
        if validate and not validate_rebuilt_schedule(patched, architecture):
            continue
        displacement = timestep - opportunity.ideal_midpoint
        sequence = LocalDDSequence(
            ion=opportunity.ion,
            window=opportunity.window,
            scheme_name=label,
            pulse_timesteps=(timestep,),
            action_ids=(inserted.action_id,),
        )
        return (
            patched,
            patched_timeline,
            sequence,
            NearestHahnOpportunityRecord(
                ion=opportunity.ion,
                window=opportunity.window,
                ideal_midpoint=opportunity.ideal_midpoint,
                status="exact" if displacement == 0 else "shifted",
                selected_timestep=timestep,
                processing_zone=zone,
                signed_displacement=displacement,
                absolute_displacement=abs(displacement),
            ),
        )

    return (
        schedule,
        timeline,
        None,
        _skipped_record(
            opportunity,
            saw_zone=saw_zone,
            saw_free_ion=saw_free_ion,
            saw_free_zone=saw_free_zone,
        ),
    )


def _skipped_record(
    opportunity: _Opportunity,
    *,
    saw_zone: bool,
    saw_free_ion: bool,
    saw_free_zone: bool,
) -> NearestHahnOpportunityRecord:
    if not saw_zone:
        reason: NearestHahnSkipReason = "no_processing_zone_access"
    elif not saw_free_ion:
        reason = "ion_busy"
    elif not saw_free_zone:
        reason = "processing_zone_busy"
    else:
        reason = "schedule_validation_failed"
    return NearestHahnOpportunityRecord(
        ion=opportunity.ion,
        window=opportunity.window,
        ideal_midpoint=opportunity.ideal_midpoint,
        status="skipped",
        skip_reason=reason,
    )


def _optional_int(mapping: dict[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer or null"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    return value


def _optional_str(mapping: dict[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{key} must be a string or null"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    return value


__all__ = [
    "NearestHahnConfig",
    "NearestHahnOpportunityRecord",
    "NearestHahnReport",
    "NearestHahnSkipReason",
    "NearestHahnStatus",
    "run_nearest_hahn",
]
