# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Idealized full-control Hahn reference for compiled Linear schedules."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, ClassVar

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, Rx, Ry, Rz
from mqt.ionshuttler.linear.dd.result import DDPassResult, LocalDDSequence
from mqt.ionshuttler.linear.dd.schedule_transform import rebuild_schedule
from mqt.ionshuttler.linear.dd.schemes import DDScheme, get_dd_scheme
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.dd.windows import find_idle_windows
from mqt.ionshuttler.linear.schedule import ActionSchedule, ScheduledAction

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture


@dataclass(frozen=True)
class IdealizedHahnConfig:
    """Configure the idealized full-control Hahn reference.

    The comparator permits local pulses at every ion position and concurrently
    with transport or other operations. Its output is therefore a reference
    schedule, not necessarily an executable schedule for the given hardware.
    """

    scheme: str | DDScheme = "hahn"
    min_idle_timesteps: int | None = None
    label: str = "IdealizedHahn"

    def __post_init__(self) -> None:
        """Validate the reference sequence configuration.

        Raises:
            TypeError: If a configuration value has the wrong type.
            ValueError: If a configuration value is outside its supported range.
        """
        if not isinstance(self.scheme, str | DDScheme):
            msg = "scheme must be a registered name or DDScheme"
            raise TypeError(msg)
        if self.min_idle_timesteps is not None:
            if isinstance(self.min_idle_timesteps, bool) or not isinstance(self.min_idle_timesteps, int):
                msg = "min_idle_timesteps must be an integer or None"
                raise TypeError(msg)
            if self.min_idle_timesteps < 1:
                msg = "min_idle_timesteps must be >= 1"
                raise ValueError(msg)
        if not isinstance(self.label, str):
            msg = "label must be a string"
            raise TypeError(msg)
        if not self.label:
            msg = "label must be non-empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class IdealizedHahnReport:
    """Summarize local pulse sequences inserted by the idealized reference."""

    report_type: ClassVar[str] = "idealized_hahn"

    sequences: tuple[LocalDDSequence, ...] = ()

    def __post_init__(self) -> None:
        """Freeze and validate insertion records.

        Raises:
            TypeError: If an item is not a local DD insertion record.
        """
        sequences = tuple(self.sequences)
        if any(not isinstance(sequence, LocalDDSequence) for sequence in sequences):
            msg = "sequences must contain LocalDDSequence values"
            raise TypeError(msg)
        object.__setattr__(self, "sequences", sequences)

    def to_dict(self) -> dict[str, object]:
        """Return this report using JSON-compatible values."""
        return {"sequences": [sequence.to_dict() for sequence in self.sequences]}

    @classmethod
    def from_dict(cls, data: object) -> IdealizedHahnReport:
        """Restore an idealized-Hahn report from JSON-compatible values.

        Returns:
            The restored report.

        Raises:
            ValueError: If the serialized report is malformed.
        """
        if not isinstance(data, dict) or not isinstance(data.get("sequences"), list):
            msg = "idealized Hahn report must contain a sequence list"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
        return cls(tuple(LocalDDSequence.from_dict(value) for value in data["sequences"]))


def apply_idealized_hahn(
    schedule: ActionSchedule,
    architecture: Architecture,
    config: IdealizedHahnConfig | None = None,
) -> DDPassResult[IdealizedHahnReport]:
    """Insert constraint-relaxed Hahn pulses into every eligible idle window.

    Pulse positions are rounded to schedule boundaries, clamped to their idle
    window, and deduplicated by boundary while retaining sequence order.

    Returns:
        The idealized schedule and its inserted local-pulse records.

    """
    resolved_config = config or IdealizedHahnConfig()
    scheme = _resolve_scheme(resolved_config.scheme)
    min_idle_timesteps = resolved_config.min_idle_timesteps or scheme.num_pulses
    timeline = build_timeline(schedule, architecture)
    pulses_by_time: dict[int, list[tuple[int, Action]]] = {}
    pending_sequences: list[tuple[int, tuple[int, int], tuple[int, ...]]] = []

    for ion, _site in schedule.initial_state.positions:
        for window in find_idle_windows(timeline, ion):
            if window[1] - window[0] < min_idle_timesteps:
                continue
            pulses = _rounded_scheme_pulses(window, scheme)
            if not pulses:
                continue
            sequence_index = len(pending_sequences)
            gate_timesteps = tuple(timestep for timestep, _spec in pulses)
            for timestep, spec in pulses:
                pulses_by_time.setdefault(timestep, []).append((sequence_index, _make_local_gate(spec, ion)))
            pending_sequences.append((ion, window, gate_timesteps))

    if not pending_sequences:
        return DDPassResult(schedule=schedule, architecture=architecture, report=IdealizedHahnReport())

    scheduled_actions, action_ids_by_sequence = _path_with_inserted_pulses(schedule, pulses_by_time)
    sequences = tuple(
        LocalDDSequence(
            ion=ion,
            window=window,
            scheme_name=resolved_config.label,
            pulse_timesteps=gate_timesteps,
            action_ids=tuple(action_ids_by_sequence[sequence_index]),
        )
        for sequence_index, (ion, window, gate_timesteps) in enumerate(pending_sequences)
    )
    report = IdealizedHahnReport(sequences)
    return DDPassResult(
        schedule=rebuild_schedule(schedule, scheduled_actions),
        architecture=architecture,
        report=report,
    )


def _resolve_scheme(scheme: str | DDScheme) -> DDScheme:
    return scheme if isinstance(scheme, DDScheme) else get_dd_scheme(scheme)


def _rounded_scheme_pulses(
    window: tuple[int, int],
    scheme: DDScheme,
) -> tuple[tuple[int, GateSpec], ...]:
    start, end = window
    length = end - start
    seen: set[int] = set()
    pulses: list[tuple[int, GateSpec]] = []
    for relative_time, spec in zip(scheme.relative_gate_times, scheme.gate_specs, strict=True):
        timestep = min(max(round(start + relative_time * length), start), end)
        if timestep in seen:
            continue
        seen.add(timestep)
        pulses.append((timestep, spec))
    return tuple(pulses)


def _make_local_gate(spec: GateSpec, ion: int) -> Action:
    if spec.theta is None:
        msg = f"gate specification for {spec.gate_name} requires theta"
        raise ValueError(msg)
    gate_types = {"Rx": Rx, "Ry": Ry, "Rz": Rz}
    try:
        gate_type = gate_types[spec.gate_name]
    except KeyError as error:
        msg = f"unsupported local DD gate: {spec.gate_name!r}"
        raise ValueError(msg) from error
    return gate_type(ion=ion, theta=spec.theta)


def _path_with_inserted_pulses(
    program: ActionSchedule,
    pulses_by_time: dict[int, list[tuple[int, Action]]],
) -> tuple[tuple[ScheduledAction, ...], dict[int, list[int]]]:
    if not pulses_by_time:
        return program.scheduled_actions, {}

    updated: list[ScheduledAction] = []
    action_ids_by_sequence: dict[int, list[int]] = {}
    action_ids = count(program.next_action_id)
    current_time = 0
    inserted_at_current_time = False
    for item in program.scheduled_actions:
        if not inserted_at_current_time and current_time in pulses_by_time:
            for sequence_index, action in pulses_by_time[current_time]:
                action_id = next(action_ids)
                updated.append(ScheduledAction(action_id, action))
                action_ids_by_sequence.setdefault(sequence_index, []).append(action_id)
            inserted_at_current_time = True
        updated.append(item)
        if isinstance(item.action, AdvanceTime):
            current_time += item.action.timestep_increment
            inserted_at_current_time = False
    if not inserted_at_current_time and current_time in pulses_by_time:
        for sequence_index, action in pulses_by_time[current_time]:
            action_id = next(action_ids)
            updated.append(ScheduledAction(action_id, action))
            action_ids_by_sequence.setdefault(sequence_index, []).append(action_id)
    return tuple(updated), action_ids_by_sequence


__all__ = ["IdealizedHahnConfig", "IdealizedHahnReport", "apply_idealized_hahn"]
