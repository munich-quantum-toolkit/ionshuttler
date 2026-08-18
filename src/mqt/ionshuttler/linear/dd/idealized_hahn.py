# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Idealized full-control Hahn reference for compiled Linear schedules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, Rx, Ry, Rz
from mqt.ionshuttler.linear.dd.result import DDPassResult
from mqt.ionshuttler.linear.dd.schedule_transform import rebuild_result
from mqt.ionshuttler.linear.dd.schemes import DDScheme, get_dd_scheme
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.dd.windows import find_idle_windows
from mqt.ionshuttler.linear.result import DDInsertionRecord

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.result import CompilationResult


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

    insertions: tuple[DDInsertionRecord, ...] = ()

    def __post_init__(self) -> None:
        """Freeze and validate insertion records.

        Raises:
            TypeError: If an item is not a local DD insertion record.
        """
        insertions = tuple(self.insertions)
        if any(not isinstance(record, DDInsertionRecord) for record in insertions):
            msg = "insertions must contain DDInsertionRecord values"
            raise TypeError(msg)
        object.__setattr__(self, "insertions", insertions)


def apply_idealized_hahn(
    result: CompilationResult,
    architecture: Architecture | None = None,
    config: IdealizedHahnConfig | None = None,
) -> DDPassResult[CompilationResult, IdealizedHahnReport]:
    """Insert constraint-relaxed Hahn pulses into every eligible idle window.

    Pulse positions are rounded to schedule boundaries, clamped to their idle
    window, and deduplicated by boundary while retaining sequence order.

    Returns:
        The idealized schedule and its inserted local-pulse records.

    Raises:
        ValueError: If replay metadata is absent or the named scheme is unknown.
    """
    resolved_architecture = architecture or result.architecture
    if resolved_architecture is None:
        msg = "architecture is required for idealized Hahn"
        raise ValueError(msg)
    if result.initial_state is None:
        msg = "result.initial_state is required for idealized Hahn"
        raise ValueError(msg)

    resolved_config = config or IdealizedHahnConfig()
    scheme = _resolve_scheme(resolved_config.scheme)
    min_idle_timesteps = resolved_config.min_idle_timesteps or scheme.num_pulses
    timeline = build_timeline(result, resolved_architecture)
    pulses_by_time: dict[int, list[Action]] = {}
    records: list[DDInsertionRecord] = []

    for ion, _site in result.initial_state.positions:
        for window in find_idle_windows(timeline, ion):
            if window[1] - window[0] < min_idle_timesteps:
                continue
            pulses = _rounded_scheme_pulses(window, scheme)
            if not pulses:
                continue
            gate_timesteps = tuple(timestep for timestep, _spec in pulses)
            for timestep, spec in pulses:
                pulses_by_time.setdefault(timestep, []).append(_make_local_gate(spec, ion))
            records.append(
                DDInsertionRecord(
                    ion=ion,
                    window=window,
                    scheme_name=resolved_config.label,
                    gate_timesteps=gate_timesteps,
                )
            )

    report = IdealizedHahnReport(tuple(records))
    if not records:
        return DDPassResult(program=result, report=report)

    path = _path_with_inserted_pulses(result.path, pulses_by_time)
    rebuilt = rebuild_result(result, path, resolved_architecture)
    updated = replace(rebuilt, dd_insertions=(*result.dd_insertions, *records))
    return DDPassResult(program=updated, report=report)


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
    path: list[Action],
    pulses_by_time: dict[int, list[Action]],
) -> list[Action]:
    if not pulses_by_time:
        return list(path)

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


__all__ = ["IdealizedHahnConfig", "IdealizedHahnReport", "apply_idealized_hahn"]
