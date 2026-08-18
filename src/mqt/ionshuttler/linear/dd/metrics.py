# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Comparison metrics for dynamical-decoupling schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast

from mqt.ionshuttler.linear.actions import (
    GlobalPulse,
    PhysicalSwap,
    Rx,
    Ry,
    Rz,
    Shuttle,
    SingleQubitGate,
    TwoQubitGate,
)
from mqt.ionshuttler.linear.dd.frame_replay import (
    FrameHistory,
    accumulated_frame_phase,
    build_frame_history,
    local_dd_action_slots,
)
from mqt.ionshuttler.linear.dd.phase import accumulated_phase
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mqt.ionshuttler.linear.result import CompilationResult


class _TimedGate(Protocol):
    """Describe the duration field shared by concrete physical gates."""

    duration: int


@dataclass(frozen=True)
class ResidualPhaseSummary:
    """Summarize longitudinal phase remaining in a transformed schedule."""

    residual_phase_by_ion: Mapping[int, float] = field(default_factory=dict)
    sum_absolute_residual_phase: float = 0.0
    sum_squared_residual_phase: float = 0.0
    max_absolute_residual_phase: float = 0.0
    sum_absolute_residual_phases_at_gates: float = 0.0

    def __post_init__(self) -> None:
        """Freeze per-ion phase values."""
        object.__setattr__(self, "residual_phase_by_ion", MappingProxyType(dict(self.residual_phase_by_ion)))


@dataclass(frozen=True)
class GateResidualEvent:
    """Record prefix residual phase when an algorithmic gate begins."""

    ion: int
    timestep: int
    duration: int
    gate_name: str
    residual_phase_at_gate: float


def decoupling_ratio(result: CompilationResult) -> float:
    """Return the fraction of ion-time volume covered by recorded DD windows."""
    num_ions = _infer_num_ions(result)
    if result.num_timesteps <= 0 or num_ions <= 0 or not result.dd_insertions:
        return 0.0
    windows_by_ion: dict[int, list[tuple[int, int]]] = {}
    for record in result.dd_insertions:
        windows_by_ion.setdefault(record.ion, []).append(record.window)
    covered_volume = sum(end - start for windows in windows_by_ion.values() for start, end in _merge_windows(windows))
    return covered_volume / (result.num_timesteps * num_ions)


def relative_phase_reduction(result: CompilationResult) -> float:
    """Return recorded absolute phase reduction relative to unrefocused phase."""
    if result.architecture is None or not result.dd_insertions:
        return 0.0
    timeline = build_timeline(result, result.architecture)
    total_phase = sum(
        abs(
            accumulated_phase(
                timeline,
                ion=ion,
                t_start=0,
                t_end=timeline.makespan,
                field_profile=result.architecture.field_profile,
            )
        )
        for ion in _infer_ion_ids(result)
    )
    if total_phase <= 0.0:
        return 0.0
    return sum(record.phase_reduction for record in result.dd_insertions) / total_phase


def phase_reduction_per_gate(result: CompilationResult) -> float:
    """Return recorded absolute phase reduction per inserted pulse."""
    num_inserted_gates = sum(len(record.gate_timesteps) for record in result.dd_insertions)
    if num_inserted_gates == 0:
        return 0.0
    return sum(record.phase_reduction for record in result.dd_insertions) / num_inserted_gates


def summarize_residual_phases(
    result: CompilationResult,
    timeline: CompiledTimeline | None = None,
) -> ResidualPhaseSummary:
    """Compute schedule-end and algorithmic-gate residual phase metrics.

    Returns:
        The aggregate and per-ion residual-phase values.
    """
    if result.architecture is None:
        return ResidualPhaseSummary()
    resolved_timeline = timeline or build_timeline(result, result.architecture)
    frame_history = build_frame_history(resolved_timeline, result=result)
    per_ion = {
        ion: accumulated_frame_phase(
            resolved_timeline,
            ion=ion,
            t_start=0,
            t_end=resolved_timeline.makespan,
            field_profile=result.architecture.field_profile,
            frame_history=frame_history,
        )
        for ion in _infer_ion_ids(result)
    }
    absolute = tuple(abs(phase) for phase in per_ion.values())
    return ResidualPhaseSummary(
        residual_phase_by_ion=per_ion,
        sum_absolute_residual_phase=sum(absolute),
        sum_squared_residual_phase=sum(phase * phase for phase in per_ion.values()),
        max_absolute_residual_phase=max(absolute, default=0.0),
        sum_absolute_residual_phases_at_gates=_sum_absolute_residual_phases_at_gates(
            result,
            resolved_timeline,
            frame_history,
        ),
    )


def residual_phase_by_ion(result: CompilationResult) -> dict[int, float]:
    """Return schedule-end residual phase for every ion."""
    return dict(summarize_residual_phases(result).residual_phase_by_ion)


def sum_absolute_residual_phase(result: CompilationResult) -> float:
    """Return the sum of absolute schedule-end residual phases."""
    return summarize_residual_phases(result).sum_absolute_residual_phase


def sum_squared_residual_phase(result: CompilationResult) -> float:
    """Return the sum of squared schedule-end residual phases."""
    return summarize_residual_phases(result).sum_squared_residual_phase


def max_absolute_residual_phase(result: CompilationResult) -> float:
    """Return the largest absolute schedule-end residual phase."""
    return summarize_residual_phases(result).max_absolute_residual_phase


def sum_absolute_residual_phases_at_gates(result: CompilationResult) -> float:
    """Return summed absolute prefix phase at algorithmic gate boundaries."""
    if result.architecture is None:
        return 0.0
    timeline = build_timeline(result, result.architecture)
    history = build_frame_history(timeline, result=result)
    return _sum_absolute_residual_phases_at_gates(result, timeline, history)


def gate_residual_events(
    result: CompilationResult,
    timeline: CompiledTimeline | None = None,
    frame_history: FrameHistory | None = None,
) -> tuple[GateResidualEvent, ...]:
    """Return ordered residual-phase observations at algorithmic gates."""
    if result.architecture is None:
        return ()
    resolved_timeline = timeline or build_timeline(result, result.architecture)
    history = frame_history or build_frame_history(resolved_timeline, result=result)
    local_dd_slots = local_dd_action_slots(resolved_timeline, result)
    events: list[GateResidualEvent] = []
    for timestep in range(resolved_timeline.makespan):
        for action_index, action in enumerate(resolved_timeline.action_at(timestep) or ()):
            if isinstance(action, GlobalPulse):
                continue
            if isinstance(action, SingleQubitGate):
                if (timestep, action_index) in local_dd_slots:
                    continue
                events.append(_gate_event(result, resolved_timeline, history, action.ion, timestep, action))
            elif isinstance(action, TwoQubitGate):
                events.extend(
                    _gate_event(result, resolved_timeline, history, ion, timestep, action)
                    for ion in (action.ion_a, action.ion_b)
                )
    return tuple(events)


def gate_residual_events_by_ion(
    result: CompilationResult,
    timeline: CompiledTimeline | None = None,
    frame_history: FrameHistory | None = None,
) -> dict[int, tuple[GateResidualEvent, ...]]:
    """Group ordered gate residual events by ion.

    Returns:
        The ordered residual events for each participating ion.
    """
    grouped: dict[int, list[GateResidualEvent]] = {}
    for event in gate_residual_events(result, timeline=timeline, frame_history=frame_history):
        grouped.setdefault(event.ion, []).append(event)
    return {ion: tuple(events) for ion, events in grouped.items()}


def rank_ions_by_residual_phase(
    result: CompilationResult,
    timeline: CompiledTimeline | None = None,
) -> tuple[tuple[int, float], ...]:
    """Rank ions by descending absolute residual phase and then identifier.

    Returns:
        Pairs of ion identifiers and residual phases in rank order.
    """
    summary = summarize_residual_phases(result, timeline=timeline)
    return tuple(sorted(summary.residual_phase_by_ion.items(), key=lambda entry: (-abs(entry[1]), entry[0])))


def window_residual_phase(
    result: CompilationResult,
    ion: int,
    window: tuple[int, int],
    timeline: CompiledTimeline | None = None,
    frame_history: FrameHistory | None = None,
) -> float:
    """Return frame-aware residual phase within a half-open window."""
    if result.architecture is None:
        return 0.0
    resolved_timeline = timeline or build_timeline(result, result.architecture)
    history = frame_history or build_frame_history(resolved_timeline, result=result)
    return accumulated_frame_phase(
        resolved_timeline,
        ion,
        window[0],
        window[1],
        result.architecture.field_profile,
        history,
    )


def residual_phase_at_timestep(
    result: CompilationResult,
    ion: int,
    timestep: int,
    timeline: CompiledTimeline | None = None,
    frame_history: FrameHistory | None = None,
) -> float:
    """Return frame-aware prefix phase at a schedule boundary.

    Raises:
        ValueError: If ``timestep`` lies outside the schedule.
    """
    if result.architecture is None:
        return 0.0
    resolved_timeline = timeline or build_timeline(result, result.architecture)
    if not 0 <= timestep <= resolved_timeline.makespan:
        msg = f"timestep must be within [0, {resolved_timeline.makespan}]"
        raise ValueError(msg)
    history = frame_history or build_frame_history(resolved_timeline, result=result)
    return accumulated_frame_phase(
        resolved_timeline,
        ion,
        0,
        timestep,
        result.architecture.field_profile,
        history,
    )


def residual_phase_at_window_end(
    result: CompilationResult,
    ion: int,
    window: tuple[int, int],
    timeline: CompiledTimeline | None = None,
    frame_history: FrameHistory | None = None,
) -> float:
    """Return inherited prefix phase at a window's closing boundary."""
    return residual_phase_at_timestep(result, ion, window[1], timeline, frame_history)


def residual_phase_at_window_end_reduction(
    before_result: CompilationResult,
    after_result: CompilationResult,
    ion: int,
    window: tuple[int, int],
    before_timeline: CompiledTimeline | None = None,
    after_timeline: CompiledTimeline | None = None,
    before_frame_history: FrameHistory | None = None,
    after_frame_history: FrameHistory | None = None,
) -> float:
    """Return reduction in absolute inherited phase at a window boundary."""
    before_phase = residual_phase_at_window_end(before_result, ion, window, before_timeline, before_frame_history)
    after_phase = residual_phase_at_window_end(after_result, ion, window, after_timeline, after_frame_history)
    return abs(before_phase) - abs(after_phase)


def _gate_event(
    result: CompilationResult,
    timeline: CompiledTimeline,
    frame_history: FrameHistory,
    ion: int,
    timestep: int,
    action: SingleQubitGate | TwoQubitGate,
) -> GateResidualEvent:
    return GateResidualEvent(
        ion=ion,
        timestep=timestep,
        duration=cast("_TimedGate", action).duration,
        gate_name=type(action).__name__,
        residual_phase_at_gate=residual_phase_at_timestep(
            result,
            ion,
            timestep,
            timeline,
            frame_history,
        ),
    )


def _sum_absolute_residual_phases_at_gates(
    result: CompilationResult,
    timeline: CompiledTimeline,
    frame_history: FrameHistory,
) -> float:
    return sum(
        abs(event.residual_phase_at_gate)
        for event in gate_residual_events(result, timeline=timeline, frame_history=frame_history)
    )


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    merged = [min(windows)]
    for start, end in sorted(windows)[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _infer_ion_ids(result: CompilationResult) -> tuple[int, ...]:
    if result.initial_state is not None:
        return tuple(ion for ion, _site in result.initial_state.positions)
    ion_ids: set[int] = set()
    for action in result.path:
        if isinstance(action, (Shuttle, Rx, Ry, Rz)):
            ion_ids.add(action.ion)
        elif isinstance(action, (PhysicalSwap, TwoQubitGate)):
            ion_ids.update((action.ion_a, action.ion_b))
    ion_ids.update(record.ion for record in result.dd_insertions)
    return tuple(sorted(ion_ids))


def _infer_num_ions(result: CompilationResult) -> int:
    return len(_infer_ion_ids(result))


__all__ = [
    "GateResidualEvent",
    "ResidualPhaseSummary",
    "decoupling_ratio",
    "gate_residual_events",
    "gate_residual_events_by_ion",
    "max_absolute_residual_phase",
    "phase_reduction_per_gate",
    "rank_ions_by_residual_phase",
    "relative_phase_reduction",
    "residual_phase_at_timestep",
    "residual_phase_at_window_end",
    "residual_phase_at_window_end_reduction",
    "residual_phase_by_ion",
    "sum_absolute_residual_phase",
    "sum_absolute_residual_phases_at_gates",
    "sum_squared_residual_phase",
    "summarize_residual_phases",
    "window_residual_phase",
]
