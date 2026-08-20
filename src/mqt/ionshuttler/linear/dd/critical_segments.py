# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Replay logic-aware normalized phase over ion-local critical segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite, pi, sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from mqt.ionshuttler.linear.actions import (
    Action,
    GateAction,
    GlobalPulse,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    SingleQubitGate,
    TwoQubitGate,
)
from mqt.ionshuttler.linear.dd.frame_replay import build_frame_history
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.field_profile import FieldProfile
    from mqt.ionshuttler.linear.schedule import ActionSchedule

SegmentationMode = Literal["critical", "whole_schedule"]


@dataclass(frozen=True)
class CriticalSegment:
    """Describe one interval over which longitudinal phase remains comparable."""

    ion: int
    index: int
    start: int
    end: int
    positions: tuple[int, ...]
    toggling_signs: tuple[int, ...]
    sensitivities: tuple[float, ...]
    phase: float
    boundary_gate: str | None = None

    @property
    def squared_phase(self) -> float:
        """This segment's normalized squared-phase objective."""
        return self.phase**2


@dataclass(frozen=True)
class CriticalSegmentResult:
    """Contain logic-aware normalized-phase replay over critical segments.

    ``phase_cost`` is a schedule-ranking surrogate: the sum of squared normalized
    segment phases. It is not a joint-coherence decay exponent or a complete
    circuit-error model.
    """

    segments: tuple[CriticalSegment, ...]
    phase_per_ion: Mapping[int, float] = field(default_factory=dict)
    squared_phase_per_ion: Mapping[int, float] = field(default_factory=dict)
    phase_cost: float = 0.0
    sensitivity_profile: tuple[float, ...] = ()
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Freeze nested per-ion results."""
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "phase_per_ion", MappingProxyType(dict(self.phase_per_ion)))
        object.__setattr__(self, "squared_phase_per_ion", MappingProxyType(dict(self.squared_phase_per_ion)))


def normalized_sensitivity_values(
    architecture: Architecture,
    profile: FieldProfile | None = None,
) -> tuple[float, ...]:
    """Return a nonnegative dimensionless sensitivity envelope with unit RMS.

    Raises:
        ValueError: If the profile does not match the architecture or cannot be normalized.
    """
    selected = architecture.field_profile if profile is None else profile
    if selected is None:
        return (1.0,) * architecture.num_sites
    if selected.num_sites != architecture.num_sites:
        msg = "sensitivity profile num_sites must match architecture.num_sites"
        raise ValueError(msg)
    values = tuple(selected.field_at(site) for site in range(architecture.num_sites))
    if any(not isfinite(value) for value in values):
        msg = "sensitivity profile values must be finite"
        raise ValueError(msg)
    if any(value < 0.0 for value in values):
        msg = "sensitivity profile values must be non-negative"
        raise ValueError(msg)
    rms = sqrt(sum(value**2 for value in values) / len(values))
    if rms <= 0.0:
        msg = "sensitivity profile must have positive RMS"
        raise ValueError(msg)
    return tuple(value / rms for value in values)


def gate_z_effect(action: Action, ion: int) -> int | None:
    """Return a preserving gate's Z sign, or ``None`` for a mixing gate.

    Raises:
        TypeError: If ``action`` is not a supported algorithmic gate.
    """
    if isinstance(action, (Rz, Rzz)):
        return 1
    if isinstance(action, (Rx, Ry)):
        return 1 if action.ion != ion else _integer_pi_sign(action.theta)
    if isinstance(action, (Rxx, Ryy)):
        return 1 if ion not in {action.ion_a, action.ion_b} else _integer_pi_sign(action.theta)
    if isinstance(action, GateAction):
        msg = f"unsupported algorithmic gate for Z classification: {action!r}"
        raise TypeError(msg)
    msg = "gate_z_effect requires a gate action"
    raise TypeError(msg)


def compute_critical_segments(
    schedule: ActionSchedule,
    architecture: Architecture,
    *,
    sensitivity_profile: FieldProfile | None = None,
    dt: float = 1.0,
    segmentation: SegmentationMode = "critical",
    local_pulse_action_ids: frozenset[int] = frozenset(),
) -> CriticalSegmentResult:
    """Replay ion-local segments and their normalized squared-phase proxy.

    Returns:
        The reconstructed critical segments and their phase objective.

    Raises:
        ValueError: If the analysis parameters or architecture metadata are invalid.
    """
    if dt <= 0.0:
        msg = "dt must be positive"
        raise ValueError(msg)
    if segmentation not in {"critical", "whole_schedule"}:
        msg = "segmentation must be 'critical' or 'whole_schedule'"
        raise ValueError(msg)

    timeline = build_timeline(schedule, architecture)
    frame_history = build_frame_history(timeline, local_pulse_action_ids)
    sensitivities = normalized_sensitivity_values(architecture, sensitivity_profile)
    ion_ids = _ion_ids(schedule)
    segments: list[CriticalSegment] = []
    for ion in ion_ids:
        segment_start = 0
        local_sign = 1
        positions: list[int] = []
        signs: list[int] = []
        weights: list[float] = []
        segment_index = 0
        for timestep in range(timeline.makespan):
            algorithmic_gates = (
                _algorithmic_gates_for_ion(timeline, ion, timestep, local_pulse_action_ids)
                if segmentation == "critical"
                else ()
            )
            for action in algorithmic_gates:
                effect = gate_z_effect(action, ion)
                if effect is None:
                    if positions:
                        segments.append(
                            _make_segment(
                                ion,
                                segment_index,
                                segment_start,
                                timestep,
                                positions,
                                signs,
                                weights,
                                dt,
                                type(action).__name__,
                            )
                        )
                        segment_index += 1
                    segment_start = timestep
                    positions, signs, weights = [], [], []
                    local_sign = 1
                else:
                    local_sign *= effect
            position = timeline.ion_position(ion, timestep)
            positions.append(position)
            signs.append(local_sign * frame_history.phase_sign_for_ion(ion, timestep, axis="Z"))
            weights.append(sensitivities[position])
        if positions:
            segments.append(
                _make_segment(
                    ion,
                    segment_index,
                    segment_start,
                    timeline.makespan,
                    positions,
                    signs,
                    weights,
                    dt,
                    None,
                )
            )

    phase_per_ion = {ion: sum(segment.phase for segment in segments if segment.ion == ion) for ion in ion_ids}
    squared_per_ion = {ion: sum(segment.squared_phase for segment in segments if segment.ion == ion) for ion in ion_ids}
    return CriticalSegmentResult(
        segments=tuple(segments),
        phase_per_ion=phase_per_ion,
        squared_phase_per_ion=squared_per_ion,
        phase_cost=sum(squared_per_ion.values()),
        sensitivity_profile=sensitivities,
        dt=dt,
    )


def _make_segment(
    ion: int,
    index: int,
    start: int,
    end: int,
    positions: list[int],
    signs: list[int],
    sensitivities: list[float],
    dt: float,
    boundary_gate: str | None,
) -> CriticalSegment:
    return CriticalSegment(
        ion=ion,
        index=index,
        start=start,
        end=end,
        positions=tuple(positions),
        toggling_signs=tuple(signs),
        sensitivities=tuple(sensitivities),
        phase=dt * sum(sign * weight for sign, weight in zip(signs, sensitivities, strict=True)),
        boundary_gate=boundary_gate,
    )


def _algorithmic_gates_for_ion(
    timeline: CompiledTimeline,
    ion: int,
    timestep: int,
    local_pulse_action_ids: frozenset[int],
) -> tuple[SingleQubitGate | TwoQubitGate, ...]:
    gates: list[SingleQubitGate | TwoQubitGate] = []
    for item in timeline.scheduled_action_at(timestep) or ():
        action = item.action
        if isinstance(action, GlobalPulse):
            continue
        if isinstance(action, SingleQubitGate):
            if action.ion != ion or item.action_id in local_pulse_action_ids:
                continue
            gates.append(action)
        elif isinstance(action, TwoQubitGate) and ion in {action.ion_a, action.ion_b}:
            gates.append(action)
    return tuple(gates)


def _integer_pi_sign(theta: float) -> int | None:
    ratio = theta / pi
    nearest = round(ratio)
    if not isclose(ratio, nearest, abs_tol=1e-9):
        return None
    return -1 if nearest % 2 else 1


def _ion_ids(program: ActionSchedule) -> tuple[int, ...]:
    return tuple(sorted(ion for ion, _site in program.initial_state.positions))


__all__ = [
    "CriticalSegment",
    "CriticalSegmentResult",
    "SegmentationMode",
    "compute_critical_segments",
    "gate_z_effect",
    "normalized_sensitivity_values",
]
