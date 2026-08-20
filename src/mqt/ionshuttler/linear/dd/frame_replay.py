# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Replay Pauli frames induced by local and global decoupling pulses."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, pi
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from mqt.ionshuttler.linear.actions import (
    Action,
    AdvanceTime,
    GateAction,
    GateSpec,
    GlobalPulse,
    PhysicalSwap,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    Shuttle,
    TransportAction,
)
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mqt.ionshuttler.linear.field_profile import FieldProfile
    from mqt.ionshuttler.linear.schedule import ActionSchedule, ScheduledAction

FrameActionKind = Literal[
    "global_dd_pulse",
    "local_dd_pulse",
    "transport",
    "algorithmic_gate",
    "advance_time",
    "other",
]

_PAULI_LABELS = frozenset({"I", "X", "Y", "Z"})
_PAULI_COMPOSITION: dict[tuple[str, str], str] = {
    ("I", "I"): "I",
    ("I", "X"): "X",
    ("I", "Y"): "Y",
    ("I", "Z"): "Z",
    ("X", "I"): "X",
    ("X", "X"): "I",
    ("X", "Y"): "Z",
    ("X", "Z"): "Y",
    ("Y", "I"): "Y",
    ("Y", "X"): "Z",
    ("Y", "Y"): "I",
    ("Y", "Z"): "X",
    ("Z", "I"): "Z",
    ("Z", "X"): "Y",
    ("Z", "Y"): "X",
    ("Z", "Z"): "I",
}


@dataclass(frozen=True)
class PauliFrameOperation:
    """Represent one Pauli operation applied to a tracked frame."""

    label: str

    def __post_init__(self) -> None:
        """Validate and normalize the Pauli label."""
        object.__setattr__(self, "label", _require_pauli_label(self.label))


@dataclass(frozen=True)
class PauliFrame:
    """Represent a Pauli frame while ignoring physically irrelevant phase."""

    label: str = "I"

    def __post_init__(self) -> None:
        """Validate and normalize the Pauli label."""
        object.__setattr__(self, "label", _require_pauli_label(self.label))

    def compose(self, other: PauliFrame | PauliFrameOperation) -> PauliFrame:
        """Return the frame after applying another Pauli operation."""
        return PauliFrame(_PAULI_COMPOSITION[self.label, other.label])

    def anticommutes_with(self, axis: str) -> bool:
        """Return whether this frame anticommutes with a Pauli axis."""
        normalized_axis = _require_pauli_label(axis)
        if self.label == "I" or normalized_axis == "I":
            return False
        return self.label != normalized_axis

    def phase_sign(self, axis: str = "Z") -> int:
        """Return the sign induced on evolution around a Pauli axis."""
        return -1 if self.anticommutes_with(axis) else 1


@dataclass(frozen=True)
class FrameHistory:
    """Store global and ion-local Pauli frames at every schedule boundary."""

    global_frames_by_time: tuple[PauliFrame, ...]
    local_frame_overrides: Mapping[int, tuple[PauliFrame, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy nested frame sequences into an immutable mapping."""
        object.__setattr__(
            self,
            "local_frame_overrides",
            MappingProxyType({ion: tuple(frames) for ion, frames in self.local_frame_overrides.items()}),
        )

    def frame_for_ion(self, ion: int, timestep: int) -> PauliFrame:
        """Return the combined global and local frame for an ion.

        Raises:
            ValueError: If ``timestep`` lies outside the tracked schedule.
        """
        if timestep < 0 or timestep >= len(self.global_frames_by_time):
            msg = f"timestep must be within [0, {len(self.global_frames_by_time) - 1}]"
            raise ValueError(msg)
        frame = self.global_frames_by_time[timestep]
        overrides = self.local_frame_overrides.get(ion)
        return frame if overrides is None else frame.compose(overrides[timestep])

    def phase_sign_for_ion(self, ion: int, timestep: int, axis: str = "Z") -> int:
        """Return the frame-induced evolution sign for an ion and axis."""
        return self.frame_for_ion(ion, timestep).phase_sign(axis)


@dataclass(frozen=True)
class FramedActionEvent:
    """Describe one scheduled action together with its effective ion frames."""

    timestep: int
    action: Action
    kind: FrameActionKind
    ion_frames: tuple[tuple[int, PauliFrame], ...]


def build_frame_history(
    timeline: CompiledTimeline,
    local_pulse_action_ids: frozenset[int] = frozenset(),
) -> FrameHistory:
    """Replay Pauli frames through every schedule boundary.

    Entry ``t`` is the frame after decoupling pulses at boundary ``t``. For
    ``t < makespan``, it applies to phase accumulation on ``[t, t + 1)``. The
    final entry represents the frame after terminal pulses.

    Returns:
        The global and ion-local frame at every schedule boundary.
    """
    frames_by_time: list[PauliFrame] = []
    current_frame = PauliFrame()
    for timestep in range(timeline.makespan + 1):
        for action in timeline.action_at(timestep) or ():
            operation = frame_operation_for_action(action)
            if operation is not None:
                current_frame = current_frame.compose(operation)
        frames_by_time.append(current_frame)
    return FrameHistory(
        global_frames_by_time=tuple(frames_by_time),
        local_frame_overrides=_build_local_frame_overrides(timeline, local_pulse_action_ids),
    )


def framed_action_events(
    schedule: ActionSchedule,
    timeline: CompiledTimeline | None = None,
    local_pulse_action_ids: frozenset[int] = frozenset(),
) -> tuple[FramedActionEvent, ...]:
    """Return ordered schedule actions annotated with their effective frames."""
    resolved_timeline = build_timeline(schedule) if timeline is None else timeline
    frame_history = build_frame_history(resolved_timeline, local_pulse_action_ids)
    events: list[FramedActionEvent] = []
    for timestep in range(resolved_timeline.makespan + 1):
        for scheduled_action in resolved_timeline.scheduled_action_at(timestep) or ():
            action = scheduled_action.action
            events.append(
                FramedActionEvent(
                    timestep=timestep,
                    action=action,
                    kind=_framed_action_kind(scheduled_action, local_pulse_action_ids),
                    ion_frames=_event_ion_frames(action, timestep, resolved_timeline, frame_history),
                )
            )
    return tuple(events)


def global_pulse_timesteps(timeline: CompiledTimeline) -> tuple[int, ...]:
    """Return boundaries containing at least one global pulse."""
    return tuple(
        timestep
        for timestep in range(timeline.makespan + 1)
        if any(isinstance(action, GlobalPulse) for action in timeline.action_at(timestep) or ())
    )


def frame_operation_for_action(action: Action) -> PauliFrameOperation | None:
    """Return the frame operation induced by a global pulse, if applicable."""
    if not isinstance(action, GlobalPulse):
        return None
    return frame_operation_for_gate_spec(action.gate)


def frame_operation_for_gate_spec(spec: GateSpec) -> PauliFrameOperation:
    """Translate an odd-pi single-axis rotation into a Pauli operation.

    Returns:
        The corresponding Pauli-frame operation.

    Raises:
        ValueError: If the gate is not a supported odd-pi rotation.
    """
    if not _is_odd_pi_rotation(spec.theta):
        msg = "frame-tracked global pulses must be odd pi rotations"
        raise ValueError(msg)
    gate_to_label = {"Rx": "X", "Ry": "Y", "Rz": "Z"}
    try:
        return PauliFrameOperation(gate_to_label[spec.gate_name])
    except KeyError as error:
        msg = f"frame tracking does not support globally applied gate: {spec.gate_name!r}"
        raise ValueError(msg) from error


def effective_gate_spec(spec: GateSpec, frame: PauliFrame) -> GateSpec:
    """Return a gate specification transformed through a Pauli frame."""
    axis = _gate_axis(spec.gate_name)
    if axis is None or spec.theta is None:
        return spec
    return GateSpec(gate_name=spec.gate_name, theta=frame.phase_sign(axis) * spec.theta)


def effective_action(action: Action, frame: PauliFrame) -> Action:
    """Return a single-ion rotation transformed through a Pauli frame."""
    if isinstance(action, Rx):
        theta = cast("float", effective_gate_spec(GateSpec("Rx", action.theta), frame).theta)
        return Rx(ion=action.ion, theta=theta, duration=action.duration, virtual=action.virtual)
    if isinstance(action, Ry):
        theta = cast("float", effective_gate_spec(GateSpec("Ry", action.theta), frame).theta)
        return Ry(ion=action.ion, theta=theta, duration=action.duration, virtual=action.virtual)
    if isinstance(action, Rz):
        theta = cast("float", effective_gate_spec(GateSpec("Rz", action.theta), frame).theta)
        return Rz(ion=action.ion, theta=theta, duration=action.duration, virtual=action.virtual)
    return action


def accumulated_frame_phase(
    timeline: CompiledTimeline,
    ion: int,
    t_start: int,
    t_end: int,
    field_profile: FieldProfile | None,
    frame_history: FrameHistory | None = None,
    local_pulse_action_ids: frozenset[int] = frozenset(),
) -> float:
    """Return signed field exposure after applying tracked Pauli frames.

    Raises:
        ValueError: If the requested interval lies outside the schedule.
    """
    if field_profile is None:
        return 0.0
    if not 0 <= t_start <= t_end <= timeline.makespan:
        msg = f"expected 0 <= t_start <= t_end <= {timeline.makespan}"
        raise ValueError(msg)
    history = build_frame_history(timeline, local_pulse_action_ids) if frame_history is None else frame_history
    return sum(
        history.phase_sign_for_ion(ion, timestep, axis="Z")
        * field_profile.field_at(timeline.ion_position(ion, timestep))
        for timestep in range(t_start, t_end)
    )


def _require_pauli_label(label: str) -> str:
    if label not in _PAULI_LABELS:
        available = ", ".join(sorted(_PAULI_LABELS))
        msg = f"expected Pauli label in {{{available}}}, got {label!r}"
        raise ValueError(msg)
    return label


def _framed_action_kind(
    scheduled_action: ScheduledAction,
    local_pulse_action_ids: frozenset[int],
) -> FrameActionKind:
    action = scheduled_action.action
    if isinstance(action, GlobalPulse):
        return "global_dd_pulse"
    if isinstance(action, TransportAction):
        return "transport"
    if scheduled_action.action_id in local_pulse_action_ids:
        return "local_dd_pulse"
    if isinstance(action, GateAction):
        return "algorithmic_gate"
    if isinstance(action, AdvanceTime):
        return "advance_time"
    return "other"


def _event_ion_frames(
    action: Action,
    timestep: int,
    timeline: CompiledTimeline,
    frame_history: FrameHistory,
) -> tuple[tuple[int, PauliFrame], ...]:
    if isinstance(action, (Rx, Ry, Rz)):
        return ((action.ion, frame_history.frame_for_ion(action.ion, timestep)),)
    if isinstance(action, (Rxx, Ryy, Rzz)):
        return (
            (action.ion_a, frame_history.frame_for_ion(action.ion_a, timestep)),
            (action.ion_b, frame_history.frame_for_ion(action.ion_b, timestep)),
        )
    if isinstance(action, Shuttle):
        return ((action.ion, frame_history.frame_for_ion(action.ion, timestep)),)
    if isinstance(action, PhysicalSwap):
        return (
            (action.ion_a, frame_history.frame_for_ion(action.ion_a, timestep)),
            (action.ion_b, frame_history.frame_for_ion(action.ion_b, timestep)),
        )
    if isinstance(action, GlobalPulse):
        return tuple(
            (ion, frame_history.frame_for_ion(ion, timestep)) for ion, _site in timeline.state_at(timestep).positions
        )
    return ()


def _gate_axis(gate_name: str) -> str | None:
    return {"Rx": "X", "Ry": "Y", "Rz": "Z"}.get(gate_name)


def _is_odd_pi_rotation(theta: float | None) -> bool:
    if theta is None:
        return False
    ratio = theta / pi
    nearest_integer = round(ratio)
    return nearest_integer % 2 != 0 and isclose(ratio, nearest_integer, abs_tol=1e-9)


def _build_local_frame_overrides(
    timeline: CompiledTimeline,
    local_pulse_action_ids: frozenset[int],
) -> dict[int, tuple[PauliFrame, ...]]:
    operations_by_ion: dict[int, dict[int, list[PauliFrameOperation]]] = {}
    for timestep in range(timeline.makespan + 1):
        for item in timeline.scheduled_action_at(timestep) or ():
            action = item.action
            if item.action_id not in local_pulse_action_ids or not isinstance(action, (Rx, Ry, Rz)):
                continue
            operations_by_ion.setdefault(action.ion, {}).setdefault(timestep, []).append(
                _frame_operation_for_local_action(action)
            )

    overrides: dict[int, tuple[PauliFrame, ...]] = {}
    for ion, operations_by_time in operations_by_ion.items():
        current_frame = PauliFrame()
        frames_for_ion: list[PauliFrame] = []
        for timestep in range(timeline.makespan + 1):
            for operation in operations_by_time.get(timestep, ()):
                current_frame = current_frame.compose(operation)
            frames_for_ion.append(current_frame)
        overrides[ion] = tuple(frames_for_ion)
    return overrides


def _frame_operation_for_local_action(action: Action) -> PauliFrameOperation:
    if isinstance(action, (Rx, Ry, Rz)):
        return frame_operation_for_gate_spec(GateSpec(type(action).__name__, theta=action.theta))
    msg = f"unsupported local DD gate action for frame tracking: {action!r}"
    raise ValueError(msg)


__all__ = [
    "FrameHistory",
    "FramedActionEvent",
    "PauliFrame",
    "PauliFrameOperation",
    "accumulated_frame_phase",
    "build_frame_history",
    "effective_action",
    "effective_gate_spec",
    "frame_operation_for_action",
    "frame_operation_for_gate_spec",
    "framed_action_events",
    "global_pulse_timesteps",
]
