# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Actions supported by the Linear hardware model and compiler."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field, fields, replace
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.state import State


@dataclass(frozen=True)
class Action(ABC):
    """An operation that the Linear compiler can place in a schedule.

    Hardware models can introduce new operations by subclassing ``Action`` and
    defining when they are available and how they change the state. Decisions
    involving several operations or circuit dependencies remain with the
    compiler.
    """

    individually_unconstrained: ClassVar[bool] = True

    def is_valid(self, state: State, architecture: Architecture) -> bool:
        """Return whether this action can start in the current state.

        Actions without their own restrictions are available by default. This
        check considers one action at a time; the compiler separately checks
        interactions between simultaneous actions.
        """
        del state, architecture
        return self.individually_unconstrained

    @abstractmethod
    def apply(self, state: State, architecture: Architecture) -> State:
        """Return the state produced when this action starts.

        Physical actions leave circuit-progress tracking unchanged. Callers
        should check :meth:`is_valid` before applying an action.
        """

    def to_dict(self) -> dict[str, object]:
        """Return a description of this action using JSON-compatible values."""
        result: dict[str, object] = {"type": type(self).__name__}
        for model_field in fields(self):
            value = getattr(self, model_field.name)
            if model_field.name == "virtual" and model_field.default is not MISSING and value == model_field.default:
                continue
            result[model_field.name] = value
        return result


@dataclass(frozen=True)
class SchedulableAction(Action):
    """An action that may appear in a compiled schedule."""

    def __post_init__(self) -> None:
        """Validate an action's duration when it defines one.

        Raises:
            ValueError: If the duration is not a positive integer.
        """
        duration = getattr(self, "duration", None)
        if duration is None:
            return
        if isinstance(duration, bool) or not isinstance(duration, int):
            msg = "action duration must be an integer"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Preserve source behavior.
        if duration < 1:
            msg = "action duration must be >= 1"
            raise ValueError(msg)

    def _duration(self) -> int:
        """Return the validated duration of an action that defines one.

        Raises:
            TypeError: If a subclass does not define an integer duration.
        """
        duration = vars(self).get("duration")
        if isinstance(duration, bool) or not isinstance(duration, int):
            msg = "action does not define a validated integer duration"
            raise TypeError(msg)
        return duration


@dataclass(frozen=True)
class PhysicalAction(SchedulableAction):
    """A control operation exposed by the hardware model.

    Physical actions update ion placement or hardware availability while
    leaving the compiler's circuit-progress tracking unchanged.
    """


@dataclass(frozen=True)
class TransportAction(PhysicalAction):
    """A hardware operation that changes where ions are located."""


@dataclass(frozen=True)
class Shuttle(TransportAction):
    """Move one ion between adjacent sites."""

    ion: int
    src: int
    dst: int
    duration: int = 1

    def is_valid(self, state: State, architecture: Architecture) -> bool:
        """Return whether the ion can move into an adjacent empty site."""
        positions = dict(state.positions)
        return (
            positions.get(self.ion) == self.src
            and 0 <= self.dst < architecture.num_sites
            and is_adjacent(self.src, self.dst)
            and self.dst not in positions.values()
            and dict(state.ions_busy_until).get(self.ion, state.time + 1) <= state.time
        )

    def apply(self, state: State, architecture: Architecture) -> State:
        """Move the ion and reserve it for the configured duration.

        Returns:
            The updated state with compiler progress preserved.
        """
        del architecture
        positions = dict(state.positions)
        ions_busy = dict(state.ions_busy_until)
        positions[self.ion] = self.dst
        ions_busy[self.ion] = state.time + self.duration
        return replace(
            state,
            positions=tuple(sorted(positions.items())),
            ions_busy_until=tuple(sorted(ions_busy.items())),
        )


@dataclass(frozen=True)
class PhysicalSwap(TransportAction):
    """Exchange two ions occupying adjacent sites."""

    ion_a: int
    ion_b: int
    pos_a: int
    pos_b: int
    duration: int = 1

    def is_valid(self, state: State, architecture: Architecture) -> bool:
        """Return whether two free ions occupy the specified adjacent sites."""
        del architecture
        positions = dict(state.positions)
        ions_busy = dict(state.ions_busy_until)
        return (
            self.ion_a != self.ion_b
            and positions.get(self.ion_a) == self.pos_a
            and positions.get(self.ion_b) == self.pos_b
            and ions_busy.get(self.ion_a, state.time + 1) <= state.time
            and ions_busy.get(self.ion_b, state.time + 1) <= state.time
            and is_adjacent(self.pos_a, self.pos_b)
        )

    def apply(self, state: State, architecture: Architecture) -> State:
        """Exchange both positions and reserve both ions.

        Returns:
            The updated state with compiler progress preserved.
        """
        del architecture
        positions = dict(state.positions)
        ions_busy = dict(state.ions_busy_until)
        positions[self.ion_a], positions[self.ion_b] = (
            positions[self.ion_b],
            positions[self.ion_a],
        )
        free_time = state.time + self.duration
        ions_busy[self.ion_a] = free_time
        ions_busy[self.ion_b] = free_time
        return replace(
            state,
            positions=tuple(sorted(positions.items())),
            ions_busy_until=tuple(sorted(ions_busy.items())),
        )


@dataclass(frozen=True)
class GateAction(PhysicalAction):
    """A gate operation supported by the hardware model."""


@dataclass(frozen=True)
class GateSpec:
    """Describe the rotation performed by a global pulse."""

    gate_name: str
    theta: float | None = None


@dataclass(frozen=True)
class SingleQubitGate(GateAction):
    """A rotation acting on one ion, implemented physically or virtually.

    A virtual rotation changes the logical frame without occupying hardware
    time or a processing zone, but it remains part of the compiled circuit. A
    physical rotation uses the ion and its processing zone, even when a hardware
    abstraction assigns it zero duration.
    """

    ion: int
    virtual: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Validate virtuality and duration consistency.

        Raises:
            TypeError: If ``virtual`` is not Boolean.
            ValueError: If the duration is invalid or a virtual gate has nonzero duration.

        Warns:
            UserWarning: If a physical gate has zero duration.
        """
        if not isinstance(self.virtual, bool):
            msg = "virtual must be a boolean"
            raise TypeError(msg)

        duration = getattr(self, "duration", None)
        if duration is None:
            return
        if isinstance(duration, bool) or not isinstance(duration, int):
            msg = "action duration must be an integer"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Preserve source behavior.
        if self.virtual:
            if duration != 0:
                msg = "virtual single-qubit gate duration must be 0"
                raise ValueError(msg)
            return
        if duration == 0:
            warnings.warn(
                "physical single-qubit gate has zero duration and may share a compiler timestep. "
                "This is fine if the intended hardware abstraction is gate-duration << timestep.",
                stacklevel=2,
            )
            return
        super().__post_init__()

    def is_valid(self, state: State, architecture: Architecture) -> bool:
        """Return whether the target ion and physical resources are eligible."""
        positions = dict(state.positions)
        if self.virtual:
            return self.ion in positions
        position = positions.get(self.ion)
        if position is None or dict(state.ions_busy_until).get(self.ion, state.time + 1) > state.time:
            return False
        zone = architecture.get_processing_zone(position)
        return zone is not None and dict(state.pzs_busy_until).get(zone, state.time + 1) <= state.time

    def apply(self, state: State, architecture: Architecture) -> State:
        """Reserve physical resources, or leave machine state unchanged if virtual.

        Returns:
            The updated state with compiler progress preserved.
        """
        if self.virtual:
            return state
        positions = dict(state.positions)
        ions_busy = dict(state.ions_busy_until)
        pzs_busy = dict(state.pzs_busy_until)
        free_time = state.time + self._duration()
        ions_busy[self.ion] = free_time
        zone = architecture.get_processing_zone(positions[self.ion])
        if zone is not None:
            pzs_busy[zone] = free_time
        return replace(
            state,
            ions_busy_until=tuple(sorted(ions_busy.items())),
            pzs_busy_until=tuple(sorted(pzs_busy.items())),
        )


@dataclass(frozen=True)
class TwoQubitGate(GateAction):
    """A rotation acting on two ions in the same processing zone."""

    ion_a: int
    ion_b: int

    def is_valid(self, state: State, architecture: Architecture) -> bool:
        """Return whether both free ions occupy one free processing zone."""
        positions = dict(state.positions)
        pos_a = positions.get(self.ion_a)
        pos_b = positions.get(self.ion_b)
        ions_busy = dict(state.ions_busy_until)
        if (
            pos_a is None
            or pos_b is None
            or self.ion_a == self.ion_b
            or ions_busy.get(self.ion_a, state.time + 1) > state.time
            or ions_busy.get(self.ion_b, state.time + 1) > state.time
        ):
            return False
        zone_a = architecture.get_processing_zone(pos_a)
        zone_b = architecture.get_processing_zone(pos_b)
        return (
            zone_a is not None
            and zone_a == zone_b
            and dict(state.pzs_busy_until).get(zone_a, state.time + 1) <= state.time
        )

    def apply(self, state: State, architecture: Architecture) -> State:
        """Reserve both ions and their shared processing zone.

        Returns:
            The updated state with compiler progress preserved.
        """
        positions = dict(state.positions)
        ions_busy = dict(state.ions_busy_until)
        pzs_busy = dict(state.pzs_busy_until)
        free_time = state.time + self._duration()
        ions_busy[self.ion_a] = free_time
        ions_busy[self.ion_b] = free_time
        zone = architecture.get_processing_zone(positions[self.ion_a])
        if zone is not None:
            pzs_busy[zone] = free_time
        return replace(
            state,
            ions_busy_until=tuple(sorted(ions_busy.items())),
            pzs_busy_until=tuple(sorted(pzs_busy.items())),
        )


@dataclass(frozen=True)
class Rx(SingleQubitGate):
    """Rotate one ion around the x axis."""

    theta: float
    duration: int = 1


@dataclass(frozen=True)
class Ry(SingleQubitGate):
    """Rotate one ion around the y axis."""

    theta: float
    duration: int = 1


@dataclass(frozen=True)
class Rz(SingleQubitGate):
    """Rotate one ion around the z axis.

    The rotation is virtual by default to align with typical trapped-ion
    hardware implementations. Set ``virtual=False`` to model a physical
    implementation subject to ordinary scheduling-resource checks.
    """

    theta: float
    duration: int = 0
    virtual: bool = field(default=True, kw_only=True)


@dataclass(frozen=True)
class Rxx(TwoQubitGate):
    """Rotate two ions around the xx axis."""

    theta: float
    duration: int = 1


@dataclass(frozen=True)
class Ryy(TwoQubitGate):
    """Rotate two ions around the yy axis."""

    theta: float
    duration: int = 1


@dataclass(frozen=True)
class Rzz(TwoQubitGate):
    """Rotate two ions around the zz axis."""

    theta: float
    duration: int = 1


@dataclass(frozen=True)
class GlobalPulse(GateAction):
    """Apply one gate to ions simultaneously, as in coil-based microwave control.

    The default Linear model treats this pulse as non-blocking w.r.t to other
    gates. A hardware model with shared-drive conflicts can subclass it and
    provide stricter validity and state-update behavior.
    """

    gate: GateSpec
    duration: int = 1

    def apply(self, state: State, architecture: Architecture) -> State:
        """Return the state with hardware availability unchanged."""
        del architecture
        assert self.duration >= 1
        return state

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of the pulse and rotation."""
        return {
            "type": type(self).__name__,
            "gate_name": self.gate.gate_name,
            "theta": self.gate.theta,
            "duration": self.duration,
        }


@dataclass(frozen=True)
class AdvanceTime(SchedulableAction):
    """Move the schedule forward by one timestep.

    This is a compiler operation advancing the clock state and has no real
    hardware equivalent. It allows running operations to finish before
    further actions are scheduled.
    """

    timestep_increment: ClassVar[int] = 1

    def apply(self, state: State, architecture: Architecture) -> State:
        """Advance time and finish operations due by the next timestep.

        Returns:
            The state at the next compiler timestep.
        """
        del architecture
        next_time = state.time + self.timestep_increment
        completed_gates = set(state.completed_gates)
        remaining_in_progress: list[tuple[int, int]] = []
        for gate_id, finish_time in state.in_progress_gates:
            if finish_time <= next_time:
                completed_gates.add(gate_id)
            else:
                remaining_in_progress.append((gate_id, finish_time))
        return replace(
            state,
            completed_gates=frozenset(completed_gates),
            in_progress_gates=tuple(remaining_in_progress),
            time=next_time,
        )


def is_adjacent(pos_a: int, pos_b: int) -> bool:
    """Return whether two Linear site indices are adjacent."""
    return abs(pos_a - pos_b) == 1


__all__ = [
    "Action",
    "AdvanceTime",
    "GateAction",
    "GateSpec",
    "GlobalPulse",
    "PhysicalAction",
    "PhysicalSwap",
    "Rx",
    "Rxx",
    "Ry",
    "Ryy",
    "Rz",
    "Rzz",
    "SchedulableAction",
    "Shuttle",
    "SingleQubitGate",
    "TransportAction",
    "TwoQubitGate",
]
