# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Actions supported by the Linear hardware model and compiler.

Malformed serialized action values raise ``ValueError`` because they represent
invalid saved data rather than incorrect Python API argument types.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.config import GateTiming, TransportTiming
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

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore an action from its serialized dataclass fields.

        Actions containing nested values can override this method.

        Returns:
            The restored action.

        Raises:
            ValueError: If the saved fields cannot construct the action.
        """
        values = {model_field.name: data[model_field.name] for model_field in fields(cls) if model_field.name in data}
        constructor = cast("Any", cls)
        try:
            return cast("Action", constructor(**values))
        except TypeError as error:
            msg = f"invalid serialized {cls.__name__} action"
            raise ValueError(msg) from error

    @classmethod
    def available_actions(
        cls,
        state: State,
        architecture: Architecture,
        transport_timing: TransportTiming,
    ) -> Iterable[Action]:
        """Return instances this action type makes available in a given state.

        Action types that are not generated directly by the compiler return no
        instances by default.
        """
        del cls, state, architecture, transport_timing
        return ()

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
            TypeError: If the duration is not an integer.
            ValueError: If the duration is not a positive integer.
        """
        duration = getattr(self, "duration", None)
        if duration is None:
            return
        if isinstance(duration, bool) or not isinstance(duration, int):
            msg = "action duration must be an integer"
            raise TypeError(msg)
        if duration < 1:
            msg = "action duration must be >= 1"
            raise ValueError(msg)

    def _duration(self) -> int:
        """Return the validated duration of an action that defines one.

        Raises:
            TypeError: If a subclass does not define an integer duration.
        """
        duration = getattr(self, "duration", None)
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

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a shuttle from serialized fields.

        Returns:
            The restored shuttle.
        """
        return cls(
            ion=_require_int(data, "ion"),
            src=_require_int(data, "src"),
            dst=_require_int(data, "dst"),
            duration=_require_duration(data, default=1),
        )

    @classmethod
    def available_actions(
        cls,
        state: State,
        architecture: Architecture,
        transport_timing: TransportTiming,
    ) -> Iterable[Action]:
        """Return shuttles to adjacent empty sites for every free ion."""
        occupied = {position for _, position in state.positions}
        busy_ions = {ion for ion, free_time in state.ions_busy_until if free_time > state.time}
        return (
            cls(
                ion=ion,
                src=position,
                dst=destination,
                duration=transport_timing.shuttle,
            )
            for ion, position in state.positions
            if ion not in busy_ions
            for destination in (position - 1, position + 1)
            if 0 <= destination < architecture.num_sites and destination not in occupied
        )

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

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a physical swap from serialized fields.

        Returns:
            The restored swap.
        """
        return cls(
            ion_a=_require_int(data, "ion_a"),
            ion_b=_require_int(data, "ion_b"),
            pos_a=_require_int(data, "pos_a"),
            pos_b=_require_int(data, "pos_b"),
            duration=_require_duration(data, default=1),
        )

    @classmethod
    def available_actions(
        cls,
        state: State,
        architecture: Architecture,
        transport_timing: TransportTiming,
    ) -> Iterable[Action]:
        """Return adjacent swaps between pairs of free ions."""
        del architecture
        busy_ions = {ion for ion, free_time in state.ions_busy_until if free_time > state.time}
        free_ions = [(ion, position) for ion, position in state.positions if ion not in busy_ions]
        return (
            cls(
                ion_a=ion_a,
                ion_b=ion_b,
                pos_a=pos_a,
                pos_b=pos_b,
                duration=transport_timing.swap,
            )
            for index, (ion_a, pos_a) in enumerate(free_ions)
            for ion_b, pos_b in free_ions[index + 1 :]
            if is_adjacent(pos_a, pos_b)
        )

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

    circuit_name: ClassVar[str | None] = None
    parameter_names: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_instruction(
        cls,
        ions: tuple[int, ...],
        parameters: tuple[float, ...],
        timing: GateTiming,
    ) -> GateAction:
        """Create a gate from frontend-independent circuit operands.

        Raises:
            ValueError: If the gate type does not support circuit lowering.
        """
        del ions, parameters, timing
        msg = f"gate type {cls.__name__} does not define circuit lowering"
        raise ValueError(msg)


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
    parameter_names: ClassVar[tuple[str, ...]] = ("theta",)

    @classmethod
    def from_instruction(
        cls,
        ions: tuple[int, ...],
        parameters: tuple[float, ...],
        timing: GateTiming,
    ) -> GateAction:
        """Create a single-ion rotation from normalized circuit operands.

        Returns:
            The lowered gate.
        """
        _require_instruction_shape(cls, ions, parameters, num_ions=1)
        name = _require_circuit_name(cls)
        values: dict[str, object] = {
            "ion": ions[0],
            **dict(zip(cls.parameter_names, parameters, strict=True)),
        }
        if name in timing.gate_durations:
            values["duration"] = timing.duration_for(name)
            values["virtual"] = timing.is_virtual(name)
        return _construct_gate_action(cls, values)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a single-ion gate from serialized fields.

        Returns:
            The restored gate.
        """
        default_duration = cast("int", _field_default(cls, "duration"))
        default_virtual = cast("bool", _field_default(cls, "virtual"))
        values: dict[str, object] = {
            "ion": _require_int(data, "ion"),
            "duration": _require_duration(data, default=default_duration, allow_zero=True),
            "virtual": _require_bool(data, "virtual", default=default_virtual),
        }
        values.update({name: _require_number(data, name) for name in cls.parameter_names})
        return _construct_action(cls, values)

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
        is_zero_duration = isinstance(duration, int) and not isinstance(duration, bool) and duration == 0
        if is_zero_duration and self.virtual:
            return
        if is_zero_duration:
            warnings.warn(
                "physical single-qubit gate has zero duration and may share a compiler timestep. "
                "This is fine if the intended hardware abstraction is gate-duration << timestep.",
                stacklevel=2,
            )
            return
        super().__post_init__()
        if self.virtual:
            msg = "virtual single-qubit gate duration must be 0"
            raise ValueError(msg)

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
    """A gate acting on two ions in the same processing zone."""

    ion_a: int
    ion_b: int

    @classmethod
    def from_instruction(
        cls,
        ions: tuple[int, ...],
        parameters: tuple[float, ...],
        timing: GateTiming,
    ) -> GateAction:
        """Create a two-ion gate from normalized circuit operands.

        Returns:
            The lowered gate.
        """
        _require_instruction_shape(cls, ions, parameters, num_ions=2)
        name = _require_circuit_name(cls)
        values: dict[str, object] = {
            "ion_a": ions[0],
            "ion_b": ions[1],
            **dict(zip(cls.parameter_names, parameters, strict=True)),
        }
        if name in timing.gate_durations:
            values["duration"] = timing.duration_for(name)
        return _construct_gate_action(cls, values)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a two-ion gate from serialized fields.

        Returns:
            The restored gate.
        """
        values: dict[str, object] = {
            "ion_a": _require_int(data, "ion_a"),
            "ion_b": _require_int(data, "ion_b"),
            "duration": _require_duration(
                data,
                default=cast("int", _field_default(cls, "duration")),
            ),
        }
        values.update({name: _require_number(data, name) for name in cls.parameter_names})
        return _construct_action(cls, values)

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
    circuit_name: ClassVar[str] = "rx"


@dataclass(frozen=True)
class Ry(SingleQubitGate):
    """Rotate one ion around the y axis."""

    theta: float
    duration: int = 1
    circuit_name: ClassVar[str] = "ry"


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
    circuit_name: ClassVar[str] = "rz"


@dataclass(frozen=True)
class Rxx(TwoQubitGate):
    """Rotate two ions around the xx axis."""

    theta: float
    duration: int = 2
    circuit_name: ClassVar[str] = "rxx"
    parameter_names: ClassVar[tuple[str, ...]] = ("theta",)


@dataclass(frozen=True)
class Ryy(TwoQubitGate):
    """Rotate two ions around the yy axis."""

    theta: float
    duration: int = 2
    circuit_name: ClassVar[str] = "ryy"
    parameter_names: ClassVar[tuple[str, ...]] = ("theta",)


@dataclass(frozen=True)
class Rzz(TwoQubitGate):
    """Rotate two ions around the zz axis."""

    theta: float
    duration: int = 2
    circuit_name: ClassVar[str] = "rzz"
    parameter_names: ClassVar[tuple[str, ...]] = ("theta",)


@dataclass(frozen=True)
class GlobalPulse(GateAction):
    """Apply one gate to ions simultaneously, as in coil-based microwave control.

    The default Linear model treats this pulse as non-blocking w.r.t to other
    gates. A hardware model with shared-drive conflicts can subclass it and
    provide stricter validity and state-update behavior.
    """

    gate: GateSpec
    duration: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a global pulse and its nested gate description.

        Returns:
            The restored pulse.
        """
        return cls(
            gate=GateSpec(
                gate_name=_require_str(data, "gate_name"),
                theta=_require_optional_number(data, "theta"),
            ),
            duration=_require_duration(data, default=1),
        )

    def apply(self, state: State, architecture: Architecture) -> State:
        """Return the state with hardware availability unchanged."""
        del self, architecture
        return state

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of the pulse and rotation."""
        return {
            "type": type(self).__name__,
            "gate_name": self.gate.gate_name,
            "theta": self.gate.theta,
            "duration": self.duration,
        }


DEFAULT_ACTION_TYPES: tuple[type[Action], ...] = (
    PhysicalSwap,
    Shuttle,
    Rx,
    Ry,
    Rz,
    Rzz,
)

BUILTIN_ACTION_TYPES: tuple[type[Action], ...] = (
    PhysicalSwap,
    Shuttle,
    Rx,
    Ry,
    Rz,
    Rxx,
    Ryy,
    Rzz,
    GlobalPulse,
)


@dataclass(frozen=True)
class AdvanceTime(SchedulableAction):
    """Move the schedule forward by one timestep.

    This is a compiler operation advancing the clock state and has no real
    hardware equivalent. It allows running operations to finish before
    further actions are scheduled.
    """

    timestep_increment: ClassVar[int] = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore the scheduler's AdvanceTime marker.

        Returns:
            The restored marker.
        """
        _require_duration(data, default=1)
        return cls()

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


def _construct_action(action_type: type[Action], values: Mapping[str, object]) -> Action:
    """Construct an action from validated field values.

    Args:
        action_type: Action class to instantiate.
        values: Constructor arguments keyed by field name.

    Returns:
        The constructed action.
    """
    constructor = cast("Any", action_type)
    return cast("Action", constructor(**values))


def _construct_gate_action(
    action_type: type[GateAction],
    values: Mapping[str, object],
) -> GateAction:
    """Construct a gate action from validated field values.

    Args:
        action_type: Gate class to instantiate.
        values: Constructor arguments keyed by field name.

    Returns:
        The constructed gate action.
    """
    return cast("GateAction", _construct_action(action_type, values))


def _require_circuit_name(gate_type: type[GateAction]) -> str:
    """Return the circuit name declared by a gate class.

    Args:
        gate_type: Gate class to inspect.

    Returns:
        The declared circuit name.

    Raises:
        ValueError: If the gate class has no circuit name.
    """
    name = gate_type.circuit_name
    if name is None:
        msg = f"gate type {gate_type.__name__} does not define a circuit name"
        raise ValueError(msg)
    return name


def _require_instruction_shape(
    gate_type: type[GateAction],
    ions: tuple[int, ...],
    parameters: tuple[float, ...],
    *,
    num_ions: int,
) -> None:
    """Validate the operand and parameter counts for a circuit instruction.

    Args:
        gate_type: Gate class receiving the instruction.
        ions: Ion identifiers supplied by the instruction.
        parameters: Numeric parameters supplied by the instruction.
        num_ions: Required number of ion operands.

    Raises:
        ValueError: If the gate lacks a circuit name or the counts do not match.
    """
    if len(ions) != num_ions or len(parameters) != len(gate_type.parameter_names):
        msg = (
            f"operation {_require_circuit_name(gate_type)!r} requires {num_ions} ions "
            f"and {len(gate_type.parameter_names)} parameters"
        )
        raise ValueError(msg)


def _field_default(action_type: type[Action], name: str) -> object:
    """Return a declared default value from an action dataclass.

    Args:
        action_type: Action class containing the field.
        name: Field whose default is required.

    Returns:
        The field's default value.

    Raises:
        ValueError: If the field is absent or has no default.
    """
    model_field = next((item for item in fields(action_type) if item.name == name), None)
    if model_field is None or model_field.default is MISSING:
        msg = f"{action_type.__name__} must define a default {name}"
        raise ValueError(msg)
    return model_field.default


def _require_int(data: Mapping[str, object], key: str) -> int:
    """Read an integer from serialized action data.

    Args:
        data: Serialized action fields.
        key: Field to read.

    Returns:
        The integer value.

    Raises:
        ValueError: If the field is absent or is not an integer.
    """
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    return value


def _require_number(data: Mapping[str, object], key: str) -> float:
    """Read a numeric value from serialized action data.

    Args:
        data: Serialized action fields.
        key: Field to read.

    Returns:
        The value converted to a float.

    Raises:
        ValueError: If the field is absent or is not numeric.
    """
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key} must be numeric"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    return float(value)


def _require_optional_number(data: Mapping[str, object], key: str) -> float | None:
    """Read an optional numeric value from serialized action data.

    Args:
        data: Serialized action fields.
        key: Field to read.

    Returns:
        The value converted to a float, or ``None``.

    Raises:
        ValueError: If a non-null value is not numeric.
    """
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key} must be numeric or null"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    return float(value)


def _require_str(data: Mapping[str, object], key: str) -> str:
    """Read a string from serialized action data.

    Args:
        data: Serialized action fields.
        key: Field to read.

    Returns:
        The string value.

    Raises:
        ValueError: If the field is absent or is not a string.
    """
    value = data.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    return value


def _require_bool(data: Mapping[str, object], key: str, *, default: bool) -> bool:
    """Read a Boolean from serialized action data.

    Args:
        data: Serialized action fields.
        key: Field to read.
        default: Value used when the field is absent.

    Returns:
        The Boolean value.

    Raises:
        ValueError: If the value is not Boolean.
    """
    value = data.get(key, default)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    return value


def _require_duration(
    data: Mapping[str, object],
    *,
    default: int,
    allow_zero: bool = False,
) -> int:
    """Read and validate an action duration.

    Args:
        data: Serialized action fields.
        default: Duration used when the field is absent.
        allow_zero: Whether zero is an accepted duration.

    Returns:
        The validated duration.

    Raises:
        ValueError: If the duration is not an integer or is below the minimum.
    """
    value = data.get("duration", default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "duration must be an integer"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    minimum = 0 if allow_zero else 1
    if value < minimum:
        msg = f"action duration must be >= {minimum}"
        raise ValueError(msg)
    return value


__all__ = [
    "BUILTIN_ACTION_TYPES",
    "DEFAULT_ACTION_TYPES",
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
