# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared gate occurrences and gate timing used by all architecture levels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast

from mqt.ionshuttler.core.actions import Action, index_action_types

from .._json_utils import require_int, require_list, require_number, require_str

if TYPE_CHECKING:
    from collections.abc import Mapping

SINGLE_QUBIT_GATE_NAMES = frozenset({"rx", "ry", "rz"})
TWO_QUBIT_GATE_NAMES = frozenset({"rxx", "ryy", "rzz"})
GATE_NAMES = SINGLE_QUBIT_GATE_NAMES | TWO_QUBIT_GATE_NAMES


@dataclass(frozen=True)
class GateTiming:
    """Configure integer gate durations and virtual single-ion rotations.

    Every architecture level uses the same gate durations. Durations of
    transport and other non-gate operations belong to the architecture that
    defines those operations. By default, Rz gates are virtual and
    instantaneous and two-qubit gates have a longer duration than
    single-qubit gates.
    """

    rx: int = 1
    ry: int = 1
    rz: int = 0
    rxx: int = 2
    ryy: int = 2
    rzz: int = 2
    virtual_single_qubit_gates: frozenset[str] = field(default_factory=lambda: frozenset({"rz"}))

    def __post_init__(self) -> None:
        """Ensure the timing describes valid gate implementations.

        Raises:
            ValueError: If a duration or virtual gate declaration is invalid.
        """
        virtual_gates = _normalize_virtual_gates(self.virtual_single_qubit_gates)
        object.__setattr__(self, "virtual_single_qubit_gates", virtual_gates)
        for gate_name in SINGLE_QUBIT_GATE_NAMES:
            duration = self.duration_for(gate_name)
            _require_integer_at_least(duration, f"{gate_name} duration", minimum=0)
            if gate_name in virtual_gates and duration != 0:
                msg = f"duration for virtual gate {gate_name!r} must be 0"
                raise ValueError(msg)
        for gate_name in TWO_QUBIT_GATE_NAMES:
            _require_integer_at_least(self.duration_for(gate_name), f"{gate_name} duration", minimum=1)

    def duration_for(self, gate_name: str) -> int:
        """Return the duration of one supported gate.

        Returns:
            The integer duration.

        Raises:
            TypeError: If a stored duration has the wrong type.
            ValueError: If the gate name is unsupported.
        """
        normalized_name = gate_name.lower()
        if normalized_name not in GATE_NAMES:
            msg = f"unsupported gate name {gate_name!r}"
            raise ValueError(msg)
        value = getattr(self, normalized_name)
        if not isinstance(value, int):
            msg = f"duration for gate {gate_name!r} must be an integer"
            raise TypeError(msg)
        return value

    def is_virtual(self, gate_name: str) -> bool:
        """Return whether a single-ion rotation uses a virtual implementation.

        Returns:
            Whether the implementation is virtual.

        Raises:
            ValueError: If the name is not a single-ion gate.
        """
        normalized_name = gate_name.lower()
        if normalized_name not in SINGLE_QUBIT_GATE_NAMES:
            msg = f"virtuality applies only to single-qubit gates, not {gate_name!r}"
            raise ValueError(msg)
        return normalized_name in self.virtual_single_qubit_gates

    @property
    def gate_durations(self) -> dict[str, int]:
        """Duration of each supported gate, keyed by lowercase name."""
        return {gate_name: self.duration_for(gate_name) for gate_name in sorted(GATE_NAMES)}

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible gate timing."""
        return {
            **self.gate_durations,
            "virtual_single_qubit_gates": sorted(self.virtual_single_qubit_gates),
        }

    @classmethod
    def from_dict(cls, data: object) -> GateTiming:
        """Restore gate timing from serialized architecture data.

        Missing fields use their default values.

        Returns:
            The restored gate timing.

        Raises:
            TypeError: If the data or a duration has the wrong type.
            ValueError: If the data contains an unknown field.
        """
        if not isinstance(data, dict):
            msg = "gate_timing must be a JSON object"
            raise TypeError(msg)
        mapping = cast("dict[str, object]", data)
        unknown_fields = sorted(set(mapping).difference({*GATE_NAMES, "virtual_single_qubit_gates"}))
        if unknown_fields:
            msg = f"unknown gate_timing fields: {', '.join(unknown_fields)}"
            raise ValueError(msg)
        defaults = cls()
        return cls(
            rx=_mapping_duration(mapping, "rx", defaults.rx),
            ry=_mapping_duration(mapping, "ry", defaults.ry),
            rz=_mapping_duration(mapping, "rz", defaults.rz),
            rxx=_mapping_duration(mapping, "rxx", defaults.rxx),
            ryy=_mapping_duration(mapping, "ryy", defaults.ryy),
            rzz=_mapping_duration(mapping, "rzz", defaults.rzz),
            virtual_single_qubit_gates=_normalize_virtual_gates(
                mapping.get("virtual_single_qubit_gates", sorted(defaults.virtual_single_qubit_gates))
            ),
        )


@dataclass(frozen=True)
class GateAction(Action):
    """One gate occurrence with stable optional circuit identity.

    A gate type defines its circuit name and parameter names. A gate value
    stores only the gate's intrinsic data: its circuit identity, target ions,
    and parameters. The executing architecture decides the gate's duration,
    resources, and state effects. Every parameter is a finite real number and
    is stored as a ``float``.

    Equality and hashing compare the gate type, ions, and parameters. They
    ignore ``gate_id``, so use ``gate_id`` to identify a circuit gate. A gate
    that the compiler inserts, such as a dynamical-decoupling pulse, has no
    ``gate_id``.
    """

    gate_id: int | None = field(default=None, kw_only=True, compare=False)
    circuit_name: ClassVar[str | None] = None
    parameter_names: ClassVar[tuple[str, ...]] = ()

    @property
    def ions(self) -> tuple[int, ...]:
        """The gate's ordered ion operands."""
        msg = "..."
        raise NotImplementedError(msg)

    def __post_init__(self) -> None:
        """Validate the optional circuit gate identifier and the parameters."""
        _validate_gate_id(self.gate_id)
        for name in self.parameter_names:
            object.__setattr__(self, name, _finite_parameter(getattr(self, name), name))

    @classmethod
    def from_instruction(
        cls,
        ions: tuple[int, ...],
        parameters: tuple[float, ...],
        *,
        gate_id: int | None = None,
    ) -> GateAction:
        """Create a gate occurrence from parsed circuit operands.

        Raises:
            ValueError: If the gate type supplies no lowering rule.
        """
        del ions, parameters, gate_id
        msg = f"gate type {cls.__name__} does not define circuit lowering"
        raise ValueError(msg)


@dataclass(frozen=True)
class SingleQubitGate(GateAction):
    """A rotation acting on one ion."""

    ion: int
    parameter_names: ClassVar[tuple[str, ...]] = ("theta",)

    def __post_init__(self) -> None:
        """Validate the target ion and the gate data shared by all gates."""
        super().__post_init__()
        _validate_ion(self.ion, "ion")

    @property
    def ions(self) -> tuple[int, ...]:
        """The gate's sole ion."""
        return (self.ion,)

    @classmethod
    def from_instruction(
        cls,
        ions: tuple[int, ...],
        parameters: tuple[float, ...],
        *,
        gate_id: int | None = None,
    ) -> GateAction:
        """Create a single-ion gate occurrence.

        Returns:
            The gate occurrence.
        """
        _require_instruction_shape(cls, ions, parameters, num_ions=1)
        values: dict[str, object] = {
            "gate_id": gate_id,
            "ion": ions[0],
            **dict(zip(cls.parameter_names, parameters, strict=True)),
        }
        return _construct_gate_action(cls, values)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a single-ion gate occurrence.

        Returns:
            The restored gate occurrence.
        """
        values: dict[str, object] = {
            "gate_id": _optional_gate_id(data),
            "ion": require_int(data, "ion"),
        }
        values.update({name: require_number(data, name) for name in cls.parameter_names})
        return _construct_gate_action(cls, values)


@dataclass(frozen=True)
class TwoQubitGate(GateAction):
    """A gate acting on two ordered ions."""

    ion_a: int
    ion_b: int

    def __post_init__(self) -> None:
        """Validate the target ions and the gate data shared by all gates.

        Raises:
            ValueError: If both target ions are the same.
        """
        super().__post_init__()
        _validate_ion(self.ion_a, "ion_a")
        _validate_ion(self.ion_b, "ion_b")
        if self.ion_a == self.ion_b:
            msg = "two-qubit gate ions must be distinct"
            raise ValueError(msg)

    @property
    def ions(self) -> tuple[int, ...]:
        """The gate's ordered ion pair."""
        return (self.ion_a, self.ion_b)

    @classmethod
    def from_instruction(
        cls,
        ions: tuple[int, ...],
        parameters: tuple[float, ...],
        *,
        gate_id: int | None = None,
    ) -> GateAction:
        """Create a two-ion gate occurrence.

        Returns:
            The gate occurrence.
        """
        _require_instruction_shape(cls, ions, parameters, num_ions=2)
        values: dict[str, object] = {
            "gate_id": gate_id,
            "ion_a": ions[0],
            "ion_b": ions[1],
            **dict(zip(cls.parameter_names, parameters, strict=True)),
        }
        return _construct_gate_action(cls, values)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a two-ion gate occurrence.

        Returns:
            The restored gate occurrence.
        """
        values: dict[str, object] = {
            "gate_id": _optional_gate_id(data),
            "ion_a": require_int(data, "ion_a"),
            "ion_b": require_int(data, "ion_b"),
        }
        values.update({name: require_number(data, name) for name in cls.parameter_names})
        return _construct_gate_action(cls, values)


@dataclass(frozen=True)
class Rx(SingleQubitGate):
    """Rotate one ion around the x axis."""

    theta: float
    circuit_name: ClassVar[str] = "rx"
    serialized_type: ClassVar[str] = "gate.rx"


@dataclass(frozen=True)
class Ry(SingleQubitGate):
    """Rotate one ion around the y axis."""

    theta: float
    circuit_name: ClassVar[str] = "ry"
    serialized_type: ClassVar[str] = "gate.ry"


@dataclass(frozen=True)
class Rz(SingleQubitGate):
    """Rotate one ion around the z axis."""

    theta: float
    circuit_name: ClassVar[str] = "rz"
    serialized_type: ClassVar[str] = "gate.rz"


@dataclass(frozen=True)
class Rxx(TwoQubitGate):
    """Rotate two ions around the xx axis."""

    theta: float
    circuit_name: ClassVar[str] = "rxx"
    serialized_type: ClassVar[str] = "gate.rxx"
    parameter_names: ClassVar[tuple[str, ...]] = ("theta",)


@dataclass(frozen=True)
class Ryy(TwoQubitGate):
    """Rotate two ions around the yy axis."""

    theta: float
    circuit_name: ClassVar[str] = "ryy"
    serialized_type: ClassVar[str] = "gate.ryy"
    parameter_names: ClassVar[tuple[str, ...]] = ("theta",)


@dataclass(frozen=True)
class Rzz(TwoQubitGate):
    """Rotate two ions around the zz axis."""

    theta: float
    circuit_name: ClassVar[str] = "rzz"
    serialized_type: ClassVar[str] = "gate.rzz"
    parameter_names: ClassVar[tuple[str, ...]] = ("theta",)


@dataclass(frozen=True)
class GlobalGate(GateAction):
    """Apply one single-ion rotation to several ions through global control.

    ``gate_name`` is the circuit name of the rotation: ``"rx"``, ``"ry"``, or
    ``"rz"``. ``ions`` lists every target ion explicitly and is stored in
    ascending order; code that applies a gate to all ions resolves them when it
    creates the gate. The gate takes the ordinary duration of its rotation. The
    executing architecture decides which other operations may run at the same
    time.
    """

    gate_name: str
    theta: float
    ions: tuple[int, ...]
    serialized_type: ClassVar[str] = "gate.global"
    parameter_names: ClassVar[tuple[str, ...]] = ("theta",)

    def __post_init__(self) -> None:
        """Validate the rotation and the target ions.

        Raises:
            TypeError: If the rotation name, angle, or a target ion has the wrong type.
            ValueError: If the angle is not finite, the rotation is unsupported,
                or the targets are empty, negative, or repeated.
        """
        super().__post_init__()
        if not isinstance(self.gate_name, str):
            msg = "gate_name must be a string"
            raise TypeError(msg)
        if self.gate_name not in SINGLE_QUBIT_GATE_NAMES:
            msg = f"global gates support only {sorted(SINGLE_QUBIT_GATE_NAMES)}, not {self.gate_name!r}"
            raise ValueError(msg)
        ions = tuple(self.ions)
        if any(isinstance(ion, bool) or not isinstance(ion, int) for ion in ions):
            msg = "global gate ions must be integers"
            raise TypeError(msg)
        if not ions:
            msg = "global gate ions must not be empty"
            raise ValueError(msg)
        if any(ion < 0 for ion in ions):
            msg = "global gate ions must be non-negative"
            raise ValueError(msg)
        if len(set(ions)) != len(ions):
            msg = "global gate ions must be unique"
            raise ValueError(msg)
        object.__setattr__(self, "ions", tuple(sorted(ions)))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a global gate.

        Returns:
            The restored global gate.

        Raises:
            ValueError: If the target ions are malformed.
        """
        try:
            return cls(
                gate_id=_optional_gate_id(data),
                gate_name=require_str(data, "gate_name"),
                theta=require_number(data, "theta"),
                ions=tuple(cast("list[int]", require_list(data, "ions"))),
            )
        except TypeError as error:
            msg = "invalid serialized GlobalGate action"
            raise ValueError(msg) from error

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of this global gate."""
        return {**super().to_dict(), "ions": list(self.ions)}


BUILTIN_GATE_TYPES: tuple[type[GateAction], ...] = (Rx, Ry, Rz, Rxx, Ryy, Rzz, GlobalGate)
GATE_TYPES: Mapping[str, type[Action]] = index_action_types(BUILTIN_GATE_TYPES)


def _construct_gate_action(action_type: type[GateAction], values: Mapping[str, object]) -> GateAction:
    constructor = cast("Any", action_type)
    return cast("GateAction", constructor(**values))


def _require_instruction_shape(
    gate_type: type[GateAction],
    ions: tuple[int, ...],
    parameters: tuple[float, ...],
    *,
    num_ions: int,
) -> None:
    if len(ions) != num_ions:
        msg = f"gate {gate_type.__name__} requires {num_ions} ions"
        raise ValueError(msg)
    if len(parameters) != len(gate_type.parameter_names):
        msg = f"gate {gate_type.__name__} requires {len(gate_type.parameter_names)} parameters"
        raise ValueError(msg)


def _optional_gate_id(data: Mapping[str, object]) -> int | None:
    value = data.get("gate_id")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = "gate_id must be a non-negative integer or null"
        raise ValueError(msg)
    return value


def _validate_gate_id(gate_id: object) -> None:
    if gate_id is not None and (isinstance(gate_id, bool) or not isinstance(gate_id, int) or gate_id < 0):
        msg = "gate_id must be a non-negative integer or None"
        raise ValueError(msg)


def _validate_ion(ion: object, name: str) -> None:
    if isinstance(ion, bool) or not isinstance(ion, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if ion < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


def _finite_parameter(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} must be a real number"
        raise TypeError(msg)
    if not math.isfinite(value):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return float(value)


def _normalize_virtual_gates(gate_names: object) -> frozenset[str]:
    if not isinstance(gate_names, frozenset | set | tuple | list):
        msg = "virtual_single_qubit_gates must be a collection of gate names"
        raise TypeError(msg)
    normalized_names: set[str] = set()
    for gate_name in gate_names:
        if not isinstance(gate_name, str):
            msg = "virtual_single_qubit_gates must contain only strings"
            raise TypeError(msg)
        normalized_names.add(gate_name.lower())
    normalized = frozenset(normalized_names)
    unknown = normalized.difference(SINGLE_QUBIT_GATE_NAMES)
    if unknown:
        msg = f"unknown virtual single-qubit gates: {sorted(unknown)}"
        raise ValueError(msg)
    return normalized


def _require_integer_at_least(value: object, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{name} must be an integer >= {minimum}"
        raise ValueError(msg)


def _mapping_duration(mapping: Mapping[str, object], name: str, default: int) -> int:
    value = mapping.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} duration must be an integer"
        raise TypeError(msg)
    return value


__all__ = [
    "BUILTIN_GATE_TYPES",
    "GATE_NAMES",
    "GATE_TYPES",
    "SINGLE_QUBIT_GATE_NAMES",
    "TWO_QUBIT_GATE_NAMES",
    "GateAction",
    "GateTiming",
    "GlobalGate",
    "Rx",
    "Rxx",
    "Ry",
    "Ryy",
    "Rz",
    "Rzz",
    "SingleQubitGate",
    "TwoQubitGate",
]
