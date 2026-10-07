# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Linear transport actions, their timing, and shared gate imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from mqt.ionshuttler.core.actions import Action, decode_action, index_action_types
from mqt.ionshuttler.core.gates import (
    BUILTIN_GATE_TYPES,
    GateAction,
    GlobalGate,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    SingleQubitGate,
    TwoQubitGate,
)

from .._json_utils import require_int

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class TransportTiming:
    """Configure integer durations for Linear transport operations."""

    shuttle: int = 1
    swap: int = 3

    def __post_init__(self) -> None:
        """Ensure every transport operation takes at least one timestep."""
        _require_integer_at_least(self.shuttle, "shuttle duration", minimum=1)
        _require_integer_at_least(self.swap, "swap duration", minimum=1)

    def to_dict(self) -> dict[str, int]:
        """Return JSON-compatible transport timing."""
        return {"shuttle": self.shuttle, "swap": self.swap}

    @classmethod
    def from_dict(cls, data: object) -> TransportTiming:
        """Restore transport timing from serialized architecture data.

        Missing fields use their default values.

        Returns:
            The restored transport timing.

        Raises:
            TypeError: If the data or a duration has the wrong type.
            ValueError: If the data contains an unknown field.
        """
        if not isinstance(data, dict):
            msg = "transport_timing must be a JSON object"
            raise TypeError(msg)
        mapping = cast("dict[str, object]", data)
        unknown_fields = sorted(set(mapping).difference({"shuttle", "swap"}))
        if unknown_fields:
            msg = f"unknown transport_timing fields: {', '.join(unknown_fields)}"
            raise ValueError(msg)
        defaults = cls()
        return cls(
            shuttle=_mapping_duration(mapping, "shuttle", defaults.shuttle),
            swap=_mapping_duration(mapping, "swap", defaults.swap),
        )


@dataclass(frozen=True)
class TransportAction(Action):
    """A Linear hardware operation that moves ions between sites."""

    if TYPE_CHECKING:

        @property
        def ions(self) -> tuple[int, ...]:
            """The transport's ordered ion operands."""
            ...


@dataclass(frozen=True)
class Shuttle(TransportAction):
    """Move one ion between adjacent sites."""

    ion: int
    src: int
    dst: int
    serialized_type: ClassVar[str] = "linear.shuttle"

    @property
    def ions(self) -> tuple[int, ...]:
        """The transported ion."""
        return (self.ion,)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a shuttle from serialized fields.

        Returns:
            The restored shuttle.
        """
        return cls(
            ion=require_int(data, "ion"),
            src=require_int(data, "src"),
            dst=require_int(data, "dst"),
        )


@dataclass(frozen=True)
class PhysicalSwap(TransportAction):
    """Exchange two ions occupying adjacent sites."""

    ion_a: int
    ion_b: int
    pos_a: int
    pos_b: int
    serialized_type: ClassVar[str] = "linear.physical_swap"

    @property
    def ions(self) -> tuple[int, ...]:
        """The swapped ions in position order."""
        return (self.ion_a, self.ion_b)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore a physical swap from serialized fields.

        Returns:
            The restored swap.
        """
        return cls(
            ion_a=require_int(data, "ion_a"),
            ion_b=require_int(data, "ion_b"),
            pos_a=require_int(data, "pos_a"),
            pos_b=require_int(data, "pos_b"),
        )


DEFAULT_ACTION_TYPES: tuple[type[Action], ...] = (PhysicalSwap, Shuttle, Rx, Ry, Rz, Rzz)
LINEAR_ACTION_TYPES: Mapping[str, type[Action]] = index_action_types((*BUILTIN_GATE_TYPES, Shuttle, PhysicalSwap))


def decode_linear_action(data: object) -> Action:
    """Restore one serialized action implemented by Linear architectures.

    Returns:
        The restored action.
    """
    return decode_action(data, LINEAR_ACTION_TYPES)


def is_adjacent(pos_a: int, pos_b: int) -> bool:
    """Return whether two Linear site indices are adjacent."""
    return abs(pos_a - pos_b) == 1


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
    "DEFAULT_ACTION_TYPES",
    "LINEAR_ACTION_TYPES",
    "Action",
    "GateAction",
    "GlobalGate",
    "PhysicalSwap",
    "Rx",
    "Rxx",
    "Ry",
    "Ryy",
    "Rz",
    "Rzz",
    "Shuttle",
    "SingleQubitGate",
    "TransportAction",
    "TransportTiming",
    "TwoQubitGate",
    "decode_linear_action",
    "is_adjacent",
]
