# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Grid transport actions and serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import mqt.ionshuttler.core.gates as core_gates
from mqt.ionshuttler.core.actions import Action, decode_action, index_action_types

from .._json_utils import require_int_list, require_list, require_mapping
from .model import SegmentEndpoint

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class JunctionMove(Action):
    """Move one ion chain between two endpoints of a junction.

    The ordering of ``ions`` follows the source segment's canonical
    start-to-end order, regardless of which endpoint the chain leaves through.
    """

    source: SegmentEndpoint
    destination: SegmentEndpoint
    ions: tuple[int, ...]
    serialized_type: ClassVar[str] = "grid.junction_move"

    def __post_init__(self) -> None:
        """Validate and normalize the movement data.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If an identifier or ion list is invalid.
        """
        if not isinstance(self.source, SegmentEndpoint) or not isinstance(self.destination, SegmentEndpoint):
            msg = "source and destination must be SegmentEndpoint values"
            raise TypeError(msg)
        if self.source == self.destination:
            msg = "source and destination must be different endpoints"
            raise ValueError(msg)
        ions = tuple(self.ions)
        if not ions:
            msg = "ions must be non-empty"
            raise ValueError(msg)
        if any(isinstance(ion, bool) or not isinstance(ion, int) for ion in ions):
            msg = "ions must contain integers"
            raise TypeError(msg)
        if any(ion < 0 for ion in ions):
            msg = "ions must be non-negative"
            raise ValueError(msg)
        if len(set(ions)) != len(ions):
            msg = "ions must not contain duplicates"
            raise ValueError(msg)
        object.__setattr__(self, "ions", ions)

    def __str__(self) -> str:
        """Return a concise description for schedule displays."""
        ions = ",".join(f"q{ion}" for ion in self.ions)
        source = f"{self.source.segment_id}:{self.source.orientation}"
        destination = f"{self.destination.segment_id}:{self.destination.orientation}"
        return f"move [{ions}] {source} → {destination}"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of this move."""
        return {
            "type": self.serialized_type,
            "source": self.source.to_dict(),
            "destination": self.destination.to_dict(),
            "ions": list(self.ions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> JunctionMove:
        """Restore a junction move from serialized data.

        Returns:
            The restored move.
        """
        return cls(
            source=SegmentEndpoint.from_dict(data.get("source")),
            destination=SegmentEndpoint.from_dict(data.get("destination")),
            ions=tuple(require_int_list(data, "ions")),
        )


@dataclass(frozen=True)
class Cycle(Action):
    """Apply a closed collection of junction moves simultaneously."""

    moves: tuple[JunctionMove, ...]
    serialized_type: ClassVar[str] = "grid.cycle"

    def __post_init__(self) -> None:
        """Validate and normalize the component moves.

        Raises:
            TypeError: If a component is not a junction move.
            ValueError: If the cycle is too short or moves one ion more than once.
        """
        moves = tuple(self.moves)
        if any(not isinstance(move, JunctionMove) for move in moves):
            msg = "moves must contain JunctionMove values"
            raise TypeError(msg)
        if len(moves) < 2:
            msg = "a cycle must contain at least two moves"
            raise ValueError(msg)
        ions = tuple(ion for move in moves for ion in move.ions)
        if len(set(ions)) != len(ions):
            msg = "a cycle must not move one ion more than once"
            raise ValueError(msg)
        object.__setattr__(self, "moves", moves)

    @property
    def ions(self) -> tuple[int, ...]:
        """All ions moved by the cycle."""
        return tuple(ion for move in self.moves for ion in move.ions)

    def __str__(self) -> str:
        """Return a concise description of the simultaneous rotation."""
        parts = []
        for move in self.moves:
            ions = ",".join(f"q{ion}" for ion in move.ions)
            label = f"q{move.ions[0]}" if len(move.ions) == 1 else f"[{ions}]"
            parts.append(f"{label}:{move.source.segment_id}→{move.destination.segment_id}")
        return f"cycle {', '.join(parts)}"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of this cycle."""
        return {"type": self.serialized_type, "moves": [move.to_dict() for move in self.moves]}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Cycle:
        """Restore a simultaneous cycle from serialized data.

        Returns:
            The restored cycle.

        """
        moves: list[JunctionMove] = []
        for raw_move in require_list(data, "moves"):
            move_data = require_mapping(raw_move, "cycle move")
            moves.append(JunctionMove.from_dict(move_data))
        return cls(tuple(moves))


DEFAULT_ACTION_TYPES: tuple[type[Action], ...] = (
    JunctionMove,
    Cycle,
    core_gates.Rx,
    core_gates.Ry,
    core_gates.Rz,
    core_gates.Rxx,
    core_gates.Ryy,
    core_gates.Rzz,
    core_gates.Measure,
    core_gates.Reset,
)
GRID_ACTION_TYPES: Mapping[str, type[Action]] = index_action_types(DEFAULT_ACTION_TYPES)


def decode_grid_action(data: object) -> Action:
    """Restore one action implemented by Grid architectures.

    Returns:
        The restored action.
    """
    return decode_action(data, GRID_ACTION_TYPES)


__all__ = ["DEFAULT_ACTION_TYPES", "GRID_ACTION_TYPES", "Cycle", "JunctionMove", "decode_grid_action"]
