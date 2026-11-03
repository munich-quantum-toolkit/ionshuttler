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

from .._json_utils import require_int_list, require_list, require_mapping, require_str
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


@dataclass(frozen=True)
class Rearrangement(Action):
    """Reorder a chain of neighboring ions inside one ordered segment.

    ``ions`` lists the chain in the segment's start-to-end order before the
    rearrangement and ``new_order`` after it. A swap of two ions is the
    rearrangement of a two-ion chain. The rearrangement occupies only the ions
    of the chain. Other ions can enter or leave the segment meanwhile; an
    arriving ion joins at the segment end it enters, outside the chain.
    """

    segment_id: str
    ions: tuple[int, ...]
    new_order: tuple[int, ...]
    serialized_type: ClassVar[str] = "grid.rearrangement"

    def __post_init__(self) -> None:
        """Validate and normalize the rearrangement data.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If the segment identifier, the chain, or the new order is invalid.
        """
        if not isinstance(self.segment_id, str):
            msg = "segment_id must be a string"
            raise TypeError(msg)
        if not self.segment_id:
            msg = "segment_id must be non-empty"
            raise ValueError(msg)
        ions = tuple(self.ions)
        new_order = tuple(self.new_order)
        if any(isinstance(ion, bool) or not isinstance(ion, int) for ion in (*ions, *new_order)):
            msg = "ions and new_order must contain integers"
            raise TypeError(msg)
        if len(ions) < 2 or len(set(ions)) != len(ions) or any(ion < 0 for ion in ions):
            msg = "ions must be at least two different non-negative ions"
            raise ValueError(msg)
        if sorted(new_order) != sorted(ions):
            msg = "new_order must contain exactly the ions of the chain"
            raise ValueError(msg)
        if new_order == ions:
            msg = "new_order must differ from the current order"
            raise ValueError(msg)
        object.__setattr__(self, "ions", ions)
        object.__setattr__(self, "new_order", new_order)

    @property
    def swap_layers(self) -> int:
        """Layers of simultaneous neighbor swaps that the rearrangement takes.

        The swaps are the exchanges of the permutation-based Linear compiler:
        an insertion sort with one swap per pair of ions in the wrong order.
        Each swap starts as soon as both of its ions are free.
        """
        position = {ion: index for index, ion in enumerate(self.new_order)}
        order = [position[ion] for ion in self.ions]
        free_after = [0] * len(order)
        for first_unsorted in range(1, len(order)):
            right = first_unsorted
            while right > 0 and order[right - 1] > order[right]:
                left = right - 1
                # Both ions are busy until the swap ends; they keep equal times when they exchange sites.
                free_after[left] = free_after[right] = max(free_after[left], free_after[right]) + 1
                order[left], order[right] = order[right], order[left]
                right = left
        return max(free_after)

    def __str__(self) -> str:
        """Return a concise description for schedule displays."""
        before = ",".join(f"q{ion}" for ion in self.ions)
        after = ",".join(f"q{ion}" for ion in self.new_order)
        return f"rearrange [{before}] → [{after}] in {self.segment_id}"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of this rearrangement."""
        return {
            "type": self.serialized_type,
            "segment_id": self.segment_id,
            "ions": list(self.ions),
            "new_order": list(self.new_order),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Rearrangement:
        """Restore a rearrangement from serialized data.

        Returns:
            The restored rearrangement.
        """
        return cls(
            segment_id=require_str(data, "segment_id"),
            ions=tuple(require_int_list(data, "ions")),
            new_order=tuple(require_int_list(data, "new_order")),
        )


DEFAULT_ACTION_TYPES: tuple[type[Action], ...] = (
    JunctionMove,
    Cycle,
    Rearrangement,
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


__all__ = [
    "DEFAULT_ACTION_TYPES",
    "GRID_ACTION_TYPES",
    "Cycle",
    "JunctionMove",
    "Rearrangement",
    "decode_grid_action",
]
