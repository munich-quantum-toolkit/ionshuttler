# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Canonical machine state for Grid architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .._json_utils import require_int, require_list, require_mapping, require_str_int_pairs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class GridMachineState:
    """Store segment occupancy and resource availability at one time."""

    occupancy: tuple[tuple[str, tuple[int, ...]], ...]
    ions_busy_until: tuple[tuple[int, int], ...]
    junctions_busy_until: tuple[tuple[str, int], ...]
    pzs_busy_until: tuple[tuple[str, int], ...]
    time: int = 0

    def __post_init__(self) -> None:
        """Normalize and validate machine-state values.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If keys, ions, or timestamps are inconsistent.
        """
        occupancy = tuple(sorted((segment_id, tuple(ions)) for segment_id, ions in self.occupancy))
        ions_busy_until = tuple(sorted(self.ions_busy_until))
        junctions_busy_until = tuple(sorted(self.junctions_busy_until))
        pzs_busy_until = tuple(sorted(self.pzs_busy_until))
        _require_unique_keys(occupancy, "occupancy")
        _require_unique_keys(ions_busy_until, "ions_busy_until")
        _require_unique_keys(junctions_busy_until, "junctions_busy_until")
        _require_unique_keys(pzs_busy_until, "pzs_busy_until")
        if any(not isinstance(segment_id, str) or not segment_id for segment_id, _ions in occupancy):
            msg = "occupancy segment identifiers must be non-empty strings"
            raise ValueError(msg)
        all_ions = tuple(ion for _segment_id, ions in occupancy for ion in ions)
        if any(isinstance(ion, bool) or not isinstance(ion, int) for ion in all_ions):
            msg = "occupancy must contain integer ion identifiers"
            raise TypeError(msg)
        if any(ion < 0 for ion in all_ions):
            msg = "ion identifiers must be non-negative"
            raise ValueError(msg)
        if len(set(all_ions)) != len(all_ions):
            msg = "each ion must occupy exactly one segment"
            raise ValueError(msg)
        if isinstance(self.time, bool) or not isinstance(self.time, int):
            msg = "time must be an integer"
            raise TypeError(msg)
        if self.time < 0:
            msg = "time must be non-negative"
            raise ValueError(msg)
        if {ion for ion, _free_time in ions_busy_until} != set(all_ions):
            msg = "ions_busy_until must contain exactly the placed ions"
            raise ValueError(msg)
        _validate_availability(ions_busy_until, self.time, "ion")
        _validate_availability(junctions_busy_until, self.time, "junction")
        _validate_availability(pzs_busy_until, self.time, "processing-zone")
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "ions_busy_until", ions_busy_until)
        object.__setattr__(self, "junctions_busy_until", junctions_busy_until)
        object.__setattr__(self, "pzs_busy_until", pzs_busy_until)

    @property
    def ions(self) -> tuple[int, ...]:
        """All placed ion identifiers in sorted order."""
        return tuple(sorted(ion for _segment_id, ions in self.occupancy for ion in ions))

    def occupants(self, segment_id: str) -> tuple[int, ...]:
        """Return the ions in one segment's canonical order."""
        return dict(self.occupancy)[segment_id]

    def ion_segment(self, ion: int) -> str:
        """Return the segment occupied by one ion.

        Raises:
            KeyError: If the state contains no such ion.
        """
        for segment_id, ions in self.occupancy:
            if ion in ions:
                return segment_id
        raise KeyError(ion)

    def at_time(self, time: int) -> GridMachineState:
        """Return this state with its clock advanced.

        Returns:
            The state at the requested time.

        Raises:
            TypeError: If the time is not an integer.
            ValueError: If time moves backwards.
        """
        if isinstance(time, bool) or not isinstance(time, int):
            msg = "time must be an integer"
            raise TypeError(msg)
        if time < self.time:
            msg = "time must not move backwards"
            raise ValueError(msg)
        return GridMachineState(
            occupancy=self.occupancy,
            ions_busy_until=tuple((key, max(value, time)) for key, value in self.ions_busy_until),
            junctions_busy_until=tuple((key, max(value, time)) for key, value in self.junctions_busy_until),
            pzs_busy_until=tuple((key, max(value, time)) for key, value in self.pzs_busy_until),
            time=time,
        )

    def to_dict(self) -> dict[str, object]:
        """Return this machine state using JSON-compatible values."""
        return {
            "occupancy": [[segment_id, list(ions)] for segment_id, ions in self.occupancy],
            "ions_busy_until": [list(item) for item in self.ions_busy_until],
            "junctions_busy_until": [list(item) for item in self.junctions_busy_until],
            "pzs_busy_until": [list(item) for item in self.pzs_busy_until],
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, data: object) -> GridMachineState:
        """Restore a Grid machine state.

        Returns:
            The restored state.

        Raises:
            ValueError: If the serialized state is invalid.
        """
        mapping = require_mapping(data, "grid machine state")
        occupancy: list[tuple[str, tuple[int, ...]]] = []
        for raw_item in require_list(mapping, "occupancy"):
            if not isinstance(raw_item, list) or len(raw_item) != 2 or not isinstance(raw_item[0], str):
                msg = "occupancy must contain segment and ion-list pairs"
                raise ValueError(msg)
            raw_ions = raw_item[1]
            if not isinstance(raw_ions, list) or any(
                isinstance(ion, bool) or not isinstance(ion, int) for ion in raw_ions
            ):
                msg = "occupancy ion lists must contain integers"
                raise ValueError(msg)
            occupancy.append((raw_item[0], tuple(cast("list[int]", raw_ions))))
        return cls(
            occupancy=tuple(occupancy),
            ions_busy_until=tuple(_require_int_pairs(mapping, "ions_busy_until")),
            junctions_busy_until=tuple(require_str_int_pairs(mapping, "junctions_busy_until")),
            pzs_busy_until=tuple(require_str_int_pairs(mapping, "pzs_busy_until")),
            time=require_int(mapping, "time"),
        )


def _require_int_pairs(data: Mapping[str, object], key: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for value in require_list(data, key):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or isinstance(value[0], bool)
            or not isinstance(value[0], int)
            or isinstance(value[1], bool)
            or not isinstance(value[1], int)
        ):
            msg = f"{key} must contain integer pairs"
            raise ValueError(msg)
        pairs.append((value[0], value[1]))
    return pairs


def _require_unique_keys(values: Sequence[tuple[object, object]], label: str) -> None:
    if len({key for key, _value in values}) != len(values):
        msg = f"{label} must not contain duplicate keys"
        raise ValueError(msg)


def _validate_availability(values: Sequence[tuple[object, int]], time: int, label: str) -> None:
    for _key, free_time in values:
        if isinstance(free_time, bool) or not isinstance(free_time, int):
            msg = f"{label} availability times must be integers"
            raise TypeError(msg)
        if free_time < time:
            msg = f"{label} availability times must not precede the machine time"
            raise ValueError(msg)


__all__ = ["GridMachineState"]
