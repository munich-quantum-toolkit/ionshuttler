# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Immutable hardware values for segment-graph architectures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from mqt.ionshuttler.core.gates import GateAction, Rx, Rxx, Ry, Ryy, Rz, Rzz

from .._json_utils import require_int, require_list, require_mapping, require_str

if TYPE_CHECKING:
    from collections.abc import Mapping

LOCAL_GATE_TYPES: tuple[type[GateAction], ...] = (Rx, Ry, Rz, Rxx, Ryy, Rzz)


class SegmentOccupancy(StrEnum):
    """Describe whether ion order inside a segment is significant."""

    ORDERED = "ordered"
    UNORDERED = "unordered"


@dataclass(frozen=True, order=True)
class SegmentEndpoint:
    """Identify one stable segment boundary."""

    segment_id: str
    orientation: Literal["start", "end"]

    def __post_init__(self) -> None:
        """Validate the segment identity and endpoint orientation.

        Raises:
            ValueError: If the orientation is not ``"start"`` or ``"end"``.
        """
        _require_identifier(self.segment_id, "segment_id")
        if self.orientation not in {"start", "end"}:
            msg = "orientation must be 'start' or 'end'"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, str]:
        """Return JSON-compatible endpoint data."""
        return {"segment_id": self.segment_id, "orientation": self.orientation}

    @classmethod
    def from_dict(cls, data: object) -> SegmentEndpoint:
        """Restore an endpoint from serialized data.

        Returns:
            The restored endpoint.

        Raises:
            ValueError: If the serialized endpoint is invalid.
        """
        mapping = require_mapping(data, "segment endpoint")
        orientation = require_str(mapping, "orientation")
        if orientation not in {"start", "end"}:
            msg = "endpoint orientation must be 'start' or 'end'"
            raise ValueError(msg)
        return cls(require_str(mapping, "segment_id"), orientation)


@dataclass(frozen=True)
class Segment:
    """Describe one capacity-bounded location in the transport graph."""

    segment_id: str
    capacity: int = 1
    occupancy: SegmentOccupancy = SegmentOccupancy.ORDERED
    processing_zones: tuple[ProcessingZone, ...] = ()

    def __post_init__(self) -> None:
        """Validate the segment.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If processing-zone identities repeat.
        """
        _require_identifier(self.segment_id, "segment_id")
        _require_positive_int(self.capacity, "capacity")
        if not isinstance(self.occupancy, SegmentOccupancy):
            msg = "occupancy must be a SegmentOccupancy"
            raise TypeError(msg)
        zones = tuple(self.processing_zones)
        if any(not isinstance(zone, ProcessingZone) for zone in zones):
            msg = "processing_zones must contain ProcessingZone values"
            raise TypeError(msg)
        zone_ids = [zone.zone_id for zone in zones]
        if len(set(zone_ids)) != len(zone_ids):
            msg = "processing_zones must have unique identifiers within a segment"
            raise ValueError(msg)
        object.__setattr__(self, "processing_zones", zones)

    @property
    def start(self) -> SegmentEndpoint:
        """The segment's canonical start boundary."""
        return SegmentEndpoint(self.segment_id, "start")

    @property
    def end(self) -> SegmentEndpoint:
        """The segment's canonical end boundary."""
        return SegmentEndpoint(self.segment_id, "end")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible segment data."""
        return {
            "segment_id": self.segment_id,
            "capacity": self.capacity,
            "occupancy": self.occupancy.value,
            "processing_zones": [zone.to_dict() for zone in self.processing_zones],
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        gate_types: Mapping[str, type[GateAction]] | None = None,
    ) -> Segment:
        """Restore a segment from serialized data.

        Returns:
            The restored segment.

        Raises:
            ValueError: If the serialized segment is invalid.
        """
        mapping = require_mapping(data, "segment")
        try:
            occupancy = SegmentOccupancy(require_str(mapping, "occupancy"))
        except ValueError as error:
            msg = "segment occupancy must be 'ordered' or 'unordered'"
            raise ValueError(msg) from error
        registry = (
            {gate_type.serialized_type: gate_type for gate_type in LOCAL_GATE_TYPES}
            if gate_types is None
            else gate_types
        )
        return cls(
            require_str(mapping, "segment_id"),
            require_int(mapping, "capacity"),
            occupancy,
            tuple(
                ProcessingZone.from_dict(item, gate_types=registry)
                for item in require_list(mapping, "processing_zones")
            ),
        )


@dataclass(frozen=True, order=True)
class Junction:
    """Join segment endpoints through one shared transport resource."""

    junction_id: str
    endpoints: tuple[SegmentEndpoint, ...]

    def __post_init__(self) -> None:
        """Validate and normalize the junction.

        Raises:
            TypeError: If an endpoint has the wrong type.
            ValueError: If an endpoint occurs more than once.
        """
        _require_identifier(self.junction_id, "junction_id")
        endpoints = tuple(self.endpoints)
        if any(not isinstance(endpoint, SegmentEndpoint) for endpoint in endpoints):
            msg = "endpoints must contain SegmentEndpoint values"
            raise TypeError(msg)
        if len(set(endpoints)) != len(endpoints):
            msg = "endpoints must not contain duplicates"
            raise ValueError(msg)
        object.__setattr__(self, "endpoints", tuple(sorted(endpoints)))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible junction data."""
        return {
            "junction_id": self.junction_id,
            "endpoints": [endpoint.to_dict() for endpoint in self.endpoints],
        }

    @classmethod
    def from_dict(cls, data: object) -> Junction:
        """Restore a junction from serialized data.

        Returns:
            The restored junction.
        """
        mapping = require_mapping(data, "junction")
        return cls(
            require_str(mapping, "junction_id"),
            tuple(SegmentEndpoint.from_dict(item) for item in require_list(mapping, "endpoints")),
        )


@dataclass(frozen=True)
class ProcessingZone:
    """Describe one gate-capable control region within its parent segment."""

    zone_id: str
    supported_gate_types: tuple[type[GateAction], ...] = LOCAL_GATE_TYPES

    def __post_init__(self) -> None:
        """Validate and normalize the processing zone.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If an identity is empty or gate types repeat.
        """
        _require_identifier(self.zone_id, "zone_id")
        gate_types = tuple(self.supported_gate_types)
        if any(not isinstance(gate_type, type) or not issubclass(gate_type, GateAction) for gate_type in gate_types):
            msg = "supported_gate_types must contain GateAction subclasses"
            raise TypeError(msg)
        if len(set(gate_types)) != len(gate_types):
            msg = "supported_gate_types must not contain duplicates"
            raise ValueError(msg)
        object.__setattr__(self, "supported_gate_types", gate_types)

    def supports(self, gate: GateAction | type[GateAction]) -> bool:
        """Return whether this zone implements a gate type."""
        gate_type = gate if isinstance(gate, type) else type(gate)
        return gate_type in self.supported_gate_types

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible processing-zone data."""
        return {
            "zone_id": self.zone_id,
            "supported_gate_types": [gate_type.serialized_type for gate_type in self.supported_gate_types],
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        gate_types: Mapping[str, type[GateAction]],
    ) -> ProcessingZone:
        """Restore a processing zone using an explicit gate registry.

        Returns:
            The restored processing zone.

        Raises:
            ValueError: If a declared gate type is unknown.
        """
        mapping = require_mapping(data, "processing zone")
        restored: list[type[GateAction]] = []
        for raw_name in require_list(mapping, "supported_gate_types"):
            if not isinstance(raw_name, str):
                msg = "supported_gate_types must contain strings"
                raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON errors use ValueError.
            gate_type = gate_types.get(raw_name)
            if gate_type is None:
                msg = f"unknown processing-zone gate type: {raw_name}"
                raise ValueError(msg)
            restored.append(gate_type)
        return cls(require_str(mapping, "zone_id"), tuple(restored))


@dataclass(frozen=True)
class TransportTiming:
    """Configure the duration of Grid transport actions."""

    junction_move: int = 1

    def __post_init__(self) -> None:
        """Ensure transport takes at least one timestep."""
        _require_positive_int(self.junction_move, "junction_move")

    def to_dict(self) -> dict[str, int]:
        """Return JSON-compatible timing data."""
        return {"junction_move": self.junction_move}

    @classmethod
    def from_dict(cls, data: object) -> TransportTiming:
        """Restore Grid transport timing.

        Returns:
            The restored timing.
        """
        mapping = require_mapping(data, "transport timing")
        return cls(require_int(mapping, "junction_move"))


def _require_identifier(value: object, label: str) -> None:
    if not isinstance(value, str):
        msg = f"{label} must be a string"
        raise TypeError(msg)
    if not value:
        msg = f"{label} must be non-empty"
        raise ValueError(msg)


def _require_positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer"
        raise TypeError(msg)
    if value < 1:
        msg = f"{label} must be >= 1"
        raise ValueError(msg)


__all__ = [
    "Junction",
    "ProcessingZone",
    "Segment",
    "SegmentEndpoint",
    "SegmentOccupancy",
    "TransportTiming",
]
