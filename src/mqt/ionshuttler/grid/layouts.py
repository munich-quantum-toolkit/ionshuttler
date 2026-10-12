# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Adapters that construct canonical Grid architectures."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

from mqt.ionshuttler.grid.architecture import GridArchitecture
from mqt.ionshuttler.grid.model import (
    Junction,
    ProcessingZone,
    Segment,
    SegmentEndpoint,
    SegmentOccupancy,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Protocol

    class _PhysicalGraph(Protocol):
        """Structural input used by the NetworkX adapter."""

        def is_directed(self) -> bool:
            """Return whether graph edges are directed."""

        def nodes(self, *, data: bool) -> Iterable[tuple[object, Mapping[str, object]]]:
            """Iterate over nodes and their attributes."""

        def edges(self, *, data: bool) -> Iterable[tuple[object, object, Mapping[str, object]]]:
            """Iterate over edges and their attributes."""


def square_grid(
    size: int,
    *,
    processing_zone_segments: Iterable[str] = (),
    segment_capacity: int = 1,
    unordered_segments: Iterable[str] = (),
) -> GridArchitecture:
    """Construct a square physical layout with edge-based Grid segments.

    ``size`` is the number of junctions along each side. Every physical edge
    becomes one compiler segment. Each physical junction owns the endpoints of
    its incident segments. An ion can move between any two distinct endpoints
    of that junction.

    Returns:
        The canonical Grid architecture.
    """
    return rectangular_grid(
        size,
        size,
        processing_zone_segments=processing_zone_segments,
        segment_capacity=segment_capacity,
        unordered_segments=unordered_segments,
    )


def rectangular_grid(
    rows: int,
    columns: int,
    *,
    processing_zone_segments: Iterable[str] = (),
    segment_capacity: int = 1,
    unordered_segments: Iterable[str] = (),
) -> GridArchitecture:
    """Construct a rectangular physical layout with stable row-column IDs.

    Horizontal segment IDs have the form ``h:<row>:<column>``. Vertical
    segment IDs have the form ``v:<row>:<column>``. Junction IDs have the form
    ``j:<row>:<column>``. These identities do not depend on drawing geometry.

    Returns:
        The canonical Grid architecture.

    Raises:
        TypeError: If a dimension or capacity is not an integer.
        ValueError: If a dimension is too small or a named segment is unknown.
    """
    _require_grid_dimension(rows, "rows")
    _require_grid_dimension(columns, "columns")
    if isinstance(segment_capacity, bool) or not isinstance(segment_capacity, int):
        msg = "segment_capacity must be an integer"
        raise TypeError(msg)
    if segment_capacity < 1:
        msg = "segment_capacity must be >= 1"
        raise ValueError(msg)
    unordered = set(unordered_segments)
    pz_segments = tuple(processing_zone_segments)
    pz_segment_set = set(pz_segments)
    segments: list[Segment] = []
    incident: dict[str, list[SegmentEndpoint]] = defaultdict(list)

    for row in range(rows):
        for column in range(columns - 1):
            segment_id = f"h:{row}:{column}"
            segment = _make_segment(segment_id, segment_capacity, unordered, pz_segment_set)
            segments.append(segment)
            incident[f"j:{row}:{column}"].append(segment.start)
            incident[f"j:{row}:{column + 1}"].append(segment.end)
    for row in range(rows - 1):
        for column in range(columns):
            segment_id = f"v:{row}:{column}"
            segment = _make_segment(segment_id, segment_capacity, unordered, pz_segment_set)
            segments.append(segment)
            incident[f"j:{row}:{column}"].append(segment.start)
            incident[f"j:{row + 1}:{column}"].append(segment.end)

    segment_ids = {segment.segment_id for segment in segments}
    _validate_named_segments(unordered, pz_segment_set, segment_ids)

    junctions = tuple(
        Junction(junction_id, tuple(incident[junction_id]))
        for junction_id in (f"j:{row}:{column}" for row in range(rows) for column in range(columns))
    )
    return GridArchitecture(tuple(segments), junctions)


def hexagonal_grid(
    rows: int,
    columns: int,
    *,
    processing_zone_segments: Iterable[str] = (),
    segment_capacity: int = 1,
    unordered_segments: Iterable[str] = (),
) -> GridArchitecture:
    """Construct a brick-wall patch of a hexagonal honeycomb lattice.

    ``rows`` and ``columns`` count junction rows and columns. The smallest
    complete hexagonal cell uses two rows and three columns. Segment IDs encode
    their endpoint junction IDs and do not assign a cardinal direction.

    Returns:
        The canonical Grid architecture.

    Raises:
        TypeError: If a dimension or capacity is not an integer.
        ValueError: If the patch is too small or a named segment is unknown.
    """
    _require_grid_dimension(rows, "rows")
    _require_grid_dimension(columns, "columns")
    if columns < 3:
        msg = "columns must be >= 3 for a hexagonal grid"
        raise ValueError(msg)
    if isinstance(segment_capacity, bool) or not isinstance(segment_capacity, int):
        msg = "segment_capacity must be an integer"
        raise TypeError(msg)
    if segment_capacity < 1:
        msg = "segment_capacity must be >= 1"
        raise ValueError(msg)

    physical_edges = {((row, column), (row, column + 1)) for row in range(rows) for column in range(columns - 1)}
    physical_edges.update(
        ((row, column), (row + 1, column))
        for row in range(rows - 1)
        for column in range(columns)
        if (row + column) % 2 == 0
    )
    unordered = set(unordered_segments)
    pz_segments = set(processing_zone_segments)
    segments: list[Segment] = []
    incident: dict[str, list[SegmentEndpoint]] = defaultdict(list)
    for start, end in sorted(physical_edges):
        start_id = f"j:{start[0]}:{start[1]}"
        end_id = f"j:{end[0]}:{end[1]}"
        segment_id = f"s:{start_id}--{end_id}"
        segment = _make_segment(segment_id, segment_capacity, unordered, pz_segments)
        segments.append(segment)
        incident[start_id].append(segment.start)
        incident[end_id].append(segment.end)

    segment_ids = {segment.segment_id for segment in segments}
    _validate_named_segments(unordered, pz_segments, segment_ids)
    junctions = tuple(
        Junction(junction_id, tuple(incident[junction_id]))
        for junction_id in (f"j:{row}:{column}" for row in range(rows) for column in range(columns))
    )
    return GridArchitecture(tuple(segments), junctions)


def from_networkx(
    graph: object,
    *,
    processing_zone_segments: Iterable[str] = (),
) -> GridArchitecture:
    """Convert a physical NetworkX graph into a canonical Grid architecture.

    Physical graph nodes become junctions and graph edges become segments. A
    node can define a stable ``junction_id`` attribute. An edge can define
    ``segment_id``, ``capacity``, and ``occupancy`` attributes. String and
    integer node values also provide stable default IDs. Rich editor data such
    as coordinates and labels remains outside the compiler architecture.

    Returns:
        The converted Grid architecture.

    Raises:
        TypeError: If the input is directed or a node has no stable identity.
        ValueError: If generated identities collide or attributes are invalid.
    """
    physical_graph = cast("_PhysicalGraph", graph)
    if physical_graph.is_directed():
        msg = "from_networkx expects an undirected physical layout graph"
        raise TypeError(msg)
    junction_ids = {node: _networkx_junction_id(node, data) for node, data in physical_graph.nodes(data=True)}
    if len(set(junction_ids.values())) != len(junction_ids):
        msg = "NetworkX junction identities must be unique"
        raise ValueError(msg)
    incident: dict[str, list[SegmentEndpoint]] = defaultdict(list)
    pz_segments = set(processing_zone_segments)
    segments: list[Segment] = []
    for node_a, node_b, data in physical_graph.edges(data=True):
        id_a = junction_ids[node_a]
        id_b = junction_ids[node_b]
        start_node, end_node = (node_a, node_b) if id_a < id_b else (node_b, node_a)
        start_junction = junction_ids[start_node]
        end_junction = junction_ids[end_node]
        segment_id = data.get("segment_id", f"segment:{start_junction}--{end_junction}")
        if not isinstance(segment_id, str):
            msg = "NetworkX segment_id attributes must be strings"
            raise TypeError(msg)
        capacity = data.get("capacity", 1)
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            msg = "NetworkX capacity attributes must be integers"
            raise TypeError(msg)
        occupancy_value = data.get("occupancy", SegmentOccupancy.ORDERED.value)
        try:
            occupancy = SegmentOccupancy(occupancy_value)
        except (TypeError, ValueError) as error:
            msg = "NetworkX occupancy attributes must be 'ordered' or 'unordered'"
            raise ValueError(msg) from error
        zones = (ProcessingZone(f"pz:{segment_id}"),) if segment_id in pz_segments else ()
        segment = Segment(segment_id, capacity, occupancy, zones)
        segments.append(segment)
        incident[start_junction].append(segment.start)
        incident[end_junction].append(segment.end)
    unknown_pzs = sorted(pz_segments.difference(segment.segment_id for segment in segments))
    if unknown_pzs:
        msg = f"processing_zone_segments contains unknown segments: {', '.join(unknown_pzs)}"
        raise ValueError(msg)
    return GridArchitecture(
        segments=tuple(segments),
        junctions=tuple(Junction(junction_id, tuple(incident[junction_id])) for junction_id in junction_ids.values()),
    )


def _make_segment(
    segment_id: str,
    capacity: int,
    unordered: set[str],
    processing_zone_segments: set[str],
) -> Segment:
    occupancy = SegmentOccupancy.UNORDERED if segment_id in unordered else SegmentOccupancy.ORDERED
    zones = (ProcessingZone(f"pz:{segment_id}"),) if segment_id in processing_zone_segments else ()
    return Segment(segment_id, capacity, occupancy, zones)


def _validate_named_segments(unordered: set[str], pz_segments: set[str], segment_ids: set[str]) -> None:
    unknown_relaxed = sorted(unordered.difference(segment_ids))
    unknown_pzs = sorted(pz_segments.difference(segment_ids))
    if unknown_relaxed:
        msg = f"unordered_segments contains unknown segments: {', '.join(unknown_relaxed)}"
        raise ValueError(msg)
    if unknown_pzs:
        msg = f"processing_zone_segments contains unknown segments: {', '.join(unknown_pzs)}"
        raise ValueError(msg)


def _networkx_junction_id(node: object, data: Mapping[str, object]) -> str:
    declared = data.get("junction_id")
    if declared is not None:
        if not isinstance(declared, str):
            msg = "NetworkX junction_id attributes must be strings"
            raise TypeError(msg)
        return declared
    if isinstance(node, str):
        return node
    if isinstance(node, int) and not isinstance(node, bool):
        return str(node)
    if (
        isinstance(node, tuple)
        and len(node) == 2
        and all(isinstance(coordinate, int) and not isinstance(coordinate, bool) for coordinate in node)
    ):
        return f"{node[0]},{node[1]}"
    msg = "NetworkX nodes need string, integer, integer-pair, or explicit junction_id identities"
    raise TypeError(msg)


def _require_grid_dimension(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer"
        raise TypeError(msg)
    if value < 2:
        msg = f"{label} must be >= 2"
        raise ValueError(msg)


__all__ = ["from_networkx", "hexagonal_grid", "rectangular_grid", "square_grid"]
