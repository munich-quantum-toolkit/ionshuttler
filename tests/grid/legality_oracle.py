# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent movement rules for small Grid test cases.

The oracle states each transport rule in terms of distances from segment ends.
It does not call :meth:`GridArchitecture.apply_layer`, so tests can compare the
two descriptions on enumerated cases. It assumes that every ion and junction is
free, and it is deliberately simple rather than fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.ionshuttler.grid import SegmentEndpoint, SegmentOccupancy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.grid import GridArchitecture, JunctionMove


def layer_outcome(
    architecture: GridArchitecture,
    occupancy: Mapping[str, tuple[int, ...]],
    moves: Sequence[JunctionMove],
) -> dict[str, tuple[int, ...]] | None:
    """Return the occupancy after one simultaneous move layer.

    A moving chain keeps its order of travel. The ion nearest the departure
    end leads, so it ends farthest from the arrival end. Ions that stay in the
    destination keep their places farther inside the segment.

    Args:
        architecture: Topology, capacities, and occupancy kinds.
        occupancy: Canonical start-to-end ion order of every segment.
        moves: Junction moves that start together.

    Returns:
        The canonical occupancy after the layer, or ``None`` if the layer is illegal.
    """
    segments = {segment.segment_id: segment for segment in architecture.segments}
    moved_ions: list[int] = []
    used_junctions: list[str] = []
    departures: dict[SegmentEndpoint, tuple[int, ...]] = {}
    arrivals: dict[SegmentEndpoint, tuple[int, ...]] = {}
    for move in moves:
        source, destination = move.source, move.destination
        try:
            source_junction = architecture.junction_for(source)
            destination_junction = architecture.junction_for(destination)
        except KeyError:
            return None
        if source_junction != destination_junction:
            return None
        if source in departures or destination in arrivals:
            return None
        # The move lists its chain in source start-to-end order.
        nearest_first = move.ions if source.orientation == "start" else move.ions[::-1]
        if not _chain_leaves_end(
            segments[source.segment_id].occupancy, occupancy[source.segment_id], source, nearest_first
        ):
            return None
        departures[source] = nearest_first
        arrivals[destination] = nearest_first
        moved_ions.extend(move.ions)
        used_junctions.append(source_junction.junction_id)
    if len(set(moved_ions)) != len(moved_ions) or len(set(used_junctions)) != len(used_junctions):
        return None

    outcome: dict[str, tuple[int, ...]] = {}
    for segment in architecture.segments:
        leaving = {*departures.get(segment.start, ()), *departures.get(segment.end, ())}
        staying = tuple(ion for ion in occupancy[segment.segment_id] if ion not in leaving)
        # The last ion of an arriving chain stops nearest the arrival end.
        from_start = arrivals.get(segment.start, ())[::-1]
        toward_end = arrivals.get(segment.end, ())
        final = (*from_start, *staying, *toward_end)
        if len(final) > segment.capacity:
            return None
        outcome[segment.segment_id] = tuple(sorted(final)) if segment.occupancy is SegmentOccupancy.UNORDERED else final
    return outcome


def is_single_rotation(_architecture: GridArchitecture, moves: Sequence[JunctionMove]) -> bool:
    """Return whether moves leave and enter the same segments along one closed loop."""
    successor: dict[str, str] = {}
    for move in moves:
        source, destination = move.source, move.destination
        if source.segment_id in successor:
            return False
        successor[source.segment_id] = destination.segment_id
    if len(moves) < 2 or sorted(successor.values()) != sorted(successor):
        return False
    start = next(iter(successor))
    loop = [start]
    while (following := successor[loop[-1]]) != start:
        loop.append(following)
    return len(loop) == len(successor)


def _chain_leaves_end(
    occupancy_kind: SegmentOccupancy,
    occupants: tuple[int, ...],
    end: SegmentEndpoint,
    nearest_first: tuple[int, ...],
) -> bool:
    if occupancy_kind is SegmentOccupancy.UNORDERED:
        return set(nearest_first) <= set(occupants)
    from_end = occupants if end.orientation == "start" else occupants[::-1]
    return from_end[: len(nearest_first)] == nearest_first
