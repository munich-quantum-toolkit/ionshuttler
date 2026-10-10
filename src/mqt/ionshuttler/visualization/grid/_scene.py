# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Drawing data for Grid results that every Grid renderer shares.

A scene stores the hardware layout, each ion's movements, and the action layers
of one replayed schedule. Its size grows with the scheduled actions, not with
the number of drawn frames.

Positions depend only on the junction coordinates. Each segment end sits at a
junction, or at a fixed offset from one. Each ion position is a segment and a
fraction of the way from the segment start to its end. A renderer can therefore
move a junction and redraw every segment and ion without a new replay. The
browser drawing in ``draw.js`` and the Matplotlib renderer use the same rules.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby
from typing import TYPE_CHECKING, Any

from mqt.ionshuttler.core.gates import GateAction, GlobalGate
from mqt.ionshuttler.grid.actions import Cycle, JunctionMove
from mqt.ionshuttler.grid.architecture import GridArchitecture
from mqt.ionshuttler.grid.model import Junction, SegmentEndpoint
from mqt.ionshuttler.grid.state import GridMachineState

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.core.result import CompilationResult
    from mqt.ionshuttler.core.schedule import ScheduledAction
    from mqt.ionshuttler.visualization.grid.visualizer import JunctionCoordinates

Coordinate = tuple[float, float]

_GRID_JUNCTION_ID = re.compile(r"j:(-?\d+):(-?\d+)\Z")
_DIGITS = 4
_ION_RADIUS_PER_SPACING = 0.35
_MAX_ION_RADIUS = 0.06
_MARGIN_PER_ION_RADIUS = 2.5

# Theme colors of the drawing. "ion" is the ring color in single-color mode,
# "ion_fill" fills ions without a chosen fill, and "gate" marks gates that run
# outside a processing zone.
COLORS: dict[str, dict[str, str]] = {
    "light": {
        "background": "#ffffff",
        "segment": "#d7dbe0",
        "junction": "#8a929d",
        "ion": "#3f4b5b",
        "ion_fill": "#ffffff",
        "gate": "#3f4b5b",
        "text": "#1c2430",
        "muted_text": "#6b7480",
        "chip": "#f1f3f5",
    },
    "dark": {
        "background": "#121418",
        "segment": "#2b3038",
        "junction": "#6f7782",
        "ion": "#c3cad3",
        "ion_fill": "#1b1f25",
        "gate": "#c3cad3",
        "text": "#e4e7eb",
        "muted_text": "#8b939e",
        "chip": "#1d2128",
    },
}


@dataclass(frozen=True)
class Place:
    """Name a position as a fraction of the way along one segment."""

    segment: int
    fraction: float


@dataclass(frozen=True)
class SegmentEnd:
    """Locate one segment end.

    The end sits at ``offset`` from junction ``junction``. An end without a
    junction sits at ``offset`` itself.
    """

    junction: int | None
    offset: Coordinate = (0.0, 0.0)


@dataclass(frozen=True)
class Movement:
    """Describe one change of an ion's drawing position.

    The ion leaves its previous place at ``start`` and reaches ``target`` at
    ``end``. A junction move passes through junction ``via``. Ions that only
    shift inside their segment have no ``via`` junction.
    """

    start: int
    end: int
    target: Place
    via: int | None = None


@dataclass(frozen=True)
class IonTrack:
    """Store the initial place and all movements of one ion."""

    ion: int
    initial: Place
    movements: tuple[Movement, ...]

    @cached_property
    def movement_starts(self) -> list[int]:
        """The start time of each movement."""
        return [movement.start for movement in self.movements]


@dataclass(frozen=True)
class GateUse:
    """Store the ions and processing zone of one scheduled gate."""

    ions: tuple[int, ...]
    processing_zone: str | None
    start: int
    end: int


@dataclass(frozen=True)
class Layer:
    """Store the actions that start at one schedule time."""

    start: int
    end: int
    description: str
    gates: tuple[GateUse, ...]


@dataclass(frozen=True)
class SegmentDrawing:
    """Store one segment line and its processing zones."""

    segment_id: str
    start: SegmentEnd
    end: SegmentEnd
    capacity: int
    processing_zones: tuple[str, ...]


@dataclass(frozen=True)
class ProcessingZoneDrawing:
    """Store the marker place of one processing zone."""

    zone_id: str
    place: Place


@dataclass(frozen=True)
class Geometry:
    """Hold the coordinates that follow from the junction positions.

    ``y`` points up. ``bounds`` is ``(min_x, min_y, max_x, max_y)`` and
    includes room for labels around the drawing.
    """

    junctions: tuple[Coordinate, ...]
    segments: tuple[tuple[Coordinate, Coordinate], ...]
    bounds: tuple[float, float, float, float]
    ion_radius: float

    def position(self, place: Place) -> Coordinate:
        """Return the coordinate of a place.

        Returns:
            The coordinate.
        """
        start, end = self.segments[place.segment]
        return _interpolate(start, end, place.fraction)


@dataclass(frozen=True)
class Scene:
    """Store the complete drawing data for one Grid result.

    Junction coordinates are scaled so that the larger side of the layout has
    length one. The ``y`` axis points up.
    """

    title: str
    start_time: int
    end_time: int
    junctions: tuple[tuple[str, Coordinate], ...]
    segments: tuple[SegmentDrawing, ...]
    processing_zones: tuple[ProcessingZoneDrawing, ...]
    ions: tuple[IonTrack, ...]
    layers: tuple[Layer, ...]

    def to_dict(self) -> dict[str, object]:
        """Return compact JSON-compatible drawing data for the HTML view."""
        return {
            "title": self.title,
            "start": self.start_time,
            "end": self.end_time,
            "junctions": [[junction_id, *position] for junction_id, position in self.junctions],
            "segments": [
                [
                    segment.segment_id,
                    [segment.start.junction, *segment.start.offset],
                    [segment.end.junction, *segment.end.offset],
                    segment.capacity,
                    list(segment.processing_zones),
                ]
                for segment in self.segments
            ],
            "processing_zones": [
                [zone.zone_id, zone.place.segment, zone.place.fraction] for zone in self.processing_zones
            ],
            "ions": [
                [
                    track.ion,
                    track.initial.segment,
                    track.initial.fraction,
                    [
                        [
                            movement.start,
                            movement.end,
                            movement.target.segment,
                            movement.target.fraction,
                            *(() if movement.via is None else (movement.via,)),
                        ]
                        for movement in track.movements
                    ],
                ]
                for track in self.ions
            ],
            "layers": [
                [
                    layer.start,
                    layer.end,
                    layer.description,
                    [[list(gate.ions), gate.processing_zone, gate.start, gate.end] for gate in layer.gates],
                ]
                for layer in self.layers
            ],
        }

    @cached_property
    def geometry(self) -> Geometry:
        """Coordinates for the stored junction positions."""
        return geometry(self, [position for _junction_id, position in self.junctions])

    def ion_positions(self, time: float) -> dict[int, Coordinate]:
        """Return each ion's drawing position at a schedule time.

        Returns:
            Drawing coordinates keyed by ion ID.
        """
        return {track.ion: track_position(track, time, self.geometry) for track in self.ions}

    def gate_ions(self, time: float) -> frozenset[int]:
        """Return the ions that take part in a running gate.

        Returns:
            The ion IDs.
        """
        return frozenset(ion for gate in self.running_gates(time) for ion in gate.ions)

    def active_processing_zones(self, time: float) -> frozenset[str]:
        """Return the processing zones that run a gate.

        Returns:
            The processing-zone IDs.
        """
        return frozenset(gate.processing_zone for gate in self.running_gates(time) if gate.processing_zone is not None)

    def running_gates(self, time: float) -> tuple[GateUse, ...]:
        """Return the gates that run at a schedule time.

        A gate runs from its start time until just before its end time. A gate
        without duration runs only at its start time.

        Returns:
            The running gates.
        """
        gates = self._gates
        index = bisect_right(self._gate_starts, time)
        earliest_start = time - self._longest_gate
        running: list[GateUse] = []
        while index > 0 and gates[index - 1].start >= earliest_start:
            index -= 1
            gate = gates[index]
            if _is_running(gate.start, gate.end, time):
                running.append(gate)
        return tuple(reversed(running))

    def current_layer(self, time: float) -> Layer | None:
        """Return the latest started action layer if it still runs.

        Returns:
            The running layer, or ``None`` between layers.
        """
        index = bisect_right(self._layer_starts, time) - 1
        if index < 0:
            return None
        layer = self.layers[index]
        return layer if _is_running(layer.start, layer.end, time) else None

    @cached_property
    def _gates(self) -> tuple[GateUse, ...]:
        return tuple(gate for layer in self.layers for gate in layer.gates)

    @cached_property
    def _gate_starts(self) -> list[int]:
        return [gate.start for gate in self._gates]

    @cached_property
    def _longest_gate(self) -> int:
        return max((gate.end - gate.start for gate in self._gates), default=0)

    @cached_property
    def _layer_starts(self) -> list[int]:
        return [layer.start for layer in self.layers]


@dataclass(frozen=True)
class PanelLayout:
    """Map scene coordinates to pixels inside one panel."""

    left: float
    top: float
    width: float
    height: float
    scale: float
    origin: Coordinate
    bounds: tuple[float, float, float, float]
    ion_radius: float
    font_size: float
    header_height: float
    padding: float

    def point(self, coordinate: Coordinate) -> Coordinate:
        """Return the pixel position of a scene coordinate, with ``y`` pointing down.

        Returns:
            The pixel position.
        """
        return (
            self.origin[0] + (coordinate[0] - self.bounds[0]) * self.scale,
            self.origin[1] + (self.bounds[3] - coordinate[1]) * self.scale,
        )


def geometry(scene: Scene, junctions: Sequence[Coordinate]) -> Geometry:
    """Compute segment ends, the ion radius, and the drawing bounds.

    The ion radius keeps the ions of a full segment apart. The bounds leave
    room for labels.

    Returns:
        The geometry.
    """

    def locate(end: SegmentEnd) -> Coordinate:
        if end.junction is None:
            return end.offset
        x, y = junctions[end.junction]
        return (x + end.offset[0], y + end.offset[1])

    segments = tuple((locate(segment.start), locate(segment.end)) for segment in scene.segments)
    points = [*junctions, *(point for ends in segments for point in ends)]
    min_x = min(x for x, _y in points)
    min_y = min(y for _x, y in points)
    max_x = max(x for x, _y in points)
    max_y = max(y for _x, y in points)
    size = max(max_x - min_x, max_y - min_y) or 1.0
    spacings = [
        math.dist(start, end) / (segment.capacity + 1)
        for segment, (start, end) in zip(scene.segments, segments, strict=True)
        if math.dist(start, end) > 0
    ]
    ion_radius = min(_ION_RADIUS_PER_SPACING * min(spacings), _MAX_ION_RADIUS * size) if spacings else size / 40
    margin = _MARGIN_PER_ION_RADIUS * ion_radius
    return Geometry(
        tuple(junctions),
        segments,
        (min_x - margin, min_y - margin, max_x + margin, max_y + margin),
        ion_radius,
    )


def build_scene(
    result: CompilationResult,
    junction_coordinates: JunctionCoordinates | None,
    title: str = "",
) -> Scene:
    """Replay a Grid result and collect its drawing data.

    Returns:
        The scene.

    Raises:
        TypeError: If the result does not use a Grid architecture and state.
    """
    architecture = result.architecture
    if not isinstance(architecture, GridArchitecture) or not isinstance(result.initial_state, GridMachineState):
        msg = "GridVisualizer requires a Grid compilation result"
        raise TypeError(msg)
    raw = junction_positions(architecture, junction_coordinates)
    junction_ids = [junction.junction_id for junction in architecture.junctions]
    points = [raw[junction_id] for junction_id in junction_ids]
    min_x = min((x for x, _y in points), default=0.0)
    min_y = min((y for _x, y in points), default=0.0)
    size = max(
        max((x for x, _y in points), default=0.0) - min_x,
        max((y for _x, y in points), default=0.0) - min_y,
    )
    size = size or 1.0
    junctions = tuple(
        (junction_id, _rounded(((x - min_x) / size, (y - min_y) / size)))
        for junction_id, (x, y) in zip(junction_ids, points, strict=True)
    )
    segments = _segment_drawings(architecture, junction_ids, [position for _id, position in junctions])
    tracks, layers = _replay(
        result, architecture, {junction_id: index for index, junction_id in enumerate(junction_ids)}
    )
    return Scene(
        title=title,
        start_time=result.start_time,
        end_time=result.end_time,
        junctions=junctions,
        segments=segments,
        processing_zones=tuple(
            ProcessingZoneDrawing(zone_id, Place(index, _round((rank + 1) / (len(segment.processing_zones) + 1))))
            for index, segment in enumerate(segments)
            for rank, zone_id in enumerate(segment.processing_zones)
        ),
        ions=tracks,
        layers=layers,
    )


def track_position(track: IonTrack, time: float, shape: Geometry) -> Coordinate:
    """Return an ion's drawing position at a schedule time.

    Between movements the ion stays at the target of its last finished
    movement. A junction move travels with constant speed along both parts of
    its path.

    Returns:
        The drawing coordinate.
    """
    movements = track.movements
    index = bisect_right(track.movement_starts, time) - 1
    if index < 0:
        return shape.position(track.initial)
    movement = movements[index]
    target = shape.position(movement.target)
    if time >= movement.end:
        return target
    source = shape.position(track.initial if index == 0 else movements[index - 1].target)
    via = None if movement.via is None else shape.junctions[movement.via]
    progress = (time - movement.start) / (movement.end - movement.start)
    return _along_path(source, via, target, progress)


def panel_layout(shape: Geometry, left: float, top: float, width: float, height: float) -> PanelLayout:
    """Fit a geometry into one panel below the header.

    Both renderers use this function so that their geometry agrees.

    Returns:
        The pixel layout.
    """
    font_size = max(11, min(17, _round_half_up(min(width, height) * 0.026)))
    header_height = _round_half_up(font_size * 3.6)
    padding = _round_half_up(min(width, height) * 0.03)
    available_width = max(width - 2 * padding, 1.0)
    available_height = max(height - header_height - 2 * padding, 1.0)
    min_x, min_y, max_x, max_y = shape.bounds
    extent_x = max(max_x - min_x, 1e-9)
    extent_y = max(max_y - min_y, 1e-9)
    scale = min(available_width / extent_x, available_height / extent_y)
    origin = (
        left + padding + (available_width - extent_x * scale) / 2,
        top + header_height + padding + (available_height - extent_y * scale) / 2,
    )
    ion_radius = min(shape.ion_radius * scale, 0.045 * min(width, height))
    return PanelLayout(
        left, top, width, height, scale, origin, shape.bounds, ion_radius, font_size, header_height, padding
    )


def label_normal(start: Coordinate, end: Coordinate) -> Coordinate:
    """Return the unit direction from a segment to its processing-zone labels.

    Labels go above a segment, or to the left of a vertical segment. Hardware
    IDs use the other side.

    Returns:
        The direction with ``y`` pointing up.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    normal = (0.0, 1.0) if length == 0 else (-dy / length, dx / length)
    if normal[1] < 0 or (normal[1] == 0 and normal[0] > 0):
        normal = (-normal[0], -normal[1])
    return normal


def ion_label_font_size(ion_radius: float, label: str) -> float:
    """Return a font size that fits an ion label inside its colored ring.

    Returns:
        The font size in pixels.
    """
    return min(0.85 * ion_radius, 2.3 * ion_radius / max(len(label), 2))


def header_lines(scene: Scene, time: float) -> tuple[str, str, str]:
    """Return the header texts for a scene at a schedule time.

    Returns:
        The title, the time, and the running layer description.
    """
    shown_time = min(max(time, scene.start_time), scene.end_time)
    layer = scene.current_layer(shown_time)
    return scene.title, f"t {shown_time:.1f} / {scene.end_time}", "" if layer is None else layer.description


def truncate(text: str, width: float, font_size: float) -> str:
    """Shorten text to an estimated pixel width.

    Returns:
        The text, shortened with an ellipsis if necessary.
    """
    limit = max(int(width / (0.55 * font_size)), 1)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def junction_positions(
    architecture: GridArchitecture,
    supplied: JunctionCoordinates | None,
) -> dict[str, Coordinate]:
    """Return validated explicit coordinates or a deterministic automatic layout.

    Returns:
        Coordinates keyed by junction ID.

    Raises:
        TypeError: If an explicit key or coordinate has the wrong type.
        ValueError: If explicit coordinates are incomplete or invalid.
    """
    if supplied is not None:
        coordinates: dict[str, Coordinate] = {}
        for raw_junction, raw_coordinate in supplied.items():
            junction_id = raw_junction.junction_id if isinstance(raw_junction, Junction) else raw_junction
            if not isinstance(junction_id, str):
                msg = "junction coordinate keys must be junction IDs or Junction values"
                raise TypeError(msg)
            if junction_id in coordinates:
                msg = f"junction coordinates contain duplicate identity {junction_id!r}"
                raise ValueError(msg)
            coordinates[junction_id] = _coordinate(raw_coordinate, junction_id)
        expected = {junction.junction_id for junction in architecture.junctions}
        missing = sorted(expected.difference(coordinates))
        unknown = sorted(set(coordinates).difference(expected))
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown: {', '.join(unknown)}")
            msg = f"junction coordinates must match the architecture ({'; '.join(details)})"
            raise ValueError(msg)
        return coordinates

    grid_coordinates = _coordinates_from_grid_ids(architecture)
    if grid_coordinates is not None:
        return grid_coordinates
    return _automatic_graph_layout(architecture)


def _segment_drawings(
    architecture: GridArchitecture,
    junction_ids: Sequence[str],
    junctions: Sequence[Coordinate],
) -> tuple[SegmentDrawing, ...]:
    """Attach each segment end to its junction, or to the junction at its other end.

    An end without a junction sits at a fixed offset from the other end's
    junction. A segment without any junction keeps fixed coordinates.

    Returns:
        The segment drawings.
    """
    index_of = {junction_id: index for index, junction_id in enumerate(junction_ids)}
    attached = {
        endpoint: index_of[junction.junction_id]
        for junction in architecture.junctions
        for endpoint in junction.endpoints
    }
    spacing = _typical_length(architecture, attached, junctions)
    drawings = []
    for index, segment in enumerate(architecture.segments):
        start = attached.get(segment.start)
        end = attached.get(segment.end)
        angle = 2.0 * math.pi * index / max(len(architecture.segments), 1)
        offset = _rounded((spacing * math.cos(angle), spacing * math.sin(angle)))
        if start is None and end is None:
            first = SegmentEnd(None, _rounded((index * 2.0 * spacing, -spacing)))
            second = SegmentEnd(None, _rounded(((index * 2.0 + 1.0) * spacing, -spacing)))
        elif start is None:
            first, second = SegmentEnd(end, offset), SegmentEnd(end)
        elif end is None:
            first, second = SegmentEnd(start), SegmentEnd(start, offset)
        else:
            first, second = SegmentEnd(start), SegmentEnd(end)
        drawings.append(
            SegmentDrawing(
                segment.segment_id,
                first,
                second,
                segment.capacity,
                tuple(zone.zone_id for zone in segment.processing_zones),
            )
        )
    return tuple(drawings)


def _typical_length(
    architecture: GridArchitecture,
    attached: Mapping[SegmentEndpoint, int],
    junctions: Sequence[Coordinate],
) -> float:
    """Return the median length of segments between two junctions, as the length of open segments.

    Returns:
        The length in scene units.
    """
    lengths = sorted(
        math.dist(junctions[attached[segment.start]], junctions[attached[segment.end]])
        for segment in architecture.segments
        if segment.start in attached and segment.end in attached
    )
    lengths = [length for length in lengths if length > 0]
    return lengths[len(lengths) // 2] if lengths else 0.25


def _replay(
    result: CompilationResult,
    architecture: GridArchitecture,
    junction_index: Mapping[str, int],
) -> tuple[tuple[IonTrack, ...], tuple[Layer, ...]]:
    """Replay the schedule layer by layer and record movements and gates.

    Returns:
        The ion tracks and the action layers.
    """
    segment_index = {segment.segment_id: index for index, segment in enumerate(architecture.segments)}
    schedule = result.schedule
    state = schedule.initial_state
    initial = _state_places(architecture, state, segment_index)
    movements: dict[int, list[Movement]] = {ion: [] for ion in initial}
    places = dict(initial)
    layers: list[Layer] = []
    for start_time, grouped_actions in groupby(schedule.scheduled_actions, key=lambda action: action.start_time):
        actions = tuple(grouped_actions)
        state = architecture.apply_layer(state, actions)
        after = _state_places(architecture, state, segment_index)
        waypoints, transport_end = _junction_waypoints(architecture, actions, junction_index)
        for ion, target in after.items():
            if target == places[ion]:
                continue
            _add_movement(movements[ion], Movement(start_time, transport_end, target, waypoints.get(ion)))
            places[ion] = target
        layers.append(
            Layer(
                start_time,
                max(action.end_time for action in actions),
                "; ".join(_describe(action) for action in actions),
                _gate_uses(actions),
            )
        )
    tracks = tuple(IonTrack(ion, initial[ion], tuple(movements[ion])) for ion in sorted(initial))
    return tracks, tuple(layers)


def _add_movement(movements: list[Movement], movement: Movement) -> None:
    """Append a movement and keep the ion's movements free of overlap.

    A segment shift that starts while the ion still moves extends the running
    movement to the new target. A junction move that starts early makes the
    running movement finish when the junction move starts.
    """
    if movements and movements[-1].end > movement.start:
        running = movements[-1]
        if movement.via is None:
            movements[-1] = Movement(running.start, max(running.end, movement.end), movement.target, running.via)
            return
        movements[-1] = Movement(running.start, movement.start, running.target, running.via)
    movements.append(movement)


def _state_places(
    architecture: GridArchitecture,
    state: GridMachineState,
    segment_index: Mapping[str, int],
) -> dict[int, Place]:
    """Place each ion along its segment in canonical occupancy order.

    Returns:
        Places keyed by ion ID.
    """
    places: dict[int, Place] = {}
    for segment in architecture.segments:
        ions = state.occupants(segment.segment_id)
        for index, ion in enumerate(ions):
            places[ion] = Place(segment_index[segment.segment_id], _round((index + 1) / (len(ions) + 1)))
    return places


def _junction_waypoints(
    architecture: GridArchitecture,
    actions: Sequence[ScheduledAction[Any]],
    junction_index: Mapping[str, int],
) -> tuple[dict[int, int], int]:
    """Find the crossed junction for each transported ion.

    Returns:
        Junction indices keyed by transported ion ID, and the time at which the
        transport of the layer ends.
    """
    waypoints: dict[int, int] = {}
    transport_end = actions[0].start_time
    for scheduled_action in actions:
        action = scheduled_action.action
        moves = action.moves if isinstance(action, Cycle) else (action,) if isinstance(action, JunctionMove) else ()
        for move in moves:
            junction = junction_index[architecture.junction_for(move.source).junction_id]
            waypoints.update(dict.fromkeys(move.ions, junction))
            transport_end = max(transport_end, scheduled_action.end_time)
    return waypoints, transport_end


def _gate_uses(actions: Sequence[ScheduledAction[Any]]) -> tuple[GateUse, ...]:
    """Collect the gates of one action layer.

    Returns:
        The gate ions, processing zones, and intervals.
    """
    return tuple(
        GateUse(tuple(item.action.ions), item.processing_zone_id, item.start_time, item.end_time)
        for item in actions
        if isinstance(item.action, GateAction)
    )


def _describe(scheduled_action: ScheduledAction[Action]) -> str:
    """Return a short text for one scheduled action.

    Returns:
        The action description.
    """
    action = scheduled_action.action
    if not isinstance(action, GateAction):
        return str(action)
    name = action.gate_name if isinstance(action, GlobalGate) else action.circuit_name or type(action).__name__
    ions = ",".join(f"q{ion}" for ion in action.ions)
    zone = "" if scheduled_action.processing_zone_id is None else f" @ {scheduled_action.processing_zone_id}"
    return f"{name} {ions}{zone}"


def _coordinate(raw_coordinate: Sequence[float], junction_id: str) -> Coordinate:
    """Validate and normalize one coordinate pair.

    Returns:
        The coordinate as a pair of floats.

    Raises:
        TypeError: If a coordinate value is not numeric.
        ValueError: If the coordinate does not contain two finite values.
    """
    if isinstance(raw_coordinate, str) or len(raw_coordinate) != 2:
        msg = f"coordinate for junction {junction_id!r} must contain two numbers"
        raise ValueError(msg)
    x, y = raw_coordinate
    if isinstance(x, bool) or not isinstance(x, int | float) or isinstance(y, bool) or not isinstance(y, int | float):
        msg = f"coordinate for junction {junction_id!r} must contain two numbers"
        raise TypeError(msg)
    if not math.isfinite(x) or not math.isfinite(y):
        msg = f"coordinate for junction {junction_id!r} must be finite"
        raise ValueError(msg)
    return float(x), float(y)


def _coordinates_from_grid_ids(architecture: GridArchitecture) -> dict[str, Coordinate] | None:
    """Recover the natural layout used by rectangular grid generators.

    Returns:
        Coordinates keyed by junction ID, or ``None`` for other ID schemes.
    """
    coordinates: dict[str, Coordinate] = {}
    grid_indices: dict[str, tuple[int, int]] = {}
    for junction in architecture.junctions:
        match = _GRID_JUNCTION_ID.fullmatch(junction.junction_id)
        if match is None:
            return None
        row, column = (int(value) for value in match.groups())
        grid_indices[junction.junction_id] = (row, column)
        coordinates[junction.junction_id] = (float(column), float(-row))
    endpoint_junction = {
        endpoint: junction.junction_id for junction in architecture.junctions for endpoint in junction.endpoints
    }
    actual_edges: set[frozenset[str]] = set()
    for segment in architecture.segments:
        start = endpoint_junction.get(segment.start)
        end = endpoint_junction.get(segment.end)
        if start is None or end is None:
            return None
        actual_edges.add(frozenset((start, end)))
    rows = {row for row, _column in grid_indices.values()}
    columns = {column for _row, column in grid_indices.values()}
    expected_nodes = {f"j:{row}:{column}" for row in rows for column in columns}
    if expected_nodes != set(grid_indices):
        return None
    expected_edges = {
        frozenset((f"j:{row}:{column}", f"j:{row}:{column + 1}"))
        for row in rows
        for column in columns
        if column + 1 in columns
    }
    expected_edges.update(
        frozenset((f"j:{row}:{column}", f"j:{row + 1}:{column}"))
        for row in rows
        for column in columns
        if row + 1 in rows
    )
    if actual_edges != expected_edges:
        return None
    return coordinates


def _automatic_graph_layout(architecture: GridArchitecture) -> dict[str, Coordinate]:
    """Lay out an arbitrary topology with planar or force-directed placement.

    Returns:
        Coordinates keyed by junction ID.
    """
    import networkx as nx  # ruff: ignore[import-outside-top-level]

    graph = nx.Graph()
    graph.add_nodes_from(junction.junction_id for junction in architecture.junctions)
    endpoint_junction = {
        endpoint: junction.junction_id for junction in architecture.junctions for endpoint in junction.endpoints
    }
    for segment in architecture.segments:
        start = endpoint_junction.get(segment.start)
        end = endpoint_junction.get(segment.end)
        if start is not None and end is not None:
            graph.add_edge(start, end)
    try:
        positions = nx.planar_layout(graph)
    except nx.NetworkXException:
        positions = nx.spring_layout(graph, seed=0)
    return {junction_id: (float(position[0]), float(position[1])) for junction_id, position in positions.items()}


def _along_path(source: Coordinate, via: Coordinate | None, target: Coordinate, progress: float) -> Coordinate:
    """Interpolate directly or with constant speed through a junction.

    Returns:
        The interpolated drawing coordinate.
    """
    if via is None:
        return _interpolate(source, target, progress)
    first = math.dist(source, via)
    second = math.dist(via, target)
    total = first + second
    if total == 0:
        return target
    travelled = progress * total
    if travelled <= first:
        return _interpolate(source, via, travelled / first) if first > 0 else via
    return _interpolate(via, target, (travelled - first) / second)


def _interpolate(start: Coordinate, end: Coordinate, progress: float) -> Coordinate:
    """Find one point between two drawing coordinates.

    Returns:
        The interpolated drawing coordinate.
    """
    return (start[0] + progress * (end[0] - start[0]), start[1] + progress * (end[1] - start[1]))


def _is_running(start: int, end: int, time: float) -> bool:
    """Return whether an interval runs at a time; a zero-length interval runs at its start."""
    return start <= time < end or start == time


def _rounded(position: Coordinate) -> Coordinate:
    """Round a coordinate to keep the HTML data small.

    Returns:
        The rounded coordinate.
    """
    return (_round(position[0]), _round(position[1]))


def _round(value: float) -> float:
    """Round one value to the stored precision.

    Returns:
        The rounded value.
    """
    return round(value, _DIGITS) + 0.0


def _round_half_up(value: float) -> int:
    """Round like JavaScript's ``Math.round`` so that both renderers agree.

    Returns:
        The nearest integer, with halves rounded up.
    """
    return math.floor(value + 0.5)
