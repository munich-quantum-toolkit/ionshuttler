# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Drawing data for Linear results that every Linear renderer shares.

A scene stores the trap sites, the field profile, each ion's movements, and
the action layers of one replayed schedule. Its size grows with the scheduled
actions, not with the number of drawn frames.

An ion's position is a site number along the chain: an ion at site 2.5 is
halfway between sites 2 and 3. On a ring, site numbers count modulo the number
of sites, so an ion at site 4.5 of a five-site ring is halfway between sites 4
and 0. During a swap the two ions pass each other, one on each side of the
chain. The browser drawing in ``draw.js`` and the Matplotlib renderer use the
same rules.

A chain lies on a straight line and a ring on an ellipse that fills the
panel, unless the user supplies a drawing position for every site. Pixel
positions use ``y`` pointing down. The *normal* at a site is the unit
direction that points away from a ring, or up from a chain that runs from
left to right. Swapping ions that move to the next site pass on the side of
the normal; gate names lie on that side, and site numbers and
processing-zone names on the other side.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby, pairwise
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from mqt.ionshuttler.core.gates import GateAction, GlobalGate
from mqt.ionshuttler.linear.actions import PhysicalSwap, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.schedule import LinearMachineState

if TYPE_CHECKING:
    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.core.result import CompilationResult, Diagnostics
    from mqt.ionshuttler.core.schedule import ScheduledAction
    from mqt.ionshuttler.visualization.linear.visualizer import SiteCoordinates

Coordinate: TypeAlias = tuple[float, float]

# Swapping ions pass each other this many ion radii away from the chain.
SWAP_LIFT = 1.15
# Site ticks reach this many ion radii to both sides of the chain.
SITE_TICK = 0.6
# Curved rails use this many straight pieces per site.
_CURVE_STEPS = 8
_DIGITS = 4


@runtime_checkable
class PlannedGateZones(Protocol):
    """Describe diagnostics that name a planned processing zone for some gates.

    The scene reads the field from the diagnostics of any compiler method,
    whatever their class. The view then marks each gate that ran in another
    zone than planned.
    """

    @property
    def preferred_gate_zones(self) -> tuple[tuple[int, str], ...]:
        """``(gate ID, processing-zone ID)`` pairs."""
        ...


@dataclass(frozen=True)
class Movement:
    """Describe one move of an ion to another site.

    The ion leaves its previous site at ``start`` and reaches ``target`` at
    ``end``. ``side`` is 0 for a shuttle along the chain. In a swap, ``side``
    is 1 for the ion that moves to the next site and passes on the side of the
    normal, and -1 for the ion that passes on the other side.
    """

    start: int
    end: int
    target: int
    side: int = 0


@dataclass(frozen=True)
class IonTrack:
    """Store the initial site and all movements of one ion."""

    ion: int
    initial: int
    movements: tuple[Movement, ...]

    @cached_property
    def movement_starts(self) -> list[int]:
        """The start time of each movement."""
        return [movement.start for movement in self.movements]


@dataclass(frozen=True)
class GateUse:
    """Store one scheduled gate.

    ``planned_zone`` is the processing zone that the compiler assigned to the
    gate before scheduling, if it assigned one. ``is_global`` marks a gate that
    acts on many ions at once.
    """

    gate_id: int | None
    name: str
    ions: tuple[int, ...]
    processing_zone: str | None
    planned_zone: str | None
    start: int
    end: int
    is_global: bool


@dataclass(frozen=True)
class Layer:
    """Store the actions that start at one schedule time."""

    start: int
    end: int
    description: str
    gates: tuple[GateUse, ...]


@dataclass(frozen=True)
class RunSummary:
    """Store the outcome and run statistics of one compilation.

    ``statistics`` holds the ``(label, value)`` pairs that the diagnostics
    report, with each value written as text.
    """

    status: str
    swaps: int
    shuttles: int
    gates: int
    statistics: tuple[tuple[str, str], ...]
    compile_seconds: float | None


@dataclass(frozen=True)
class Scene:
    """Store the complete drawing data for one Linear result.

    ``ring`` marks a ring architecture, whose last site neighbors site 0.
    ``coordinates`` holds the drawing position of every site, scaled so that
    the larger extent is 1 and with ``y`` pointing up, or ``None`` for the
    automatic line or ellipse. ``field`` holds the field of every site, or
    ``None`` when all sites have the same field or the sites do not lie on
    the automatic straight chain.
    """

    title: str
    start_time: int
    end_time: int
    num_sites: int
    ring: bool
    coordinates: tuple[Coordinate, ...] | None
    field: tuple[float, ...] | None
    processing_zones: tuple[tuple[str, tuple[int, ...]], ...]
    ions: tuple[IonTrack, ...]
    layers: tuple[Layer, ...]
    summary: RunSummary

    def to_dict(self) -> dict[str, object]:
        """Return compact JSON-compatible drawing data for the HTML view."""
        summary = self.summary
        return {
            "title": self.title,
            "start": self.start_time,
            "end": self.end_time,
            "sites": self.num_sites,
            "ring": self.ring,
            "coordinates": None if self.coordinates is None else [list(point) for point in self.coordinates],
            "field": None if self.field is None else list(self.field),
            "processing_zones": [[zone_id, list(sites)] for zone_id, sites in self.processing_zones],
            "ions": [
                [
                    track.ion,
                    track.initial,
                    [[movement.start, movement.end, movement.target, movement.side] for movement in track.movements],
                ]
                for track in self.ions
            ],
            "layers": [
                [
                    layer.start,
                    layer.end,
                    layer.description,
                    [
                        [
                            gate.gate_id,
                            gate.name,
                            list(gate.ions),
                            gate.processing_zone,
                            gate.planned_zone,
                            gate.start,
                            gate.end,
                            gate.is_global,
                        ]
                        for gate in layer.gates
                    ],
                ]
                for layer in self.layers
            ],
            "summary": {
                "status": summary.status,
                "swaps": summary.swaps,
                "shuttles": summary.shuttles,
                "gates": summary.gates,
                "statistics": [list(statistic) for statistic in summary.statistics],
                "compile_seconds": summary.compile_seconds,
            },
        }

    def ion_positions(self, time: float) -> dict[int, tuple[float, float]]:
        """Return each ion's site and lift at a schedule time.

        Returns:
            ``(site, lift)`` pairs keyed by ion ID; see :func:`track_position`.
        """
        ring_size = self.num_sites if self.ring else None
        return {track.ion: track_position(track, time, ring_size) for track in self.ions}

    def running_gates(self, time: float) -> tuple[GateUse, ...]:
        """Return the gates that run at a schedule time.

        A gate runs from its start time until just before its end time. A gate
        without duration runs only at its start time.

        Returns:
            The running gates.
        """
        gates = self.gates
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
    def gates(self) -> tuple[GateUse, ...]:
        """All scheduled gates in start-time order."""
        return tuple(gate for layer in self.layers for gate in layer.gates)

    @cached_property
    def _gate_starts(self) -> list[int]:
        return [gate.start for gate in self.gates]

    @cached_property
    def _longest_gate(self) -> int:
        return max((gate.end - gate.start for gate in self.gates), default=0)

    @cached_property
    def _layer_starts(self) -> list[int]:
        return [layer.start for layer in self.layers]


@dataclass(frozen=True)
class PanelLayout:
    """Map sites to pixels inside one panel.

    ``sites`` holds the pixel position of every site. ``ellipse`` holds the
    center and the two radii of the automatic ring, or ``None`` when straight
    pieces join neighboring sites. ``spacing`` is the smallest distance between
    neighboring sites. On the automatic straight chain, the field profile
    fills the panel behind the chain: its curve lies between ``field_top``,
    the highest field, and ``field_bottom``, the lowest field.
    """

    left: float
    top: float
    width: float
    height: float
    sites: tuple[Coordinate, ...]
    ring: bool
    ellipse: tuple[float, float, float, float] | None
    spacing: float
    field_top: float
    field_bottom: float
    ion_radius: float
    font_size: float
    header_height: float
    padding: float

    def point(self, site: float, lift: float = 0.0) -> Coordinate:
        """Return the pixel position of a site, moved by ``lift`` swap distances along the normal.

        Returns:
            The pixel position, with ``y`` pointing down.
        """
        return self.beside(site, lift * SWAP_LIFT * self.ion_radius)

    def beside(self, site: float, distance: float) -> Coordinate:
        """Return the pixel position ``distance`` pixels from a site along the normal.

        A negative distance selects the other side of the chain.

        Returns:
            The pixel position.
        """
        x, y = self.on_chain(site)
        if distance == 0:
            return x, y
        normal_x, normal_y = self.normal(site)
        return x + normal_x * distance, y + normal_y * distance

    def on_chain(self, site: float) -> Coordinate:
        """Return the pixel position of a site, or of a point between two neighboring sites.

        Positions before the first or after the last site of a chain continue
        its end pieces.

        Returns:
            The pixel position.
        """
        count = len(self.sites)
        if self.ellipse is not None:
            center_x, center_y, radius_x, radius_y = self.ellipse
            angle = _ring_angle(site, count)
            return center_x + radius_x * math.cos(angle), center_y + radius_y * math.sin(angle)
        if count == 1:
            x, y = self.sites[0]
            return x + site * self.spacing, y
        index, fraction = self._piece(site)
        x0, y0 = self.sites[index]
        x1, y1 = self.sites[(index + 1) % count]
        return x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction

    def normal(self, site: float) -> Coordinate:
        """Return the unit normal at a site, or between two neighboring sites.

        At a site where two straight pieces meet, the normal is the mean of
        their normals.

        Returns:
            The direction, with ``y`` pointing down.
        """
        count = len(self.sites)
        if self.ellipse is not None:
            _center_x, _center_y, radius_x, radius_y = self.ellipse
            angle = _ring_angle(site, count)
            return _unit(math.cos(angle) / radius_x, math.sin(angle) / radius_y)
        if count == 1:
            return 0.0, -1.0
        index, fraction = self._piece(site)
        if fraction != 0:
            return self._piece_normals[index]
        if self.ring or index > 0:
            before_x, before_y = self._piece_normals[index - 1]
            after_x, after_y = self._piece_normals[index]
            if math.hypot(before_x + after_x, before_y + after_y) > 1e-9:
                return _unit(before_x + after_x, before_y + after_y)
        return self._piece_normals[index]

    def _piece(self, site: float) -> tuple[int, float]:
        """Return the straight piece that contains a position and the fraction along it.

        Piece ``i`` joins site ``i`` to the next site. On a chain, positions
        beyond the ends use the end pieces, and the last site lies at the end
        of the last piece.

        Returns:
            The piece index and the fraction.
        """
        count = len(self.sites)
        if self.ring:
            site %= count
            index = min(math.floor(site), count - 1)
        else:
            index = min(max(math.floor(site), 0), count - 2)
            if site == count - 1:
                return count - 1, 0.0
        return index, site - index

    @cached_property
    def _piece_normals(self) -> tuple[Coordinate, ...]:
        """The normal of each straight piece: to the left of the direction of travel to the next site.

        A ring turns its normals outward.
        """
        count = len(self.sites)
        ends = range(count) if self.ring else range(count - 1)
        pieces = [(self.sites[index], self.sites[(index + 1) % count]) for index in ends]
        # The shoelace area is positive for a ring that runs clockwise on the screen.
        area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in pieces)
        orientation = -1.0 if self.ring and area < 0 else 1.0
        normals: list[Coordinate] = []
        for (x0, y0), (x1, y1) in pieces:
            length = math.hypot(x1 - x0, y1 - y0)
            normals.append(
                (0.0, -1.0) if length == 0 else (orientation * (y1 - y0) / length, orientation * (x0 - x1) / length)
            )
        if not self.ring:
            # The last site uses the last piece.
            normals.append(normals[-1])
        return tuple(normals)


def build_scene(
    result: CompilationResult,
    site_coordinates: SiteCoordinates | None = None,
    title: str = "",
) -> Scene:
    """Replay a Linear result and collect its drawing data.

    Returns:
        The scene.

    Raises:
        TypeError: If the result does not use a Linear architecture and state.
    """
    architecture = result.architecture
    initial_state = result.initial_state
    if not isinstance(architecture, LinearArchitecture) or not isinstance(initial_state, LinearMachineState):
        msg = "LinearVisualizer requires a Linear compilation result"
        raise TypeError(msg)
    ring = architecture.topology == "ring"
    coordinates = None if site_coordinates is None else _site_positions(architecture.num_sites, site_coordinates)
    diagnostics = result.diagnostics
    planned = dict(diagnostics.preferred_gate_zones) if isinstance(diagnostics, PlannedGateZones) else {}
    tracks, layers = _replay(result, dict(initial_state.positions), planned, architecture.num_sites if ring else None)
    actions = [item.action for item in result.schedule.scheduled_actions]
    field = tuple(architecture.field_at(site) for site in range(architecture.num_sites))
    straight_chain = not ring and coordinates is None
    return Scene(
        title=title,
        start_time=result.start_time,
        end_time=result.end_time,
        num_sites=architecture.num_sites,
        ring=ring,
        coordinates=coordinates,
        field=field if straight_chain and len(set(field)) > 1 else None,
        processing_zones=tuple(
            (zone_id, tuple(sites)) for zone_id, sites in (architecture.processing_zones or {}).items()
        ),
        ions=tracks,
        layers=layers,
        summary=RunSummary(
            status=result.status.value,
            swaps=sum(isinstance(action, PhysicalSwap) for action in actions),
            shuttles=sum(isinstance(action, Shuttle) for action in actions),
            gates=sum(isinstance(action, GateAction) for action in actions),
            statistics=() if diagnostics is None else _statistics(diagnostics),
            compile_seconds=result.wall_clock_s,
        ),
    )


def _statistics(diagnostics: Diagnostics) -> tuple[tuple[str, str], ...]:
    """Write the run statistics of the diagnostics as text.

    Returns:
        The labels and values. Fractional numbers keep four significant digits.

    Raises:
        TypeError: If a statistic is not a pair of a text label and a number or text value.
    """
    shown: list[tuple[str, str]] = []
    for statistic in diagnostics.statistics():
        if not isinstance(statistic, tuple) or len(statistic) != 2:
            msg = "diagnostics statistics must be (label, value) pairs"
            raise TypeError(msg)
        label, value = statistic
        if not isinstance(label, str) or isinstance(value, bool) or not isinstance(value, int | float | str):
            msg = "a diagnostics statistic needs a text label and a number or text value"
            raise TypeError(msg)
        shown.append((label, f"{value:.4g}" if isinstance(value, float) else str(value)))
    return tuple(shown)


def track_position(track: IonTrack, time: float, ring_size: int | None = None) -> tuple[float, float]:
    """Return an ion's site and lift at a schedule time.

    Between movements the ion stays at the target of its last finished
    movement. A moving ion travels with constant speed along the chain. A
    swapping ion also moves away from the chain and back: its lift is
    ``side`` times the sine of the progress angle, so it is largest halfway
    through the swap. On a ring of ``ring_size`` sites, a move between the
    last site and site 0 crosses the edge that joins them.

    Returns:
        The site, as a fraction between sites while moving, and the lift
        between -1 and 1.
    """
    movements = track.movements
    index = bisect_right(track.movement_starts, time) - 1
    if index < 0:
        return float(track.initial), 0.0
    movement = movements[index]
    if time >= movement.end:
        return float(movement.target), 0.0
    source = track.initial if index == 0 else movements[index - 1].target
    progress = (time - movement.start) / (movement.end - movement.start)
    lift = movement.side * math.sin(math.pi * progress)
    step = movement.target - source
    if ring_size is None:
        return source + step * progress, lift
    if abs(step) == ring_size - 1:
        step = -1 if step > 0 else 1
    return (source + step * progress) % ring_size, lift


def panel_layout(
    scene: Scene,
    left: float,
    top: float,
    width: float,
    height: float,
    marker_size: float = 1.0,
) -> PanelLayout:
    """Fit the chain into one panel below the header.

    Both renderers use this function so that their geometry agrees. On the
    automatic chain, sites share the width evenly. The automatic ring is an
    ellipse that fills the panel, with sites at equal angles clockwise from
    the top. Explicit coordinates keep their shape and fill the panel as far
    as the shape allows. Rings and explicit coordinates leave a margin for
    swapping ions and gate names. ``marker_size`` scales the ions, sites, and
    chain lines without moving any site.

    Returns:
        The pixel layout.
    """
    font_size = max(11, min(17, _round_half_up(min(width, height) * 0.026)))
    header_height = _round_half_up(font_size * 3.6)
    padding = _round_half_up(min(width, height) * 0.03)
    available_width = max(width - 2 * padding, 1.0)
    available_height = max(height - header_height - 2 * padding, 1.0)
    area_left = left + padding
    area_top = top + header_height + padding
    count = scene.num_sites
    ellipse = None
    if scene.coordinates is None and not scene.ring:
        spacing = available_width / count
        chain_y = area_top + 0.45 * available_height
        sites = tuple((area_left + (site + 0.5) * spacing, chain_y) for site in range(count))
    else:
        largest_radius = min(0.09 * available_height, 0.05 * min(width, height))
        # Room for swapping ions and gate names, which use the gate-name font size.
        margin = (SWAP_LIFT + 1) * largest_radius + max(9.0, 0.72 * font_size) + 4
        inner_width = max(available_width - 2 * margin, 1.0)
        inner_height = max(available_height - 2 * margin, 1.0)
        center_x = area_left + available_width / 2
        center_y = area_top + available_height / 2
        if scene.coordinates is None:
            ellipse = (center_x, center_y, inner_width / 2, inner_height / 2)
            sites = tuple(
                (center_x + inner_width / 2 * math.cos(angle), center_y + inner_height / 2 * math.sin(angle))
                for angle in (_ring_angle(site, count) for site in range(count))
            )
        else:
            sites = _fit(scene.coordinates, center_x, center_y, inner_width, inner_height, ring=scene.ring)
        neighbors = [
            math.dist(sites[site], sites[(site + 1) % count]) for site in range(count if scene.ring else count - 1)
        ]
        spacing = min(neighbors, default=inner_width)
    ion_radius = max(min(0.3 * spacing, 0.09 * available_height, 0.05 * min(width, height)) * marker_size, 2.0)
    # The field profile fills the panel behind the chain.
    return PanelLayout(
        left=left,
        top=top,
        width=width,
        height=height,
        sites=sites,
        ring=scene.ring,
        ellipse=ellipse,
        spacing=spacing,
        field_top=area_top + 0.04 * available_height,
        field_bottom=area_top + 0.86 * available_height,
        ion_radius=ion_radius,
        font_size=font_size,
        header_height=header_height,
        padding=padding,
    )


def field_points(scene: Scene, layout: PanelLayout) -> list[tuple[float, float]]:
    """Return the pixel point of every site on the field profile curve.

    The lowest field lies at ``field_bottom`` and the highest at ``field_top``.

    Returns:
        One point per site, or no points when the field is uniform.
    """
    if scene.field is None:
        return []
    low = min(scene.field)
    span = max(scene.field) - low
    return [
        (
            layout.point(site)[0],
            layout.field_bottom - (value - low) / span * (layout.field_bottom - layout.field_top),
        )
        for site, value in enumerate(scene.field)
    ]


def site_zones(scene: Scene) -> list[str | None]:
    """Return the processing zone of each site.

    Returns:
        One zone ID, or ``None``, per site.
    """
    zones: list[str | None] = [None] * scene.num_sites
    for zone_id, sites in scene.processing_zones:
        for site in sites:
            zones[site] = zone_id
    return zones


def rail_runs(scene: Scene) -> list[tuple[int, int, str | None]]:
    """Split the chain into stretches of neighboring sites in the same processing zone, or in none.

    On a ring, a stretch that continues from the last site to site 0 starts
    at a negative site: ``-1`` is the last site.

    Returns:
        The first site, the last site, and the processing zone of each stretch.
    """
    runs: list[tuple[int, int, str | None]] = []
    for site, zone in enumerate(site_zones(scene)):
        if runs and runs[-1][2] == zone:
            runs[-1] = (runs[-1][0], site, zone)
        else:
            runs.append((site, site, zone))
    if scene.ring and len(runs) > 1 and runs[0][2] == runs[-1][2]:
        first, _last, zone = runs.pop()
        runs[0] = (first - scene.num_sites, runs[0][1], zone)
    return runs


def rail_path(layout: PanelLayout, first: float, last: float) -> list[Coordinate]:
    """Return the pixel points of the chain from position ``first`` to position ``last``.

    The points include every site in between. The automatic ring adds
    points between sites, so that the path follows the ellipse.

    Returns:
        The points in order.
    """
    steps = 1 if layout.ellipse is None else _CURVE_STEPS
    marks = [first, *range(math.floor(first) + 1, math.ceil(last)), last]
    points = [layout.on_chain(first)]
    for start, end in pairwise(marks):
        points.extend(layout.on_chain(start + (end - start) * step / steps) for step in range(1, steps + 1))
    return points


def site_span(sites: Sequence[float], scene: Scene) -> tuple[float, float]:
    """Return the shortest stretch of the chain that contains some sites.

    On a ring, the stretch leaves out the largest gap between the sites. Its
    last site can then exceed the number of sites, as in :func:`rail_runs`.

    Returns:
        The first and the last position of the stretch.
    """
    if not scene.ring:
        return min(sites), max(sites)
    ordered = sorted(site % scene.num_sites for site in sites)
    gaps = [*(right - left for left, right in pairwise(ordered)), ordered[0] + scene.num_sites - ordered[-1]]
    widest = max(range(len(gaps)), key=gaps.__getitem__)
    if widest == len(gaps) - 1:
        return ordered[0], ordered[-1]
    return ordered[widest + 1], ordered[widest] + scene.num_sites


def site_tick_width(radius: float) -> float:
    """Return the line width of the tick that marks each site.

    Returns:
        The width in pixels.
    """
    return max(1.0, 0.12 * radius)


def rail_width(radius: float, *, in_zone: bool) -> float:
    """Return the line width of the chain at one site.

    Returns:
        The width in pixels. Processing zones draw a wider line.
    """
    return max(2.0, 0.7 * radius) if in_zone else max(1.5, 0.45 * radius)


def header_lines(scene: Scene, time: float) -> tuple[str, str, str]:
    """Return the header texts for a scene at a schedule time.

    Returns:
        The title, the time, and the running layer description.
    """
    shown_time = min(max(time, scene.start_time), scene.end_time)
    layer = scene.current_layer(shown_time)
    return scene.title, f"t {shown_time:.1f} / {scene.end_time}", "" if layer is None else layer.description


def _replay(
    result: CompilationResult,
    initial: Mapping[int, int],
    planned: Mapping[int, str],
    ring_size: int | None,
) -> tuple[tuple[IonTrack, ...], tuple[Layer, ...]]:
    """Replay the schedule from the initial ion sites and record movements and gates.

    ``ring_size`` is the number of sites of a ring, or ``None`` for a chain.

    Returns:
        The ion tracks and the action layers.
    """
    movements: dict[int, list[Movement]] = {ion: [] for ion in initial}
    layers: list[Layer] = []
    for start_time, grouped in groupby(result.schedule.scheduled_actions, key=lambda item: item.start_time):
        actions = tuple(grouped)
        for item in actions:
            action = item.action
            if isinstance(action, Shuttle):
                movements[action.ion].append(Movement(item.start_time, item.end_time, action.dst))
            elif isinstance(action, PhysicalSwap):
                # The ion that moves to the next site passes on the side of the normal.
                for ion, source, target in (
                    (action.ion_a, action.pos_a, action.pos_b),
                    (action.ion_b, action.pos_b, action.pos_a),
                ):
                    forward = target > source if ring_size is None else (target - source) % ring_size == 1
                    side = 1 if forward else -1
                    movements[ion].append(Movement(item.start_time, item.end_time, target, side))
        layers.append(
            Layer(
                start_time,
                max(item.end_time for item in actions),
                "; ".join(_describe(item) for item in actions),
                tuple(
                    GateUse(
                        item.action.gate_id,
                        _gate_name(item.action),
                        tuple(item.action.ions),
                        item.processing_zone_id,
                        None if item.action.gate_id is None else planned.get(item.action.gate_id),
                        item.start_time,
                        item.end_time,
                        isinstance(item.action, GlobalGate),
                    )
                    for item in actions
                    if isinstance(item.action, GateAction)
                ),
            )
        )
    tracks = tuple(IonTrack(ion, initial[ion], tuple(movements[ion])) for ion in sorted(initial))
    return tracks, tuple(layers)


def _describe(scheduled_action: ScheduledAction[Action]) -> str:
    """Return a short text for one scheduled action.

    Returns:
        The action description.
    """
    action = scheduled_action.action
    if isinstance(action, Shuttle):
        return f"shuttle q{action.ion} {action.src} → {action.dst}"
    if isinstance(action, PhysicalSwap):
        return f"swap q{action.ion_a} q{action.ion_b} {action.pos_a} ↔ {action.pos_b}"
    if not isinstance(action, GateAction):
        return str(action)
    ions = ",".join(f"q{ion}" for ion in action.ions)
    zone = "" if scheduled_action.processing_zone_id is None else f" @ {scheduled_action.processing_zone_id}"
    return f"{_gate_name(action)} {ions}{zone}"


def _gate_name(gate: GateAction) -> str:
    """Return the circuit name of a gate, such as ``rx`` or ``global rx``.

    Returns:
        The name.
    """
    if isinstance(gate, GlobalGate):
        return f"global {gate.gate_name}"
    return gate.circuit_name or type(gate).__name__


def _is_running(start: int, end: int, time: float) -> bool:
    return start <= time < end or start == time


def _site_positions(num_sites: int, supplied: SiteCoordinates) -> tuple[Coordinate, ...]:
    """Validate explicit site coordinates and scale them so that the larger extent is 1.

    Returns:
        One coordinate per site, with ``y`` pointing up.

    Raises:
        TypeError: If the coordinates are not a mapping or sequence, or a key
            or coordinate has the wrong type.
        ValueError: If the coordinates do not name every site exactly once or
            a coordinate is invalid.
    """
    if isinstance(supplied, Mapping):
        items = list(supplied.items())
    elif isinstance(supplied, Sequence) and not isinstance(supplied, str):
        items = list(enumerate(supplied))
    else:
        msg = "site_coordinates must be a mapping from site to coordinate or a sequence of coordinates"
        raise TypeError(msg)
    coordinates: dict[int, Coordinate] = {}
    for site, raw_coordinate in items:
        if isinstance(site, bool) or not isinstance(site, int):
            msg = "site coordinate keys must be site numbers"
            raise TypeError(msg)
        coordinates[site] = _coordinate(raw_coordinate, site)
    missing = sorted(set(range(num_sites)).difference(coordinates))
    unknown = sorted(set(coordinates).difference(range(num_sites)))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing: {', '.join(map(str, missing))}")
        if unknown:
            details.append(f"unknown: {', '.join(map(str, unknown))}")
        msg = f"site coordinates must match the architecture ({'; '.join(details)})"
        raise ValueError(msg)
    points = [coordinates[site] for site in range(num_sites)]
    min_x = min(x for x, _y in points)
    min_y = min(y for _x, y in points)
    size = max(max(x for x, _y in points) - min_x, max(y for _x, y in points) - min_y) or 1.0
    return tuple((round((x - min_x) / size, _DIGITS), round((y - min_y) / size, _DIGITS)) for x, y in points)


def _coordinate(raw_coordinate: object, site: int) -> Coordinate:
    """Validate one coordinate pair.

    Returns:
        The coordinate as a pair of floats.

    Raises:
        TypeError: If a coordinate value is not numeric.
        ValueError: If the coordinate does not contain two finite values.
    """
    if not isinstance(raw_coordinate, Sequence) or isinstance(raw_coordinate, str) or len(raw_coordinate) != 2:
        msg = f"coordinate for site {site} must contain two numbers"
        raise ValueError(msg)
    x, y = raw_coordinate
    if isinstance(x, bool) or not isinstance(x, int | float) or isinstance(y, bool) or not isinstance(y, int | float):
        msg = f"coordinate for site {site} must contain two numbers"
        raise TypeError(msg)
    if not math.isfinite(x) or not math.isfinite(y):
        msg = f"coordinate for site {site} must be finite"
        raise ValueError(msg)
    return float(x), float(y)


def _fit(
    coordinates: Sequence[Coordinate],
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    *,
    ring: bool,
) -> tuple[Coordinate, ...]:
    """Scale coordinates with ``y`` pointing up into a pixel box around a center, keeping their shape.

    The box also holds the ends of a chain, which reach half a piece beyond
    the first and the last site.

    Returns:
        The pixel positions.
    """
    shown = list(coordinates)
    if not ring and len(coordinates) > 1:
        for end, inner in ((coordinates[0], coordinates[1]), (coordinates[-1], coordinates[-2])):
            shown.append((1.5 * end[0] - 0.5 * inner[0], 1.5 * end[1] - 0.5 * inner[1]))
    min_x = min(x for x, _y in shown)
    max_x = max(x for x, _y in shown)
    min_y = min(y for _x, y in shown)
    max_y = max(y for _x, y in shown)
    scale = min(width / max(max_x - min_x, 1e-9), height / max(max_y - min_y, 1e-9))
    middle_x = (min_x + max_x) / 2
    middle_y = (min_y + max_y) / 2
    return tuple((center_x + (x - middle_x) * scale, center_y - (y - middle_y) * scale) for x, y in coordinates)


def _ring_angle(site: float, num_sites: int) -> float:
    """Return the angle of a site on the automatic ring: site 0 at the top, then clockwise on the screen.

    Returns:
        The angle in radians, for ``y`` pointing down.
    """
    return -math.pi / 2 + 2 * math.pi * site / num_sites


def _unit(x: float, y: float) -> Coordinate:
    length = math.hypot(x, y)
    return x / length, y / length


def _round_half_up(value: float) -> int:
    """Round like ``Math.round`` in JavaScript, so both renderers agree.

    Returns:
        The rounded value.
    """
    return math.floor(value + 0.5)
