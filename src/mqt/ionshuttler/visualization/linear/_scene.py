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
halfway between sites 2 and 3. During a swap the two ions pass each other, one
above the chain and one below it. The browser drawing in ``draw.js`` and the
Matplotlib renderer use the same rules.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mqt.ionshuttler.core.gates import GateAction, GlobalGate
from mqt.ionshuttler.linear.actions import PhysicalSwap, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.schedule import LinearMachineState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.core.result import CompilationResult, Diagnostics
    from mqt.ionshuttler.core.schedule import ScheduledAction

# Swapping ions pass each other this many ion radii above and below the chain.
SWAP_LIFT = 1.15
# Site ticks reach this many ion radii above and below the chain.
SITE_TICK = 0.6


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
    is 1 for the ion that passes above the chain and -1 for the ion that
    passes below it.
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

    ``field`` holds the field of every site, or ``None`` when all sites have
    the same field.
    """

    title: str
    start_time: int
    end_time: int
    num_sites: int
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
        return {track.ion: track_position(track, time) for track in self.ions}

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

    Sites lie on a horizontal chain at height ``chain_y``. The field profile
    fills the panel behind the chain: its curve lies between ``field_top``,
    the highest field, and ``field_bottom``, the lowest field.
    """

    left: float
    top: float
    width: float
    height: float
    first_site_x: float
    spacing: float
    chain_y: float
    field_top: float
    field_bottom: float
    ion_radius: float
    font_size: float
    header_height: float
    padding: float

    def point(self, site: float, lift: float = 0.0) -> tuple[float, float]:
        """Return the pixel position of a site, raised by ``lift`` swap heights.

        Returns:
            The pixel position, with ``y`` pointing down.
        """
        return (self.first_site_x + site * self.spacing, self.chain_y - lift * SWAP_LIFT * self.ion_radius)


def build_scene(result: CompilationResult, title: str = "") -> Scene:
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
    diagnostics = result.diagnostics
    planned = dict(diagnostics.preferred_gate_zones) if isinstance(diagnostics, PlannedGateZones) else {}
    tracks, layers = _replay(result, dict(initial_state.positions), planned)
    actions = [item.action for item in result.schedule.scheduled_actions]
    field = tuple(architecture.field_at(site) for site in range(architecture.num_sites))
    return Scene(
        title=title,
        start_time=result.start_time,
        end_time=result.end_time,
        num_sites=architecture.num_sites,
        field=field if len(set(field)) > 1 else None,
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


def track_position(track: IonTrack, time: float) -> tuple[float, float]:
    """Return an ion's site and lift at a schedule time.

    Between movements the ion stays at the target of its last finished
    movement. A moving ion travels with constant speed along the chain. A
    swapping ion also rises and falls: its lift is ``side`` times the sine of
    the progress angle, so it is largest halfway through the swap.

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
    return source + (movement.target - source) * progress, movement.side * math.sin(math.pi * progress)


def panel_layout(
    scene: Scene,
    left: float,
    top: float,
    width: float,
    height: float,
    marker_size: float = 1.0,
) -> PanelLayout:
    """Fit the chain into one panel below the header.

    Both renderers use this function so that their geometry agrees. Sites
    share the width evenly. ``marker_size`` scales the ions, sites, and chain
    lines without moving any site.

    Returns:
        The pixel layout.
    """
    font_size = max(11, min(17, _round_half_up(min(width, height) * 0.026)))
    header_height = _round_half_up(font_size * 3.6)
    padding = _round_half_up(min(width, height) * 0.03)
    available_width = max(width - 2 * padding, 1.0)
    available_height = max(height - header_height - 2 * padding, 1.0)
    area_top = top + header_height + padding
    spacing = available_width / scene.num_sites
    ion_radius = max(min(0.3 * spacing, 0.09 * available_height, 0.05 * min(width, height)) * marker_size, 2.0)
    # The field profile fills the panel behind the chain.
    return PanelLayout(
        left=left,
        top=top,
        width=width,
        height=height,
        first_site_x=left + padding + 0.5 * spacing,
        spacing=spacing,
        chain_y=area_top + 0.45 * available_height,
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

    Returns:
        The first site, the last site, and the processing zone of each stretch.
    """
    runs: list[tuple[int, int, str | None]] = []
    for site, zone in enumerate(site_zones(scene)):
        if runs and runs[-1][2] == zone:
            runs[-1] = (runs[-1][0], site, zone)
        else:
            runs.append((site, site, zone))
    return runs


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
) -> tuple[tuple[IonTrack, ...], tuple[Layer, ...]]:
    """Replay the schedule from the initial ion sites and record movements and gates.

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
                # The ion that moves toward higher sites passes above the chain.
                for ion, source, target in (
                    (action.ion_a, action.pos_a, action.pos_b),
                    (action.ion_b, action.pos_b, action.pos_a),
                ):
                    side = 1 if target > source else -1
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


def _round_half_up(value: float) -> int:
    """Round like ``Math.round`` in JavaScript, so both renderers agree.

    Returns:
        The rounded value.
    """
    return math.floor(value + 0.5)
