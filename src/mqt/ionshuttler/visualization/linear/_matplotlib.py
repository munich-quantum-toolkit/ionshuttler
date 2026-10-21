# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Draw Linear scenes with Matplotlib for static figures and video files.

The drawing creates every artist once. Each new schedule time only moves ions
and changes colors and texts, so long videos do not rebuild the figure. The
layout, colors, and labels follow the browser drawing in ``draw.js``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._player.figure_style import AMBIENT_GLOW, LINE_GLOW, add_ambient_halo, add_halos, move_halos
from .._player.style import (
    COLORS,
    RING_SHADE,
    ZONE_PALETTE,
    distinct_color,
    ion_colors,
    ion_label_font_size,
    legend,
    shade,
    truncate,
)
from .._player.video import DPI, points
from ._scene import (
    SITE_TICK,
    SWAP_LIFT,
    field_points,
    header_lines,
    panel_layout,
    rail_runs,
    rail_width,
    site_tick_width,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    from .._player.style import ColorChange
    from ._scene import GateUse, PanelLayout, Scene


class LinearDrawing:
    """Draw one or more scenes from top to bottom and update them for new times."""

    def __init__(
        self,
        figure: Figure,
        scenes: Sequence[Scene],
        *,
        theme: str,
        show_ion_labels: bool,
        show_processing_zone_labels: bool,
        show_site_numbers: bool,
        show_field_profile: bool,
        ion_color_mode: str,
        ion_color_changes: Mapping[int, Sequence[ColorChange]],
        zone_colors: Mapping[str, str],
        marker_size: float,
    ) -> None:
        """Create all artists for the scenes at their start times."""
        self.figure = figure
        self.colors = COLORS["dark" if theme == "dark" else "light"]
        width = figure.get_figwidth() * DPI
        height = figure.get_figheight() * DPI
        figure.set_facecolor(self.colors["background"])
        self.axis: Axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        self.axis.set_xlim(0, width)
        self.axis.set_ylim(height, 0)
        self.axis.set_axis_off()
        panel_height = height / len(scenes)
        style = _Style(
            show_ion_labels=show_ion_labels,
            show_processing_zone_labels=show_processing_zone_labels,
            show_site_numbers=show_site_numbers,
            show_field_profile=show_field_profile,
            ion_color_mode=ion_color_mode,
            ion_color_changes=ion_color_changes,
            zone_colors=zone_colors,
        )
        self.panels = []
        for index, scene in enumerate(scenes):
            layout = panel_layout(scene, 0, index * panel_height, width, panel_height, marker_size)
            if index > 0:
                # Comparison panels stack from top to bottom, divided by a line.
                self.axis.plot(
                    [layout.padding, width - layout.padding],
                    [round(index * panel_height) + 0.5] * 2,
                    color=self.colors["chip"],
                    linewidth=points(1),
                    zorder=0,
                )
            self.panels.append(_PanelArtists(self.axis, scene, layout, self.colors, style))
        self.draw_at(min(scene.start_time for scene in scenes))

    def draw_at(self, time: float) -> list[Artist]:
        """Move ions and update gates and texts for a schedule time.

        Returns:
            The artists that changed.
        """
        return [artist for panel in self.panels for artist in panel.update(time)]


@dataclass(frozen=True, kw_only=True)
class _Style:
    """Hold the display choices that every panel shares."""

    show_ion_labels: bool
    show_processing_zone_labels: bool
    show_site_numbers: bool
    show_field_profile: bool
    ion_color_mode: str
    ion_color_changes: Mapping[int, Sequence[ColorChange]]
    zone_colors: Mapping[str, str]


class _PanelArtists:
    """Own the artists of one scene."""

    def __init__(self, axis: Axes, scene: Scene, layout: PanelLayout, colors: dict[str, str], style: _Style) -> None:
        from matplotlib.collections import LineCollection  # ruff: ignore[import-outside-top-level]

        self.scene = scene
        self.layout = layout
        self.colors = colors
        self.style = style
        radius = layout.ion_radius
        self.zone_color = {
            zone_id: style.zone_colors.get(zone_id, distinct_color(index, ZONE_PALETTE))
            for index, (zone_id, _sites) in enumerate(scene.processing_zones)
        }
        if style.show_field_profile and scene.field is not None:
            self._add_field(axis)

        # The chain: wider and colored in processing zones.
        runs = rail_runs(scene)
        self.zone_runs = [(first, last, zone) for first, last, zone in runs if zone is not None]
        zone_lines = [self._run(first, last) for first, last, _zone in self.zone_runs]
        zone_width = rail_width(radius, in_zone=True)
        # Processing zones glow faintly in their own color, and fully while they run a gate.
        self.glow: LineCollection = LineCollection(
            zone_lines,
            linewidths=points(2.2 * zone_width),
            colors="none",
            alpha=LINE_GLOW,
            capstyle="round",
            zorder=1,
        )
        axis.add_collection(self.glow)
        axis.add_collection(
            LineCollection(
                zone_lines,
                linewidths=points(2.2 * zone_width),
                colors=[self.zone_color[zone] for _first, _last, zone in self.zone_runs],
                alpha=LINE_GLOW * AMBIENT_GLOW,
                capstyle="round",
                zorder=1,
            )
        )
        plain_runs = [self._run(first, last) for first, last, zone in runs if zone is None]
        axis.add_collection(
            LineCollection(
                [*plain_runs, *zone_lines],
                linewidths=[points(rail_width(radius, in_zone=False))] * len(plain_runs)
                + [points(zone_width)] * len(zone_lines),
                colors=[colors["segment"]] * len(plain_runs)
                + [self.zone_color[zone] for _first, _last, zone in self.zone_runs],
                capstyle="round",
                zorder=1,
            )
        )
        # Each site is a tick across the chain, like the marks of a ruler.
        tick = SITE_TICK * radius
        axis.add_collection(
            LineCollection(
                [((x, y - tick), (x, y + tick)) for x, y in (layout.point(site) for site in range(scene.num_sites))],
                linewidths=points(site_tick_width(radius)),
                colors=colors["junction"],
                capstyle="butt",
                zorder=2,
            )
        )
        self._add_site_labels(axis)

        # Gates under the ions: a band behind global gates, a bar between the
        # ions of two-ion gates, and a glow around every gate ion.
        self.bands: LineCollection = LineCollection(
            [], linewidths=points(2.8 * radius), alpha=0.14, capstyle="round", zorder=3
        )
        axis.add_collection(self.bands)
        self.bars: LineCollection = LineCollection(
            [], linewidths=points(0.9 * radius), alpha=0.35, capstyle="round", zorder=3
        )
        axis.add_collection(self.bars)
        self.ion_order = [track.ion for track in scene.ions]
        start_points = list(itertools.starmap(layout.point, scene.ion_positions(scene.start_time).values()))
        # Every ion outside a running gate glows faintly in its ring color.
        self.ambient_halo = add_ambient_halo(axis, radius, 3)
        self.halos = add_halos(axis, radius, 3)
        ring_width = max(1.5, 0.28 * radius)
        self.ions: PathCollection = axis.scatter(
            [x for x, _y in start_points],
            [y for _x, y in start_points],
            s=points(2 * radius - ring_width) ** 2,
            linewidths=points(ring_width),
            zorder=4,
        )
        # Matplotlib has no radial stroke. The ring is a little darker, and its
        # inner half in the ion's own ring color stands in for the shade toward
        # the outer edge in the browser view.
        self.inner_rings: PathCollection = axis.scatter(
            [x for x, _y in start_points],
            [y for _x, y in start_points],
            s=points(2 * radius - 1.5 * ring_width) ** 2,
            facecolor="none",
            linewidths=points(0.5 * ring_width),
            zorder=4,
        )
        self.ion_labels: list[Text] = []
        if style.show_ion_labels:
            for ion in self.ion_order:
                label = f"q{ion}"
                label_size = ion_label_font_size(radius, label)
                self.ion_labels.append(
                    axis.text(
                        0,
                        0,
                        label,
                        fontsize=points(label_size),
                        fontweight="semibold",
                        ha="center",
                        va="center",
                        zorder=5,
                        visible=label_size >= 6,
                    )
                )
        # Gate names above the running gates. The pool holds one text for each
        # gate that can run at the same time.
        name_size = max(9.0, 0.72 * layout.font_size)
        self.name_y = layout.chain_y - (SWAP_LIFT + 1) * radius - 0.5 * name_size - 2
        most_running = max((len(scene.running_gates(gate.start)) for gate in scene.gates), default=0)
        self.gate_names: list[Text] = [
            axis.text(
                0,
                self.name_y,
                "",
                fontsize=points(name_size),
                fontweight="semibold",
                ha="center",
                va="center",
                zorder=6,
                visible=False,
            )
            for _ in range(most_running)
        ]
        if style.ion_color_mode == "custom":
            self._add_legend(axis)
        font_size = layout.font_size
        left = layout.left + layout.padding
        top = layout.top + layout.padding * 0.6
        self.title: Text = axis.text(
            left,
            top,
            "",
            color=colors["text"],
            fontsize=points(font_size),
            fontweight="semibold",
            va="top",
            zorder=7,
        )
        self.time: Text = axis.text(
            layout.left + layout.width - layout.padding,
            top,
            "",
            color=colors["muted_text"],
            fontsize=points(0.9 * font_size),
            ha="right",
            va="top",
            zorder=7,
        )
        self.description: Text = axis.text(
            left,
            # Without a title, the action line moves up beside the time.
            top + (1.55 if scene.title else 0.12) * font_size,
            "",
            color=colors["muted_text"],
            fontsize=points(0.85 * font_size),
            va="top",
            zorder=7,
        )

    def update(self, time: float) -> list[Artist]:
        """Update the dynamic artists for a schedule time.

        Returns:
            The changed artists.
        """
        scene = self.scene
        style = self.style
        layout = self.layout
        shown_time = min(max(time, scene.start_time), scene.end_time)
        gates = scene.running_gates(shown_time)
        positions = scene.ion_positions(shown_time)
        point_of = {ion: layout.point(*positions[ion]) for ion in self.ion_order}
        points_in_order = [point_of[ion] for ion in self.ion_order]
        bands: list[tuple[tuple[float, float], tuple[float, float]]] = []
        band_colors: list[str] = []
        bars: list[tuple[tuple[float, float], tuple[float, float]]] = []
        bar_colors: list[str] = []
        glows: dict[int, str] = {}
        for gate in gates:
            if gate.end == gate.start:
                continue
            color = self._gate_color(gate)
            gate_points = [point_of[ion] for ion in gate.ions if ion in point_of]
            if not gate_points:
                continue
            if gate.is_global:
                xs = [x for x, _y in gate_points]
                bands.append(((min(xs), layout.chain_y), (max(xs), layout.chain_y)))
                band_colors.append(color)
            elif len(gate_points) == 2:
                bars.append((gate_points[0], gate_points[1]))
                bar_colors.append(color)
            glows.update(dict.fromkeys(gate.ions, color))
        self.bands.set_segments(bands)
        self.bands.set_color(band_colors)
        self.bars.set_segments(bars)
        self.bars.set_color(bar_colors)
        styles = [
            ion_colors(
                ion,
                shown_time,
                style.ion_color_mode,
                self.colors["ion"],
                style.ion_color_changes,
                self.colors["ion_fill"],
            )
            for ion in self.ion_order
        ]
        self.ions.set_offsets(points_in_order)
        move_halos(self.halos, points_in_order, [glows.get(ion, "none") for ion in self.ion_order])
        self.ions.set_edgecolor([shade(border, RING_SHADE) for border, _fill, _text in styles])
        self.ions.set_facecolor([fill for _border, fill, _text in styles])
        self.inner_rings.set_offsets(points_in_order)
        self.inner_rings.set_edgecolor([border for border, _fill, _text in styles])
        self.ambient_halo.set_offsets(points_in_order)
        self.ambient_halo.set_facecolor([
            "none" if ion in glows else border
            for ion, (border, _fill, _text) in zip(self.ion_order, styles, strict=True)
        ])
        for text, point, (_border, _fill, text_color) in zip(self.ion_labels, points_in_order, styles, strict=False):
            text.set_position(point)
            text.set_color(text_color)
        active_zones = {gate.processing_zone for gate in gates if gate.processing_zone is not None}
        self.glow.set_color([
            self.zone_color[zone] if zone in active_zones else "none" for _first, _last, zone in self.zone_runs
        ])
        for index, text in enumerate(self.gate_names):
            gate = gates[index] if index < len(gates) else None
            xs = [] if gate is None else [point_of[ion][0] for ion in gate.ions if ion in point_of]
            text.set_visible(bool(xs))
            if gate is not None and xs:
                text.set_text(gate.name)
                text.set_position(((min(xs) + max(xs)) / 2, self.name_y))
                text.set_color(self._gate_color(gate))
        title, time_text, description = header_lines(scene, time)
        text_width = layout.width - 2 * layout.padding
        self.title.set_text(truncate(title, text_width * 0.6, layout.font_size))
        self.time.set_text(time_text)
        description_width = text_width if title else text_width * 0.75
        self.description.set_text(truncate(description, description_width, 0.85 * layout.font_size))
        return [
            self.ions,
            self.inner_rings,
            self.ambient_halo,
            *self.halos,
            self.bands,
            self.bars,
            self.glow,
            self.title,
            self.time,
            self.description,
            *self.ion_labels,
            *self.gate_names,
        ]

    def _run(self, first: int, last: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the part of the chain from site ``first`` to site ``last``.

        Returns:
            The two ends of the part.
        """
        return self.layout.point(first - 0.5), self.layout.point(last + 0.5)

    def _gate_color(self, gate: GateUse) -> str:
        if gate.processing_zone is None:
            return self.colors["gate"]
        return self.zone_color.get(gate.processing_zone, self.colors["gate"])

    def _add_field(self, axis: Axes) -> None:
        """Draw the field profile behind the chain as a faint, filled smooth curve through the field of every site."""
        from matplotlib.patches import PathPatch  # ruff: ignore[import-outside-top-level]
        from matplotlib.path import Path as CurvePath  # ruff: ignore[import-outside-top-level]

        layout = self.layout
        curve = field_points(self.scene, layout)
        if len(curve) < 2:
            return
        # Quadratic curves through the midpoints between sites, as in ``draw.js``.
        vertices = [curve[0]]
        codes = [CurvePath.MOVETO]
        for (x, y), (next_x, next_y) in itertools.pairwise(curve):
            vertices += [(x, y), ((x + next_x) / 2, (y + next_y) / 2)]
            codes += [CurvePath.CURVE3, CurvePath.CURVE3]
        vertices.append(curve[-1])
        codes.append(CurvePath.LINETO)
        muted = self.colors["muted_text"]
        area = [*vertices, (curve[-1][0], layout.field_bottom), (curve[0][0], layout.field_bottom), curve[0]]
        area_codes = [*codes, CurvePath.LINETO, CurvePath.LINETO, CurvePath.CLOSEPOLY]
        axis.add_patch(PathPatch(CurvePath(area, area_codes), facecolor=muted, edgecolor="none", alpha=0.07, zorder=0))
        axis.add_patch(
            PathPatch(
                CurvePath(vertices, codes),
                facecolor="none",
                edgecolor=muted,
                linewidth=points(1.5),
                alpha=0.5,
                joinstyle="round",
                zorder=0,
            )
        )
        axis.text(
            layout.left + layout.padding,
            layout.field_top,
            "field",
            color=muted,
            fontsize=points(max(9.0, 0.72 * layout.font_size)),
            va="top",
            zorder=6,
        )

    def _add_site_labels(self, axis: Axes) -> None:
        """Write site numbers below the chain, then processing-zone names below them."""
        layout = self.layout
        scene = self.scene
        size = site_label_size(layout)
        numbers_y = layout.chain_y + (SWAP_LIFT + 1) * layout.ion_radius + 0.4 * size + 2
        if self.style.show_site_numbers:
            for site in range(0, scene.num_sites, site_label_step(scene, layout, size)):
                axis.text(
                    layout.point(site)[0],
                    numbers_y,
                    str(site),
                    color=self.colors["muted_text"],
                    fontsize=points(size),
                    ha="center",
                    va="top",
                    zorder=6,
                )
        if self.style.show_processing_zone_labels:
            zones_y = numbers_y + (1.5 * size if self.style.show_site_numbers else 0)
            for zone_id, sites in scene.processing_zones:
                if not sites:
                    continue
                axis.text(
                    layout.point((min(sites) + max(sites)) / 2)[0],
                    zones_y,
                    zone_id,
                    color=self.zone_color[zone_id],
                    fontsize=points(zone_label_size(layout)),
                    fontweight="semibold",
                    ha="center",
                    va="top",
                    zorder=6,
                )

    def _add_legend(self, axis: Axes) -> None:
        layout = self.layout
        size = max(9.0, 0.8 * layout.font_size)
        entries = legend(self.style.ion_color_changes, self.colors["ion_fill"])
        for index, (label, border, fill) in enumerate(reversed(entries)):
            x = layout.left + layout.padding + size / 2
            y = layout.top + layout.height - layout.padding - index * 1.6 * size - size / 2
            axis.scatter(
                [x], [y], s=points(size) ** 2, facecolor=fill, edgecolor=border, linewidths=points(1.5), zorder=7
            )
            axis.text(
                x + size,
                y,
                label,
                color=self.colors["text"],
                fontsize=points(size),
                va="center",
                zorder=7,
            )


def zone_label_size(layout: PanelLayout) -> float:
    """Return the font size of processing-zone names.

    Returns:
        The font size in pixels, as ``zoneLabelSize`` in ``draw.js``.
    """
    return max(11.0, 0.95 * layout.font_size)


def site_label_size(layout: PanelLayout) -> float:
    """Return the font size of site numbers and processing-zone names.

    Returns:
        The font size in pixels, as ``siteLabelSize`` in ``draw.js``.
    """
    return max(9.0, 0.72 * layout.font_size)


def site_label_step(scene: Scene, layout: PanelLayout, size: float) -> int:
    """Return how many sites lie between two written site numbers.

    Long chains number every second, third, or later site, so that numbers
    do not overlap.

    Returns:
        The step, as ``siteLabelStep`` in ``draw.js``.
    """
    widest = len(str(scene.num_sites - 1)) * 0.6 * size + 6
    return max(1, math.ceil(widest / layout.spacing))
