# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Draw Grid scenes with Matplotlib for static figures and video files.

The drawing creates every artist once. Each new schedule time only moves ions
and changes colors and header text, so long videos do not rebuild the figure.
The layout, colors, and labels follow the browser drawing in ``draw.js``.
"""

from __future__ import annotations

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
from ._scene import hardware_id_font_size, header_lines, label_normal, panel_layout, zone_label_size

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    from .._player.style import ColorChange
    from ._scene import Coordinate, PanelLayout, Scene


class GridDrawing:
    """Draw one or more scenes side by side and update them for new times."""

    def __init__(
        self,
        figure: Figure,
        scenes: Sequence[Scene],
        *,
        theme: str,
        show_ion_labels: bool,
        show_processing_zone_labels: bool,
        show_hardware_ids: bool,
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
        panel_width = width / len(scenes)
        self.panels = [
            _PanelArtists(
                self.axis,
                scene,
                panel_layout(scene.geometry, index * panel_width, 0, panel_width, height, marker_size),
                self.colors,
                _Style(
                    show_ion_labels=show_ion_labels,
                    show_processing_zone_labels=show_processing_zone_labels,
                    show_hardware_ids=show_hardware_ids,
                    ion_color_mode=ion_color_mode,
                    ion_color_changes=ion_color_changes,
                    zone_colors=zone_colors,
                ),
            )
            for index, scene in enumerate(scenes)
        ]
        self.draw_at(min(scene.start_time for scene in scenes))

    def draw_at(self, time: float) -> list[Artist]:
        """Move ions and update highlights for a schedule time.

        Returns:
            The artists that changed.
        """
        return [artist for panel in self.panels for artist in panel.update(time)]


@dataclass(frozen=True, kw_only=True)
class _Style:
    """Hold the display choices that every panel shares."""

    show_ion_labels: bool
    show_processing_zone_labels: bool
    show_hardware_ids: bool
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
        shape = scene.geometry
        radius = layout.ion_radius
        font_size = layout.font_size
        self.zone_color = {
            zone.zone_id: style.zone_colors.get(zone.zone_id, distinct_color(index, ZONE_PALETTE))
            for index, zone in enumerate(scene.processing_zones)
        }
        segment_points = [(layout.point(start), layout.point(end)) for start, end in shape.segments]
        widths = [self._segment_width(segment.processing_zones) for segment in scene.segments]
        # Processing zones glow faintly in their own color, and fully while they run a gate.
        zone_segments = [index for index, segment in enumerate(scene.segments) if segment.processing_zones]
        axis.add_collection(
            LineCollection(
                [segment_points[index] for index in zone_segments],
                linewidths=[points(2.2 * widths[index]) for index in zone_segments],
                colors=[self._segment_color(scene.segments[index].processing_zones) for index in zone_segments],
                alpha=LINE_GLOW * AMBIENT_GLOW,
                capstyle="round",
                zorder=1,
            )
        )
        self.glow: LineCollection = LineCollection(
            segment_points,
            linewidths=[points(2.2 * width) for width in widths],
            colors="none",
            alpha=LINE_GLOW,
            capstyle="round",
            zorder=1,
        )
        axis.add_collection(self.glow)
        axis.add_collection(
            LineCollection(
                segment_points,
                linewidths=[points(width) for width in widths],
                colors=[self._segment_color(segment.processing_zones) for segment in scene.segments],
                capstyle="round",
                zorder=1,
            )
        )
        zone_points = [layout.point(shape.position(zone.place)) for zone in scene.processing_zones]
        if zone_points:
            axis.scatter(
                [x for x, _y in zone_points],
                [y for _x, y in zone_points],
                s=points(1.5 * radius) ** 2,
                marker="D",
                color=[self.zone_color[zone.zone_id] for zone in scene.processing_zones],
                linewidths=0,
                zorder=2,
            )
        if style.show_processing_zone_labels:
            label_size = zone_label_size(font_size)
            for zone, point in zip(scene.processing_zones, zone_points, strict=True):
                start, end = segment_points[zone.place.segment]
                normal = label_normal(*shape.segments[zone.place.segment])
                _parallel_text(
                    axis,
                    zone.zone_id,
                    point,
                    start,
                    end,
                    (normal[0], -normal[1]),
                    _zone_label_gap(radius, label_size),
                    color=self.zone_color[zone.zone_id],
                    size=label_size,
                    weight="semibold",
                )
        junction_points = [layout.point(position) for position in shape.junctions]
        if junction_points:
            axis.scatter(
                [x for x, _y in junction_points],
                [y for _x, y in junction_points],
                s=points(0.95 * radius) ** 2,
                marker="s",
                color=colors["junction"],
                linewidths=0,
                zorder=3,
            )
        if style.show_hardware_ids:
            self._add_hardware_ids(axis, segment_points, hardware_id_font_size(radius, font_size))
        initial = scene.ion_positions(scene.start_time)
        self.ion_order = [track.ion for track in scene.ions]
        start_points = [layout.point(initial[ion]) for ion in self.ion_order]
        # A running gate makes its ions glow in the color of its processing zone.
        # Every ion outside a running gate glows faintly in its ring color.
        self.ambient_halo = add_ambient_halo(axis, radius, 4)
        self.halos = add_halos(axis, radius, 4)
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
        if style.ion_color_mode == "custom":
            self._add_legend(axis)
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
            # Without a title, the action line moves up beside the time chip.
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
        shown_time = min(max(time, scene.start_time), scene.end_time)
        # Gates that take no time, such as virtual rz gates, get no glow.
        gate_glows = {
            ion: self.zone_color.get(gate.processing_zone or "", self.colors["gate"])
            for gate in scene.running_gates(shown_time)
            if gate.end > gate.start
            for ion in gate.ions
        }
        active_zones = scene.active_processing_zones(shown_time)
        positions = scene.ion_positions(shown_time)
        points = [self.layout.point(positions[ion]) for ion in self.ion_order]
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
        self.ions.set_offsets(points)
        move_halos(self.halos, points, [gate_glows.get(ion, "none") for ion in self.ion_order])
        self.ions.set_edgecolor([shade(border, RING_SHADE) for border, _fill, _text in styles])
        self.ions.set_facecolor([fill for _border, fill, _text in styles])
        self.inner_rings.set_offsets(points)
        self.inner_rings.set_edgecolor([border for border, _fill, _text in styles])
        self.ambient_halo.set_offsets(points)
        self.ambient_halo.set_facecolor([
            "none" if ion in gate_glows else border
            for ion, (border, _fill, _text) in zip(self.ion_order, styles, strict=True)
        ])
        for text, point, (_border, _fill, text_color) in zip(self.ion_labels, points, styles, strict=False):
            text.set_position(point)
            text.set_color(text_color)
        self.glow.set_color([
            next((self.zone_color[zone] for zone in segment.processing_zones if zone in active_zones), "none")
            for segment in scene.segments
        ])
        title, time_text, description = header_lines(scene, time)
        text_width = self.layout.width - 2 * self.layout.padding
        self.title.set_text(truncate(title, text_width * 0.6, self.layout.font_size))
        self.time.set_text(time_text)
        description_width = text_width if title else text_width * 0.75
        self.description.set_text(truncate(description, description_width, 0.85 * self.layout.font_size))
        return [
            self.ions,
            self.inner_rings,
            self.ambient_halo,
            *self.halos,
            self.glow,
            self.title,
            self.time,
            self.description,
            *self.ion_labels,
        ]

    def _segment_width(self, zones: tuple[str, ...]) -> float:
        """Return a segment's line width in pixels, as ``segmentWidth`` in ``draw.js``.

        Returns:
            The width.
        """
        radius = self.layout.ion_radius
        return max(2.0, 0.7 * radius) if zones else max(1.5, 0.45 * radius)

    def _segment_color(self, zones: tuple[str, ...]) -> str:
        return self.zone_color[zones[0]] if zones else self.colors["segment"]

    def _add_hardware_ids(
        self,
        axis: Axes,
        segment_points: Sequence[tuple[Coordinate, Coordinate]],
        font_size: float,
    ) -> None:
        # Segment IDs run along their segment, on the side away from the
        # processing-zone labels. Junction IDs go above and left of their junction.
        layout = self.layout
        radius = layout.ion_radius
        shape = self.scene.geometry
        for segment, (start, end), ends in zip(self.scene.segments, segment_points, shape.segments, strict=True):
            normal = label_normal(*ends)
            middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            _parallel_text(
                axis,
                segment.segment_id,
                middle,
                start,
                end,
                (-normal[0], normal[1]),
                0.6 * radius + 0.6 * font_size,
                color=self.colors["muted_text"],
                size=font_size,
                weight="normal",
            )
        for (junction_id, _position), point in zip(self.scene.junctions, shape.junctions, strict=True):
            x, y = layout.point(point)
            axis.text(
                x - 0.7 * radius,
                y - 0.7 * radius,
                junction_id,
                color=self.colors["muted_text"],
                fontsize=points(font_size),
                ha="right",
                va="bottom",
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


def _parallel_text(
    axis: Axes,
    text: str,
    anchor: Coordinate,
    start: Coordinate,
    end: Coordinate,
    normal: Coordinate,
    gap: float,
    *,
    color: str,
    size: float,
    weight: str,
) -> None:
    """Write text along a segment, moved away from it along ``normal`` (pixels, y down).

    The text stays readable from left to right, as ``parallelText`` in ``draw.js``.
    """
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    axis.text(
        anchor[0] + normal[0] * gap,
        anchor[1] + normal[1] * gap,
        text,
        color=color,
        fontsize=points(size),
        fontweight=weight,
        rotation=-angle,
        rotation_mode="anchor",
        ha="center",
        va="center",
        zorder=6,
    )


def _zone_label_gap(ion_radius: float, label_size: float) -> float:
    """Return the distance from a processing zone to its label, clear of the gate glow.

    Returns:
        The distance in pixels, as ``zoneLabelGap`` in ``draw.js``.
    """
    return 1.75 * ion_radius + 0.5 * label_size + 2
