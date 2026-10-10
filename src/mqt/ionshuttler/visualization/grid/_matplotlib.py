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

from ._colors import ZONE_PALETTE, distinct_color, ion_colors, legend
from ._scene import COLORS, header_lines, ion_label_font_size, label_normal, panel_layout, truncate

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    from ._colors import ColorChange
    from ._scene import Coordinate, PanelLayout, Scene

_DPI = 100
_POINTS_PER_PIXEL = 72 / _DPI
_VIDEO_CODECS = {".mp4": "h264", ".m4v": "h264", ".mov": "h264", ".mkv": "h264", ".webm": "libvpx-vp9"}


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
    ) -> None:
        """Create all artists for the scenes at their start times."""
        self.figure = figure
        self.colors = COLORS["dark" if theme == "dark" else "light"]
        width = figure.get_figwidth() * _DPI
        height = figure.get_figheight() * _DPI
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
                panel_layout(scene.geometry, index * panel_width, 0, panel_width, height),
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
        self.segments: LineCollection = LineCollection(
            segment_points,
            linewidths=[
                _points(self._segment_width(segment.processing_zones, active=False)) for segment in scene.segments
            ],
            colors=[self._segment_color(segment.processing_zones) for segment in scene.segments],
            capstyle="round",
            zorder=1,
        )
        axis.add_collection(self.segments)
        zone_points = [layout.point(shape.position(zone.place)) for zone in scene.processing_zones]
        if zone_points:
            axis.scatter(
                [x for x, _y in zone_points],
                [y for _x, y in zone_points],
                s=_points(1.1 * radius) ** 2,
                marker="D",
                color=[self.zone_color[zone.zone_id] for zone in scene.processing_zones],
                linewidths=0,
                zorder=2,
            )
        if style.show_processing_zone_labels:
            label_size = max(9.0, 0.78 * font_size)
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
                    weight="normal",
                )
        junction_points = [layout.point(position) for position in shape.junctions]
        if junction_points:
            axis.scatter(
                [x for x, _y in junction_points],
                [y for _x, y in junction_points],
                s=_points(2 * max(1.5, 0.3 * radius)) ** 2,
                color=colors["junction"],
                linewidths=0,
                zorder=3,
            )
        if style.show_hardware_ids:
            self._add_hardware_ids(axis, segment_points, max(8.0, 0.72 * font_size))
        initial = scene.ion_positions(scene.start_time)
        self.ion_order = [track.ion for track in scene.ions]
        start_points = [layout.point(initial[ion]) for ion in self.ion_order]
        # A running gate draws a thin outer ring in the color of its processing zone.
        gate_ring = radius + max(2.5, 0.3 * radius)
        self.gate_rings: PathCollection = axis.scatter(
            [x for x, _y in start_points],
            [y for _x, y in start_points],
            s=_points(2 * gate_ring) ** 2,
            facecolor="none",
            edgecolor="none",
            linewidths=_points(max(1.25, 0.15 * radius)),
            zorder=4,
        )
        ring_width = max(1.25, 0.18 * radius)
        self.ions: PathCollection = axis.scatter(
            [x for x, _y in start_points],
            [y for _x, y in start_points],
            s=_points(2 * radius - ring_width) ** 2,
            linewidths=_points(ring_width),
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
                        fontsize=_points(label_size),
                        fontweight="normal",
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
            fontsize=_points(font_size),
            fontweight="semibold",
            va="top",
            zorder=7,
        )
        self.time: Text = axis.text(
            layout.left + layout.width - layout.padding,
            top,
            "",
            color=colors["muted_text"],
            fontsize=_points(0.9 * font_size),
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
            fontsize=_points(0.85 * font_size),
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
        # Gates that take no time, such as virtual rz gates, get no ring.
        gate_rings = {
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
        self.ions.set_edgecolor([border for border, _fill, _text in styles])
        self.ions.set_facecolor([fill for _border, fill, _text in styles])
        self.gate_rings.set_offsets(points)
        self.gate_rings.set_edgecolor([gate_rings.get(ion, "none") for ion in self.ion_order])
        for text, point, (_border, _fill, text_color) in zip(self.ion_labels, points, styles, strict=False):
            text.set_position(point)
            text.set_color(text_color)
        self.segments.set_linewidth([
            _points(self._segment_width(zones, active=any(zone in active_zones for zone in zones)))
            for zones in (segment.processing_zones for segment in scene.segments)
        ])
        title, time_text, description = header_lines(scene, time)
        text_width = self.layout.width - 2 * self.layout.padding
        self.title.set_text(truncate(title, text_width * 0.6, self.layout.font_size))
        self.time.set_text(time_text)
        description_width = text_width if title else text_width * 0.75
        self.description.set_text(truncate(description, description_width, 0.85 * self.layout.font_size))
        return [self.ions, self.gate_rings, self.segments, self.title, self.time, self.description, *self.ion_labels]

    def _segment_width(self, zones: tuple[str, ...], *, active: bool) -> float:
        """Return a segment's line width in pixels, as ``segmentWidth`` in ``draw.js``.

        Returns:
            The width.
        """
        radius = self.layout.ion_radius
        if not zones:
            return max(1.25, 0.3 * radius)
        return max(2.5, 0.6 * radius) if active else max(2.0, 0.4 * radius)

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
                fontsize=_points(font_size),
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
                [x], [y], s=_points(size) ** 2, facecolor=fill, edgecolor=border, linewidths=_points(1.5), zorder=7
            )
            axis.text(
                x + size,
                y,
                label,
                color=self.colors["text"],
                fontsize=_points(size),
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
        fontsize=_points(size),
        fontweight=weight,
        rotation=-angle,
        rotation_mode="anchor",
        ha="center",
        va="center",
        zorder=6,
    )


def _zone_label_gap(ion_radius: float, label_size: float) -> float:
    """Return the distance from a processing zone to its label, clear of the gate ring.

    Returns:
        The distance in pixels, as ``zoneLabelGap`` in ``draw.js``.
    """
    return ion_radius + max(2.5, 0.3 * ion_radius) + 0.6 * label_size + 2


def _points(pixels: float) -> float:
    """Convert a pixel length to Matplotlib points.

    Returns:
        The length in points.
    """
    return pixels * _POINTS_PER_PIXEL


def save_video(
    figure: Figure,
    path: Path,
    times: Sequence[float],
    frames_per_second: int,
    draw_at: Callable[[float], object],
    background: str,
) -> None:
    """Write Matplotlib frames to a GIF or an FFmpeg video.

    Raises:
        ValueError: If the file suffix names no supported video format.
        RuntimeError: If the video format needs FFmpeg and FFmpeg is missing.
    """
    from matplotlib.animation import (  # ruff: ignore[import-outside-top-level]
        AbstractMovieWriter,
        FFMpegWriter,
        PillowWriter,
    )

    suffix = path.suffix.lower()
    writer: AbstractMovieWriter
    if suffix == ".gif":
        writer = PillowWriter(fps=frames_per_second)
    else:
        codec = _VIDEO_CODECS.get(suffix)
        if codec is None:
            formats = ", ".join([".gif", *_VIDEO_CODECS])
            msg = f"unsupported video file suffix {path.suffix!r}; use one of: {formats}"
            raise ValueError(msg)
        if not FFMpegWriter.isAvailable():
            msg = (
                f"writing {suffix} videos requires FFmpeg; install FFmpeg, write a .gif file, "
                "or export the video from the HTML view"
            )
            raise RuntimeError(msg)
        writer = FFMpegWriter(fps=frames_per_second, codec=codec)
    with writer.saving(figure, str(path), dpi=_DPI):
        for time in times:
            draw_at(time)
            writer.grab_frame(facecolor=background)
