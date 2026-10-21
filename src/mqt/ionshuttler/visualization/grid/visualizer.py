# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Show Grid compilation results as interactive views, figures, and videos."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mqt.ionshuttler.grid.model import Junction

from .._player.page import notebook_frame, player_page, view_data
from .._player.settings import (
    check_display_settings,
    check_flag,
    require_positive_int,
    require_positive_number,
    video_frame_times,
    video_range,
)
from .._player.style import COLORS, color_changes, color_mode, color_mode_option, zone_colors
from ._scene import build_scene

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from mqt.ionshuttler.core.result import CompilationResult

    from .._player.page import PlayerOption
    from .._player.style import IonColors
    from ._matplotlib import GridDrawing
    from ._scene import Scene

JunctionCoordinates = (
    Mapping[str, Sequence[float]] | Mapping[Junction, Sequence[float]] | Mapping[str | Junction, Sequence[float]]
)
Theme = Literal["auto", "light", "dark"]
_FLAGS = (
    ("show_ion_labels", "Ion labels"),
    ("show_processing_zone_labels", "Processing-zone labels"),
    ("show_hardware_ids", "Segment and junction IDs"),
)
# The view offers this side bar only for results with a compiler planning trace.
_PLANNING_SIDEBAR = ("show_compiler_planning", "Compiler planning")


@dataclass(frozen=True)
class GridVisualizer:
    """Show Grid compilation results with one set of display settings.

    :meth:`visualize` returns an interactive :class:`GridView` for notebooks
    and browsers. :meth:`compare` shows several results side by side.
    :meth:`plot` draws one schedule time as a Matplotlib figure. The view,
    its video export, and the figure use the same layout, colors, and labels.

    Explicit coordinates may use junction IDs or :class:`Junction` values as
    keys. If no coordinates are supplied, generated grid IDs recover their
    row-column layout. Other planar architectures use a crossing-free layout;
    non-planar architectures use a deterministic spring layout.

    Attributes:
        junction_coordinates: Optional drawing position of every junction.
        theme: Color theme: ``"dark"``, ``"light"``, or ``"auto"``. Matplotlib
            figures and videos use the same theme; ``"auto"`` follows the
            browser setting and draws them light. Use ``"light"`` for print.
        show_ion_labels: Whether to write each ion ID inside its circle.
        show_processing_zone_labels: Whether to write processing-zone IDs.
        show_hardware_ids: Whether to write segment and junction IDs.
        timesteps_per_second: Playback speed in schedule timesteps per second.
            Videos use the same speed.
        width: View and video width in pixels.
        height: View and video height in pixels.
        frames_per_second: Video frame rate. It sets smoothness, not duration.
        video_start_time: First timestep of exported videos. ``None`` selects
            the schedule start.
        video_end_time: Last timestep of exported videos. ``None`` selects the
            schedule end.
        ion_colors: ``"distinct"`` gives every ion its own ring color,
            ``"single"`` gives all ions one color, and a mapping sets the
            colors of chosen ions. A mapping value is an
            :class:`~mqt.ionshuttler.visualization.IonColor` for
            the whole schedule, or ``(time, IonColor)`` pairs in increasing
            time order: each pair applies from its time on. Ions without an
            entry, and ions before their first time, have a gray ring. The
            view can switch between all three modes.
        processing_zone_colors: Colors of chosen processing zones, keyed by
            zone ID. Other zones take distinct colors.
        show_compiler_planning: Whether the interactive view starts with the
            compiler planning side bar open. The side bar and marks on the
            drawing show what the compiler intended to do at the latest
            decision before the current time. Only results compiled with
            ``record_planning_trace=True`` have them. Matplotlib figures and
            videos do not draw them.
        marker_size: Size of ions, junctions, and segment lines relative to
            the automatic size. At 1, the ions of the fullest segment fit side
            by side with a small gap. Use a smaller value for dense layouts. The size does
            not move any junction. The view's layout editor has a slider for
            it.
    """

    junction_coordinates: JunctionCoordinates | None = None
    theme: Theme = "dark"
    show_ion_labels: bool = True
    show_processing_zone_labels: bool = True
    show_hardware_ids: bool = False
    timesteps_per_second: float = 4.0
    width: int = 960
    height: int = 600
    frames_per_second: int = 30
    video_start_time: int | None = None
    video_end_time: int | None = None
    ion_colors: IonColors = "distinct"
    processing_zone_colors: Mapping[str, str] | None = None
    show_compiler_planning: bool = False
    marker_size: float = 1.0

    def __post_init__(self) -> None:
        """Validate the display settings.

        A setting of the wrong type raises :class:`TypeError`. A setting with
        an invalid value raises :class:`ValueError`.
        """
        check_display_settings(
            theme=self.theme,
            timesteps_per_second=self.timesteps_per_second,
            width=self.width,
            height=self.height,
            frames_per_second=self.frames_per_second,
            video_start_time=self.video_start_time,
            video_end_time=self.video_end_time,
        )
        for name, _label in (*_FLAGS, _PLANNING_SIDEBAR):
            check_flag(getattr(self, name), name)
        require_positive_number(self.marker_size, "marker_size")
        color_changes(self.ion_colors)
        zone_colors(self.processing_zone_colors)

    def visualize(self, result: CompilationResult, /) -> GridView:
        """Return an interactive view of a Grid result.

        A result without a Grid architecture and state raises
        :class:`TypeError`. A video range outside the schedule raises
        :class:`ValueError`.

        Args:
            result: Grid compilation result to show.

        Returns:
            The view. It displays itself in notebooks and can be saved as a
            standalone HTML file.
        """
        return GridView(self, (build_scene(result, self.junction_coordinates),))

    def compare(
        self,
        results: Mapping[str, CompilationResult],
        /,
        *,
        junction_coordinates: Mapping[str, JunctionCoordinates] | None = None,
    ) -> GridView:
        """Return a view that plays several Grid results side by side.

        All panels share one clock in absolute schedule time. A panel shows its
        initial state before its schedule starts and its final state after its
        schedule ends.

        A result that is not a Grid result raises :class:`TypeError`.

        Args:
            results: Grid results keyed by the title of their panel.
            junction_coordinates: Explicit coordinates for some results, keyed by
                panel title. Other results use the visualizer's
                ``junction_coordinates``, or the automatic layout.

        Returns:
            The view with one panel per result, in mapping order.

        Raises:
            TypeError: If a title is not a string.
            ValueError: If fewer than two results are given, a title is empty,
                or a coordinate title names no result.
        """
        if len(results) < 2:
            msg = "compare needs at least two results"
            raise ValueError(msg)
        for title in results:
            if not isinstance(title, str):
                msg = "result titles must be strings"
                raise TypeError(msg)
            if not title:
                msg = "result titles must be non-empty"
                raise ValueError(msg)
        coordinates = {} if junction_coordinates is None else dict(junction_coordinates)
        unknown = sorted(set(coordinates).difference(results))
        if unknown:
            msg = f"junction_coordinates names unknown results: {', '.join(unknown)}"
            raise ValueError(msg)
        scenes = tuple(
            build_scene(result, coordinates.get(title, self.junction_coordinates), title)
            for title, result in results.items()
        )
        return GridView(self, scenes)

    def plot(self, result: CompilationResult, time: float, /) -> Figure:
        """Draw a Grid result at one schedule time as a Matplotlib figure.

        Args:
            result: Grid compilation result to draw.
            time: Schedule time to draw. Fractional times show ions in motion.

        Returns:
            A Matplotlib figure of ``width`` by ``height`` pixels.

        Raises:
            TypeError: If the result is not a Grid result or the time is not a number.
            ValueError: If the time lies outside the schedule.
        """
        import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]

        scene = build_scene(result, self.junction_coordinates)
        if isinstance(time, bool) or not isinstance(time, int | float) or not math.isfinite(time):
            msg = "time must be a finite number"
            raise TypeError(msg)
        if not scene.start_time <= time <= scene.end_time:
            msg = f"time must lie within the schedule ({scene.start_time} to {scene.end_time})"
            raise ValueError(msg)
        figure = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        _drawing(self, figure, (scene,)).draw_at(time)
        return figure

    def open(self, result: CompilationResult, /, *, wait: bool | None = None) -> str:
        """Show a Grid result in the result viewer in the browser.

        This is ``visualize(result).open(wait=wait)``. In a notebook the call
        returns at once; in a plain script it blocks until Ctrl+C so the viewer
        keeps running.

        Args:
            result: Grid compilation result to show.
            wait: Whether to block until Ctrl+C. ``None`` waits in plain scripts only.

        Returns:
            The viewer address of the result.
        """
        return self.visualize(result).open(wait=wait)


class GridView:
    """Interactive view of one or more Grid results.

    The view is a self-contained HTML page with a canvas, playback controls,
    a time slider, display options, and a video export panel. It needs no
    network access and no web server. Create views with
    :meth:`GridVisualizer.visualize` or :meth:`GridVisualizer.compare`.
    """

    def __init__(self, visualizer: GridVisualizer, scenes: tuple[Scene, ...]) -> None:
        """Store the settings and drawing data, and check the video range against the schedules."""
        self.visualizer = visualizer
        self._scenes = scenes
        self._video_range()

    @property
    def start_time(self) -> int:
        """Earliest schedule start of all shown results."""
        return min(scene.start_time for scene in self._scenes)

    @property
    def end_time(self) -> int:
        """Latest schedule end of all shown results."""
        return max(scene.end_time for scene in self._scenes)

    def to_html(self) -> str:
        """Return the view as a standalone HTML document.

        Returns:
            The HTML document.
        """
        return player_page(self.player_data(), files("mqt.ionshuttler.visualization.grid").joinpath("draw.js"))

    def open(self, *, wait: bool | None = None) -> str:
        """Show the view in the result viewer in the browser.

        The viewer runs in this Python process; see
        :func:`~mqt.ionshuttler.visualization.viewer.open_viewer`.

        Args:
            wait: Whether to block until Ctrl+C so the viewer keeps running.
                ``None`` waits in plain scripts only.

        Returns:
            The viewer address of this view.
        """
        from ..viewer import show_in_viewer  # ruff: ignore[import-outside-top-level]

        return show_in_viewer(self.player_data(), wait=wait)

    def player_data(self) -> dict[str, object]:
        """Return the title, settings, and drawing data that the browser player shows.

        Returns:
            JSON-compatible data.
        """
        settings = self.visualizer
        start, end = self._video_range()
        return view_data(
            title=" vs. ".join(scene.title for scene in self._scenes) or "Grid schedule",
            settings={
                "theme": settings.theme,
                "show_ion_labels": settings.show_ion_labels,
                "show_processing_zone_labels": settings.show_processing_zone_labels,
                "show_hardware_ids": settings.show_hardware_ids,
                "show_compiler_planning": settings.show_compiler_planning,
                "marker_size": settings.marker_size,
                "timesteps_per_second": settings.timesteps_per_second,
                "width": settings.width,
                "height": settings.height,
                "frames_per_second": settings.frames_per_second,
                "video_start_time": start,
                "video_end_time": end,
                "ion_colors": color_mode(settings.ion_colors),
            },
            options=_options(settings),
            drawing={
                "colors": COLORS,
                "panels": [scene.to_dict() for scene in self._scenes],
                "ion_colors": {str(ion): changes for ion, changes in color_changes(settings.ion_colors).items()},
                "zone_colors": zone_colors(settings.processing_zone_colors),
            },
            drawing_name="GridDrawing",
            sidebar=_PLANNING_SIDEBAR if any(scene.planning is not None for scene in self._scenes) else None,
        )

    def save(self, path: str | Path) -> Path:
        """Write the view as a standalone HTML file.

        Returns:
            The written path.
        """
        target = Path(path)
        target.write_text(self.to_html(), encoding="utf-8")
        return target

    def export_video(
        self,
        path: str | Path,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        frames_per_second: int | None = None,
    ) -> Path:
        """Write a video of a time range with Matplotlib.

        The video uses the visualizer settings. It shows
        ``timesteps_per_second`` schedule timesteps per video second. Its first
        frame shows ``start_time`` and its last frame shows ``end_time``. The
        ``.gif`` format needs no further software. The ``.mp4``, ``.m4v``,
        ``.mov``, ``.mkv``, and ``.webm`` formats need FFmpeg.

        An unsupported format or an invalid range raises :class:`ValueError`.
        A missing FFmpeg program raises :class:`RuntimeError`.

        Args:
            path: Video file to write. Its suffix selects the format.
            start_time: First timestep. ``None`` uses the visualizer setting.
            end_time: Last timestep. ``None`` uses the visualizer setting.
            frames_per_second: Frame rate. ``None`` uses the visualizer setting.

        Returns:
            The written path.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # ruff: ignore[import-outside-top-level]
        from matplotlib.figure import Figure  # ruff: ignore[import-outside-top-level]

        from .._player.video import save_video  # ruff: ignore[import-outside-top-level]

        settings = self.visualizer
        start, end = self._video_range(start_time, end_time)
        rate = settings.frames_per_second if frames_per_second is None else frames_per_second
        require_positive_int(rate, "frames_per_second")
        target = Path(path)
        figure = Figure(figsize=(settings.width / 100, settings.height / 100), dpi=100)
        FigureCanvasAgg(figure)
        drawing = _drawing(settings, figure, self._scenes)
        save_video(
            figure,
            target,
            video_frame_times(start, end, settings.timesteps_per_second, rate),
            rate,
            drawing.draw_at,
            drawing.colors["background"],
        )
        return target

    def _repr_html_(self) -> str:  # ruff: ignore[bad-dunder-method-name] - Jupyter calls this display method.
        """Return the view for notebook display.

        Returns:
            An inline frame that contains the standalone document.
        """
        return notebook_frame(self.to_html(), self.visualizer.height)

    def _video_range(self, start_time: int | None = None, end_time: int | None = None) -> tuple[int, int]:
        """Return a validated video range.

        Missing bounds use the visualizer settings, then the schedule bounds.

        Returns:
            The first and last timestep of the video.
        """
        settings = self.visualizer
        if start_time is None:
            start_time = self.start_time if settings.video_start_time is None else settings.video_start_time
        if end_time is None:
            end_time = self.end_time if settings.video_end_time is None else settings.video_end_time
        return video_range(start_time, end_time, self.start_time, self.end_time)


def _drawing(settings: GridVisualizer, figure: Figure, scenes: tuple[Scene, ...]) -> GridDrawing:
    """Create a Matplotlib drawing with the given settings.

    Returns:
        The drawing.
    """
    from ._matplotlib import GridDrawing  # ruff: ignore[import-outside-top-level]

    return GridDrawing(
        figure,
        scenes,
        theme=settings.theme,
        show_ion_labels=settings.show_ion_labels,
        show_processing_zone_labels=settings.show_processing_zone_labels,
        show_hardware_ids=settings.show_hardware_ids,
        ion_color_mode=color_mode(settings.ion_colors),
        ion_color_changes=color_changes(settings.ion_colors),
        zone_colors=zone_colors(settings.processing_zone_colors),
        marker_size=settings.marker_size,
    )


def _options(settings: GridVisualizer) -> list[PlayerOption]:
    """Return the label checkboxes and the ion color choice of the browser player.

    Returns:
        The player options.
    """
    return [*_FLAGS, color_mode_option(settings.ion_colors)]


__all__ = ["GridView", "GridVisualizer", "JunctionCoordinates"]
