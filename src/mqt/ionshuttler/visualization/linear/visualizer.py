# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Show Linear compilation results as interactive views, figures, and videos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
from ._trajectories import plot_trajectories

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.figure import Figure

    from mqt.ionshuttler.core.result import CompilationResult

    from .._player.page import PlayerOption
    from .._player.style import IonColors
    from ._matplotlib import LinearDrawing
    from ._scene import Scene

Theme = Literal["auto", "light", "dark"]
_FLAGS = (
    ("show_ion_labels", "Ion labels"),
    ("show_processing_zone_labels", "Processing-zone labels"),
    ("show_site_numbers", "Site numbers"),
)
# The view offers this switch only for results with a field profile.
_FIELD_FLAG = ("show_field_profile", "Field profile")
_DETAILS_SIDEBAR = ("show_schedule_details", "Schedule details")


@dataclass(frozen=True)
class LinearVisualizer:
    """Show Linear compilation results with one set of display settings.

    :meth:`visualize` returns an interactive :class:`LinearView` for notebooks
    and browsers. The view shows the ion chain on its trap sites, with the
    field profile behind it. Ions swap past each other, and running gates glow
    in the color of their processing zone. :meth:`compare` stacks several
    results. :meth:`plot` draws one schedule time as a Matplotlib figure, and
    :meth:`plot_trajectories` draws every ion's site over time. The view, its
    video export, and :meth:`plot` use the same layout, colors, and labels.

    Attributes:
        theme: Color theme: ``"dark"``, ``"light"``, or ``"auto"``. Matplotlib
            figures and videos use the same theme; ``"auto"`` follows the
            browser setting and draws them light. Use ``"light"`` for print.
        show_ion_labels: Whether to write each ion ID inside its circle.
        show_processing_zone_labels: Whether to write processing-zone IDs.
        show_site_numbers: Whether to number the trap sites.
        show_field_profile: Whether to draw the field profile behind the
            chain. Results whose sites all have the same field have no
            profile.
        show_schedule_details: Whether the interactive view starts with the
            schedule side bar open. The side bar lists the running and next
            gates with their ions and processing zones, the moving ions, each
            ion's site, and the outcome of the compilation with the run
            statistics that the compiler reports. Matplotlib figures and
            videos do not show it.
        timesteps_per_second: Playback speed in schedule timesteps per second.
            Videos use the same speed.
        width: View and video width in pixels.
        height: View and video height in pixels. Compared results share it.
        frames_per_second: Video frame rate. It sets smoothness, not duration.
        video_start_time: First timestep of exported videos. ``None`` selects
            the schedule start.
        video_end_time: Last timestep of exported videos. ``None`` selects the
            schedule end.
        ion_colors: ``"distinct"`` gives every ion its own ring color,
            ``"single"`` gives all ions one color, and a mapping sets the
            colors of chosen ions. A mapping value is an
            :class:`~mqt.ionshuttler.visualization.IonColor` for the whole
            schedule, or ``(time, IonColor)`` pairs in increasing time order:
            each pair applies from its time on. Ions without an entry, and ions
            before their first time, have a gray ring. The view can switch
            between all three modes.
        processing_zone_colors: Colors of chosen processing zones, keyed by
            zone ID. Other zones take distinct colors.
        marker_size: Size of ions, sites, and chain lines relative to the
            automatic size. The view's layout editor has a slider for it.
    """

    theme: Theme = "dark"
    show_ion_labels: bool = True
    show_processing_zone_labels: bool = True
    show_site_numbers: bool = True
    show_field_profile: bool = False
    show_schedule_details: bool = False
    timesteps_per_second: float = 4.0
    width: int = 960
    height: int = 420
    frames_per_second: int = 30
    video_start_time: int | None = None
    video_end_time: int | None = None
    ion_colors: IonColors = "distinct"
    processing_zone_colors: Mapping[str, str] | None = None
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
        for name, _label in (*_FLAGS, _FIELD_FLAG, _DETAILS_SIDEBAR):
            check_flag(getattr(self, name), name)
        require_positive_number(self.marker_size, "marker_size")
        color_changes(self.ion_colors)
        zone_colors(self.processing_zone_colors)

    def visualize(self, result: CompilationResult, /) -> LinearView:
        """Return an interactive view of a Linear result.

        A result without a Linear architecture and state raises
        :class:`TypeError`. A video range outside the schedule raises
        :class:`ValueError`.

        Args:
            result: Linear compilation result to show.

        Returns:
            The view. It displays itself in notebooks and can be saved as a
            standalone HTML file.
        """
        return LinearView(self, (_scene(result),))

    def compare(self, results: Mapping[str, CompilationResult], /) -> LinearView:
        """Return a view that plays several Linear results one above the other.

        All panels share one clock in absolute schedule time. A panel shows its
        initial state before its schedule starts and its final state after its
        schedule ends.

        A result that is not a Linear result raises :class:`TypeError`.

        Args:
            results: Linear results keyed by the title of their panel.

        Returns:
            The view with one panel per result, in mapping order.

        Raises:
            TypeError: If a title is not a string.
            ValueError: If fewer than two results are given or a title is empty.
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
        return LinearView(self, tuple(_scene(result, title) for title, result in results.items()))

    def plot(self, result: CompilationResult, time: float, /) -> Figure:
        """Draw a Linear result at one schedule time as a Matplotlib figure.

        Args:
            result: Linear compilation result to draw.
            time: Schedule time to draw. Fractional times show ions in motion.

        Returns:
            A Matplotlib figure of ``width`` by ``height`` pixels.

        Raises:
            TypeError: If the result is not a Linear result or the time is not a number.
            ValueError: If the time lies outside the schedule.
        """
        import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]

        scene = _scene(result)
        if isinstance(time, bool) or not isinstance(time, int | float) or not math.isfinite(time):
            msg = "time must be a finite number"
            raise TypeError(msg)
        if not scene.start_time <= time <= scene.end_time:
            msg = f"time must lie within the schedule ({scene.start_time} to {scene.end_time})"
            raise ValueError(msg)
        figure = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        _drawing(self, figure, (scene,)).draw_at(time)
        return figure

    @staticmethod
    def plot_trajectories(result: CompilationResult, /) -> Figure:
        """Draw every ion's site over the whole schedule as a Matplotlib figure.

        The figure marks each gate on its ions. Long schedules wrap into
        several rows. The figure uses Matplotlib's own style, not the display
        settings.

        Args:
            result: Linear compilation result to draw.

        Returns:
            The figure.
        """
        return plot_trajectories(result)

    def open(self, result: CompilationResult, /, *, wait: bool | None = None) -> str:
        """Show a Linear result in the result viewer in the browser.

        This is ``visualize(result).open(wait=wait)``. In a notebook the call
        returns at once; in a plain script it blocks until Ctrl+C so the viewer
        keeps running.

        Args:
            result: Linear compilation result to show.
            wait: Whether to block until Ctrl+C. ``None`` waits in plain scripts only.

        Returns:
            The viewer address of the result.
        """
        return self.visualize(result).open(wait=wait)


class LinearView:
    """Interactive view of one or more Linear results.

    The view is a self-contained HTML page with a canvas, playback controls,
    a time slider, display options, a schedule side bar, and a video export
    panel. It needs no network access and no web server. Create views with
    :meth:`LinearVisualizer.visualize` or :meth:`LinearVisualizer.compare`.
    """

    def __init__(self, visualizer: LinearVisualizer, scenes: tuple[Scene, ...]) -> None:
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
        return player_page(self.player_data(), files("mqt.ionshuttler.visualization.linear").joinpath("draw.js"))

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
            title=" vs. ".join(scene.title for scene in self._scenes) or "Linear schedule",
            settings={
                "theme": settings.theme,
                "show_ion_labels": settings.show_ion_labels,
                "show_processing_zone_labels": settings.show_processing_zone_labels,
                "show_site_numbers": settings.show_site_numbers,
                "show_field_profile": settings.show_field_profile,
                "show_schedule_details": settings.show_schedule_details,
                "marker_size": settings.marker_size,
                "timesteps_per_second": settings.timesteps_per_second,
                "width": settings.width,
                "height": settings.height,
                "frames_per_second": settings.frames_per_second,
                "video_start_time": start,
                "video_end_time": end,
                "ion_colors": color_mode(settings.ion_colors),
            },
            options=self._options(),
            drawing={
                "colors": COLORS,
                "panels": [scene.to_dict() for scene in self._scenes],
                "ion_colors": {str(ion): changes for ion, changes in color_changes(settings.ion_colors).items()},
                "zone_colors": zone_colors(settings.processing_zone_colors),
            },
            drawing_name="LinearDrawing",
            sidebar=_DETAILS_SIDEBAR,
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

    def _options(self) -> list[PlayerOption]:
        """Return the label switches and the ion color choice of the browser player.

        The field profile switch appears only if a result has a field profile.

        Returns:
            The player options.
        """
        has_field = any(scene.field is not None for scene in self._scenes)
        flags = [*_FLAGS, _FIELD_FLAG] if has_field else list(_FLAGS)
        return [*flags, color_mode_option(self.visualizer.ion_colors)]

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


def _scene(result: CompilationResult, title: str = "") -> Scene:
    """Replay a result for drawing.

    The Linear compiler modules load only here, so importing the package
    stays light.

    Returns:
        The scene.
    """
    from ._scene import build_scene  # ruff: ignore[import-outside-top-level]

    return build_scene(result, title)


def _drawing(settings: LinearVisualizer, figure: Figure, scenes: tuple[Scene, ...]) -> LinearDrawing:
    """Create a Matplotlib drawing with the given settings.

    Returns:
        The drawing.
    """
    from ._matplotlib import LinearDrawing  # ruff: ignore[import-outside-top-level]

    return LinearDrawing(
        figure,
        scenes,
        theme=settings.theme,
        show_ion_labels=settings.show_ion_labels,
        show_processing_zone_labels=settings.show_processing_zone_labels,
        show_site_numbers=settings.show_site_numbers,
        show_field_profile=settings.show_field_profile,
        ion_color_mode=color_mode(settings.ion_colors),
        ion_color_changes=color_changes(settings.ion_colors),
        zone_colors=zone_colors(settings.processing_zone_colors),
        marker_size=settings.marker_size,
    )


__all__ = ["LinearView", "LinearVisualizer"]
