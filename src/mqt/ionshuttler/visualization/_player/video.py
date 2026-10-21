# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Write Matplotlib drawings as video files and convert pixel sizes for Matplotlib.

Every Matplotlib drawing uses 100 pixels per inch, so a figure of ``width`` by
``height`` pixels matches the browser view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from matplotlib.figure import Figure

DPI = 100
_POINTS_PER_PIXEL = 72 / DPI
_VIDEO_CODECS = {".mp4": "h264", ".m4v": "h264", ".mov": "h264", ".mkv": "h264", ".webm": "libvpx-vp9"}


def points(pixels: float) -> float:
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
    with writer.saving(figure, str(path), dpi=DPI):
        for time in times:
            draw_at(time)
            writer.grab_frame(facecolor=background)
