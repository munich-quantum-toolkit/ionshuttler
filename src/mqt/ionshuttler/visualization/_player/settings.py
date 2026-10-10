# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Checks and time rules for the display settings that every player view shares."""

from __future__ import annotations

import math

THEMES = ("auto", "light", "dark")


def check_display_settings(
    *,
    theme: object,
    timesteps_per_second: object,
    width: object,
    height: object,
    frames_per_second: object,
    video_start_time: object,
    video_end_time: object,
) -> None:
    """Validate the settings for theme, playback speed, size, and video export.

    A setting of the wrong type raises :class:`TypeError`.

    Raises:
        ValueError: If a setting has an invalid value.
    """
    if theme not in THEMES:
        msg = f"theme must be one of: {', '.join(THEMES)}"
        raise ValueError(msg)
    require_positive_number(timesteps_per_second, "timesteps_per_second")
    require_positive_int(frames_per_second, "frames_per_second")
    require_positive_int(width, "width")
    require_positive_int(height, "height")
    if video_start_time is not None:
        require_time(video_start_time, "video_start_time")
    if video_end_time is not None:
        require_time(video_end_time, "video_end_time")
    if isinstance(video_start_time, int) and isinstance(video_end_time, int) and video_start_time > video_end_time:
        msg = "video_start_time must not exceed video_end_time"
        raise ValueError(msg)


def check_flag(value: object, label: str) -> None:
    """Require a boolean setting.

    Raises:
        TypeError: If the value is not a boolean.
    """
    if not isinstance(value, bool):
        msg = f"{label} must be a boolean"
        raise TypeError(msg)


def video_range(
    start_time: int,
    end_time: int,
    schedule_start: int,
    schedule_end: int,
) -> tuple[int, int]:
    """Validate a video range against the schedule bounds.

    A bound that is not an integer raises :class:`TypeError`.

    Returns:
        The first and last timestep of the video.

    Raises:
        ValueError: If a bound lies outside the schedule or the range is reversed.
    """
    require_time(start_time, "start_time")
    require_time(end_time, "end_time")
    for name, value in (("start_time", start_time), ("end_time", end_time)):
        if not schedule_start <= value <= schedule_end:
            msg = f"{name} must lie within the schedule ({schedule_start} to {schedule_end})"
            raise ValueError(msg)
    if start_time > end_time:
        msg = "start_time must not exceed end_time"
        raise ValueError(msg)
    return start_time, end_time


def video_frame_times(
    start_time: int,
    end_time: int,
    timesteps_per_second: float,
    frames_per_second: int,
) -> list[float]:
    """Return the schedule time of each video frame.

    The video shows ``timesteps_per_second`` schedule timesteps per second.
    The first frame shows ``start_time`` and the last frame shows ``end_time``.
    The browser player uses the same rule.

    Returns:
        The frame times in increasing order.
    """
    duration = (end_time - start_time) / timesteps_per_second
    count = math.floor(duration * frames_per_second + 0.5) + 1
    if count == 1:
        return [float(start_time)]
    return [start_time + (end_time - start_time) * index / (count - 1) for index in range(count)]


def require_positive_number(value: object, label: str) -> None:
    """Require a finite positive number.

    Raises:
        TypeError: If the value is not a number.
        ValueError: If the value is not finite and positive.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{label} must be a number"
        raise TypeError(msg)
    if not math.isfinite(value) or value <= 0:
        msg = f"{label} must be positive and finite"
        raise ValueError(msg)


def require_positive_int(value: object, label: str) -> None:
    """Require a positive integer.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer"
        raise TypeError(msg)
    if value < 1:
        msg = f"{label} must be positive"
        raise ValueError(msg)


def require_time(value: object, label: str) -> None:
    """Require a non-negative integer timestep.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is negative.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer timestep"
        raise TypeError(msg)
    if value < 0:
        msg = f"{label} must be non-negative"
        raise ValueError(msg)
