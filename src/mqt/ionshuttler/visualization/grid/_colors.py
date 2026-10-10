# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Ion and processing-zone colors that both Grid renderers share.

``draw.js`` implements the same rules, so a figure, a video, and the browser
view color every ion and processing zone alike.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# One change of an ion's colors: (time, border, fill, label).
ColorChange = tuple[int, str | None, str | None, str | None]

ION_PALETTE = (
    "#3b6ea8",
    "#d1752e",
    "#c4484f",
    "#4f9a94",
    "#5a9a4b",
    "#b8962e",
    "#8e6aa0",
    "#c76d8e",
    "#8a6a52",
    "#6f7782",
    "#2f8fb8",
    "#7d8a35",
)
ZONE_PALETTE = ("#c08a2e", "#3f8f88", "#b05d78", "#6f68a8", "#6f8f3a", "#3f7fa8", "#b3653a", "#8a5a5a")
NEUTRAL_BORDER = "#8a919c"
WHITE = "#ffffff"
DARK_TEXT = "#1c2430"


def distinct_color(index: int, palette: Sequence[str]) -> str:
    """Return a palette color, then evenly spread hues once the palette runs out.

    Returns:
        The color as ``#rrggbb``.
    """
    if index < len(palette):
        return palette[index]
    hue = (index * 137.508) % 360
    return _hsl(hue, 0.42, 0.48)


def ion_colors(
    ion: int,
    time: float,
    mode: str,
    single: str,
    changes: Mapping[int, Sequence[ColorChange]],
    plain_fill: str = WHITE,
) -> tuple[str, str, str]:
    """Return the border, fill, and label text color of an ion at a schedule time.

    ``mode`` is ``"distinct"``, ``"single"``, or ``"custom"``. In custom mode an
    ion keeps the colors of its latest change; before its first change, and
    without changes, it has a neutral border. Ions without a chosen fill use
    ``plain_fill``, which depends on the theme.

    Returns:
        The border, fill, and text color.
    """
    border, fill = NEUTRAL_BORDER, plain_fill
    if mode == "distinct":
        border = distinct_color(ion, ION_PALETTE)
    elif mode == "single":
        border = single
    else:
        timeline = changes.get(ion, ())
        index = bisect_right([change[0] for change in timeline], time) - 1
        if index >= 0:
            _time, chosen_border, chosen_fill, _label = timeline[index]
            border = chosen_border or border
            fill = chosen_fill or fill
    return border, fill, text_color(fill)


def legend(changes: Mapping[int, Sequence[ColorChange]], plain_fill: str = WHITE) -> list[tuple[str, str, str]]:
    """Return one legend entry per label, in order of first appearance.

    Returns:
        The label, border, and fill of each entry.
    """
    entries: dict[str, tuple[str, str]] = {}
    for ion in sorted(changes):
        for _time, border, fill, label in changes[ion]:
            if label is not None and label not in entries:
                entries[label] = (border or NEUTRAL_BORDER, fill or plain_fill)
    return [(label, border, fill) for label, (border, fill) in entries.items()]


def text_color(fill: str) -> str:
    """Return dark text on light fills and white text on dark fills.

    Returns:
        The text color.
    """
    red, green, blue = (_linear(int(fill[index : index + 2], 16) / 255) for index in (1, 3, 5))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return DARK_TEXT if luminance > 0.35 else WHITE


def _linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _hsl(hue: float, saturation: float, lightness: float) -> str:
    """Convert a color from hue, saturation, and lightness to ``#rrggbb``.

    Returns:
        The color.
    """
    chroma = (1 - abs(2 * lightness - 1)) * saturation
    second = chroma * (1 - abs((hue / 60) % 2 - 1))
    match int(hue // 60):
        case 0:
            red, green, blue = chroma, second, 0.0
        case 1:
            red, green, blue = second, chroma, 0.0
        case 2:
            red, green, blue = 0.0, chroma, second
        case 3:
            red, green, blue = 0.0, second, chroma
        case 4:
            red, green, blue = second, 0.0, chroma
        case _:
            red, green, blue = chroma, 0.0, second
    shift = lightness - chroma / 2
    return "#" + "".join(f"{math.floor((value + shift) * 255 + 0.5):02x}" for value in (red, green, blue))
