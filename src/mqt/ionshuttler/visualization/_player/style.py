# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Theme colors, ion colors, and text sizes that every drawing shares.

``style.js`` implements the same rules for the browser, so a figure, a video,
and the browser view of any result color every ion and processing zone alike.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .settings import require_time

# One change of an ion's colors: (time, border, fill, label).
ColorChange = tuple[int, str | None, str | None, str | None]

# Theme colors of the drawings. "ion" is the ring color in single-color mode,
# "ion_fill" fills ions without a chosen fill, and "gate" marks gates that run
# outside a processing zone. "segment" draws transport lines and "junction"
# draws junctions and trap sites. "planning" marks the compiler planning
# overlay of the Grid view.
COLORS: dict[str, dict[str, str]] = {
    "light": {
        "background": "#ffffff",
        "segment": "#c5cbd3",
        "junction": "#64748b",
        "ion": "#2563eb",
        "ion_fill": "#ffffff",
        "gate": "#9333ea",
        "text": "#0f172a",
        "muted_text": "#64748b",
        "chip": "#f1f5f9",
        "planning": "#b45309",
    },
    "dark": {
        "background": "#0f1218",
        "segment": "#323a46",
        "junction": "#7c8798",
        "ion": "#3b82f6",
        "ion_fill": "#ffffff",
        "gate": "#a855f7",
        "text": "#e2e8f0",
        "muted_text": "#8b95a5",
        "chip": "#1b212b",
        "planning": "#facc15",
    },
}
ION_PALETTE = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#ca8a04",
    "#4f46e5",
    "#0d9488",
    "#78716c",
)
ZONE_PALETTE = ("#f59e0b", "#14b8a6", "#ec4899", "#8b5cf6", "#84cc16", "#06b6d4", "#f97316", "#ef4444")
NEUTRAL_BORDER = "#64748b"
WHITE = "#ffffff"
DARK_TEXT = "#0f172a"
# How much darker an ion's ring gets toward its outer edge. Same value as
# `RING_SHADE` in `style.js`.
RING_SHADE = 0.2


@dataclass(frozen=True)
class IonColor:
    """Colors and an optional legend label for one ion.

    Colors accept any Matplotlib color, for example ``"#e11d48"``,
    ``"crimson"``, or ``"tab:blue"``.

    Attributes:
        border: Color of the ion's ring. ``None`` keeps a neutral gray ring.
        fill: Color inside the ring. ``None`` keeps it white. The ion label
            switches to white text on dark fills.
        label: Legend entry for ions with these colors, for example
            ``"ancilla"``. Ions with the same label share one entry.
    """

    border: str | None = None
    fill: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        """Validate the colors and the label.

        Raises:
            TypeError: If the label is not a string.
        """
        for name in ("border", "fill"):
            value = getattr(self, name)
            if value is not None:
                hex_color(value, name)
        if self.label is not None and not isinstance(self.label, str):
            msg = "label must be a string"
            raise TypeError(msg)


IonColors = Literal["distinct", "single"] | Mapping[int, IonColor | Sequence[tuple[int, IonColor]]]
_COLOR_MODES = (("distinct", "Distinct"), ("single", "Single"), ("custom", "Custom"))


def color_mode(ion_colors: IonColors) -> str:
    """Return the ion color mode: ``"distinct"``, ``"single"``, or ``"custom"``.

    Returns:
        The mode.
    """
    return ion_colors if isinstance(ion_colors, str) else "custom"


def color_mode_option(ion_colors: IonColors) -> tuple[str, str, list[tuple[str, str]]]:
    """Return the player's ion color choice.

    The choice offers "Custom" only when the settings define ion colors.

    Returns:
        The player option.
    """
    custom = color_mode(ion_colors) == "custom"
    return ("ion_colors", "Ion colors", [mode for mode in _COLOR_MODES if mode[0] != "custom" or custom])


def color_changes(ion_colors: IonColors) -> dict[int, list[ColorChange]]:
    """Validate custom ion colors and list each ion's color changes in time order.

    Returns:
        The changes keyed by ion.

    Raises:
        TypeError: If an entry has the wrong type.
        ValueError: If a mode name is unknown or times are not increasing.
    """
    if isinstance(ion_colors, str):
        if ion_colors not in {"distinct", "single"}:
            msg = "ion_colors must be 'distinct', 'single', or a mapping from ions to colors"
            raise ValueError(msg)
        return {}
    if not isinstance(ion_colors, Mapping):
        msg = "ion_colors must be 'distinct', 'single', or a mapping from ions to colors"
        raise TypeError(msg)
    changes: dict[int, list[ColorChange]] = {}
    for ion, entry in ion_colors.items():
        if isinstance(ion, bool) or not isinstance(ion, int) or ion < 0:
            msg = "ion_colors keys must be non-negative ion IDs"
            raise TypeError(msg)
        timeline = [(0, entry)] if isinstance(entry, IonColor) else list(entry)
        times = []
        for item in timeline:
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], IonColor)):
                msg = f"colors of ion {ion} must be an IonColor or (time, IonColor) pairs"
                raise TypeError(msg)
            require_time(item[0], f"color time of ion {ion}")
            times.append(item[0])
        if times != sorted(set(times)):
            msg = f"color times of ion {ion} must increase"
            raise ValueError(msg)
        changes[ion] = [
            (
                time,
                None if color.border is None else hex_color(color.border, "border"),
                None if color.fill is None else hex_color(color.fill, "fill"),
                color.label,
            )
            for time, color in timeline
        ]
    return changes


def zone_colors(processing_zone_colors: Mapping[str, str] | None) -> dict[str, str]:
    """Validate chosen processing-zone colors.

    Returns:
        Colors as ``#rrggbb`` keyed by zone ID.

    Raises:
        TypeError: If the setting is not a mapping.
    """
    if processing_zone_colors is None:
        return {}
    if not isinstance(processing_zone_colors, Mapping):
        msg = "processing_zone_colors must be a mapping from zone IDs to colors"
        raise TypeError(msg)
    return {str(zone): hex_color(color, f"color of {zone!r}") for zone, color in processing_zone_colors.items()}


def hex_color(value: object, label: str) -> str:
    """Convert any Matplotlib color to ``#rrggbb``.

    Returns:
        The color.

    Raises:
        TypeError: If the value is not a string.
        ValueError: If Matplotlib does not know the color.
    """
    from matplotlib.colors import to_hex  # ruff: ignore[import-outside-top-level]

    if not isinstance(value, str):
        msg = f"{label} must be a color string"
        raise TypeError(msg)
    try:
        return to_hex(value, keep_alpha=False)
    except ValueError as error:
        msg = f"{label} {value!r} is not a known color"
        raise ValueError(msg) from error


def distinct_color(index: int, palette: Sequence[str]) -> str:
    """Return a palette color, then evenly spread hues once the palette runs out.

    Returns:
        The color as ``#rrggbb``.
    """
    if index < len(palette):
        return palette[index]
    hue = (index * 137.508) % 360
    return _hsl(hue, 0.65, 0.45)


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


def shade(color: str, amount: float) -> str:
    """Blend a ``#rrggbb`` color toward black.

    Returns:
        The color as ``#rrggbb``.
    """
    return "#" + "".join(
        f"{math.floor(int(color[index : index + 2], 16) * (1 - amount) + 0.5):02x}" for index in (1, 3, 5)
    )


def ion_label_font_size(ion_radius: float, label: str) -> float:
    """Return a font size that fits an ion label inside its colored ring.

    Returns:
        The font size in pixels.
    """
    return min(0.95 * ion_radius, 2.5 * ion_radius / max(len(label), 2))


def truncate(text: str, width: float, font_size: float) -> str:
    """Shorten text to an estimated pixel width.

    Returns:
        The text, shortened with an ellipsis if necessary.
    """
    limit = max(int(width / (0.55 * font_size)), 1)
    return text if len(text) <= limit else text[: limit - 1] + "…"


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
