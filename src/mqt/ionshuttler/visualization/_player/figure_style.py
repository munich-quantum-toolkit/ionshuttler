# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""The glows of the Matplotlib renderers.

Every ion and processing zone glows faintly at all times; the ions and zones
of a running gate glow fully. The browser drawings draw the glows with
``paintGlow`` and ``paintLineGlow`` in ``style.js``. Matplotlib has no radial
fill, so stacked halos stand in for its fade. Lengths are in pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .video import points

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection

# Strength of the glow that every ion and processing zone has at all times.
# Same value as `AMBIENT_GLOW` in `style.js`.
AMBIENT_GLOW = 0.55
# Opacity of the glow along a processing zone that runs a gate.
LINE_GLOW = 0.45
_HALOS = ((3.8, 0.18), (2.7, 0.27))
_AMBIENT_HALO = (3.2, 0.5 * AMBIENT_GLOW)


def add_ambient_halo(axis: Axes, radius: float, zorder: float) -> PathCollection:
    """Create the faint halo around every ion.

    Place and color it with ``set_offsets`` and ``set_facecolor``.

    Returns:
        The halo layer.
    """
    scale, alpha = _AMBIENT_HALO
    return axis.scatter([], [], s=points(scale * radius) ** 2, alpha=alpha, linewidths=0, zorder=zorder)


def add_halos(axis: Axes, radius: float, zorder: float) -> list[PathCollection]:
    """Create the halos around the ions of running gates.

    :func:`move_halos` places and colors them.

    Returns:
        The halo layers, from the outside in.
    """
    return [
        axis.scatter([], [], s=points(scale * radius) ** 2, alpha=alpha, linewidths=0, zorder=zorder)
        for scale, alpha in _HALOS
    ]


def move_halos(halos: Sequence[PathCollection], centers: Sequence[tuple[float, float]], colors: Sequence[str]) -> None:
    """Place the halos on the ions and color them; ``"none"`` hides an ion's halo."""
    for halo in halos:
        halo.set_offsets(list(centers))
        halo.set_facecolor(list(colors))
