# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Assemble the standalone player page, the viewer page, and the notebook frame.

A view is a JSON-compatible mapping with the keys ``title``, ``settings``,
``options``, ``drawing``, and ``drawing_name``. The standalone page embeds one
view. The viewer page loads views from the local viewer server.
"""

from __future__ import annotations

import base64
import html
import json
import re
from collections.abc import Sequence
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.resources.abc import Traversable

# A player option is ``(setting, label)`` for a checkbox, or
# ``(setting, label, ((value, label), ...))`` for a choice between values.
PlayerOption = tuple[str, str] | tuple[str, str, Sequence[tuple[str, str]]]

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_CONTROL_HEIGHT = 120
_EXPORT_PANEL_HEIGHT = 170


def view_data(
    *,
    title: str,
    settings: Mapping[str, object],
    options: Sequence[PlayerOption],
    drawing: object,
    drawing_name: str,
) -> dict[str, object]:
    """Collect everything the player needs to show one view.

    Args:
        title: View title.
        settings: Initial player settings, including ``width`` and ``height``.
        options: Settings shown as controls: ``(setting, label)`` for a
            checkbox, ``(setting, label, choices)`` for a switch between
            ``(value, label)`` choices.
        drawing: JSON-compatible data for the drawing script.
        drawing_name: Global name of the drawing object that draws this view.

    Returns:
        The view.
    """
    return {
        "title": title,
        "settings": dict(settings),
        "options": [
            [option[0], option[1], [list(choice) for choice in option[2]]] if len(option) == 3 else list(option)
            for option in options
        ],
        "drawing": drawing,
        "drawing_name": drawing_name,
    }


def player_page(view: Mapping[str, object], drawing_script: Traversable) -> str:
    """Return a standalone HTML page that plays one view.

    The page embeds the view as Base64-encoded UTF-8 JSON. Identifiers and
    descriptions therefore cannot form markup.

    Returns:
        The HTML document.
    """
    encoded = base64.b64encode(json.dumps(view, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    data = f'<script type="application/octet-stream" class="player-data">{encoded.decode("ascii")}</script>'
    body = _read("player.html").replace("</main>", f"  {data}\n</main>")
    scripts = (_read("webm.js"), _read("player.js"), drawing_script.read_text(encoding="utf-8"))
    return _document(str(view["title"]), body, (*scripts, f"Player.start({view['drawing_name']});\n"))


def viewer_page(token: str, drawing_scripts: Mapping[str, Traversable]) -> str:
    """Return the page of the local viewer.

    Args:
        token: Access token that the page sends with every request.
        drawing_scripts: Drawing scripts keyed by the global name they define.

    Returns:
        The HTML document.
    """
    body = _read("viewer.html").replace("{{player}}", _read("player.html"))
    names = ", ".join(drawing_scripts)
    scripts = (
        _read("webm.js"),
        _read("player.js"),
        *(script.read_text(encoding="utf-8") for script in drawing_scripts.values()),
        _read("viewer.js"),
        f"Viewer.start({json.dumps(token)}, {{ {names} }});\n",
    )
    return _document("IonShuttler viewer", body, scripts)


def notebook_frame(document: str, height: int) -> str:
    """Wrap a standalone page in an inline frame for notebook output.

    The frame keeps the page's styles and scripts apart from the notebook.

    Returns:
        The ``iframe`` element.
    """
    frame_height = height + _CONTROL_HEIGHT + _EXPORT_PANEL_HEIGHT
    return (
        f'<iframe srcdoc="{html.escape(document, quote=True)}" title="Schedule view" '
        f'style="width: 100%; height: {frame_height}px; border: 0;"></iframe>'
    )


def _document(title: str, body: str, scripts: Sequence[str]) -> str:
    """Fill the page skeleton.

    Returns:
        The HTML document.
    """
    document = _PLACEHOLDER.sub(lambda _match: html.escape(title), _read("page.html"))
    document = document.replace('<div data-fill="body"></div>', body)
    document = document.replace('<style data-fill="style"></style>', f"<style>\n{_read('player.css')}</style>")
    return document.replace('<script data-fill="script"></script>', "<script>\n" + "\n".join(scripts) + "</script>")


@cache
def _read(name: str) -> str:
    """Read a player file that ships with this package.

    Returns:
        The file content.
    """
    return files("mqt.ionshuttler.visualization._player").joinpath(name).read_text(encoding="utf-8")
