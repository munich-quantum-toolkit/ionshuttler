# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the shared browser player: page, settings rules, and video files."""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess
from importlib.resources import files
from typing import TYPE_CHECKING, Any, cast

import pytest

from mqt.ionshuttler.visualization._player.page import notebook_frame, player_page, view_data
from mqt.ionshuttler.visualization._player.settings import video_frame_times, video_range

if TYPE_CHECKING:
    from pathlib import Path

_PLAYER = files("mqt.ionshuttler.visualization._player")
_SETTINGS = {
    "theme": "auto",
    "show_labels": True,
    "timesteps_per_second": 4.0,
    "width": 320,
    "height": 200,
    "frames_per_second": 30,
    "video_start_time": 0,
    "video_end_time": 4,
}
_DRAWING = """
const TestDrawing = {
  prepare: (data) => data,
  timeBounds: () => [0, 4],
  stepTimes: () => [0, 2],
  drawAtTime: () => {},
};
"""


def _view(title: str = "Schedule", drawing: object = None) -> dict[str, object]:
    return view_data(
        title=title,
        settings=_SETTINGS,
        options=[("show_labels", "Labels")],
        drawing=drawing,
        drawing_name="TestDrawing",
    )


def _page(tmp_path: Path, title: str = "Schedule", drawing_data: object = None) -> str:
    script = tmp_path / "drawing.js"
    script.write_text(_DRAWING, encoding="utf-8")
    return player_page(_view(title, drawing_data), script)


def _embedded_data(document: str) -> dict[str, Any]:
    match = re.search(r'class="player-data">\s*([^<]*?)\s*</script>', document)
    assert match is not None
    return cast("dict[str, Any]", json.loads(base64.b64decode(match.group(1)).decode("utf-8")))


def test_page_is_standalone_and_offline(tmp_path: Path) -> None:
    """Embed styles, scripts, and data without loading other files."""
    document = _page(tmp_path)

    assert document.startswith("<!doctype html>")
    assert "<canvas" in document
    assert "const TestDrawing" in document
    assert document.rstrip().endswith("</html>")
    assert "Player.start(TestDrawing);" in document
    assert not re.search(r"<script[^>]*\bsrc=|<link\b|https?://", document)


def test_page_embeds_settings_options_and_drawing_data(tmp_path: Path) -> None:
    """Start the browser controls from the Python settings."""
    data = _embedded_data(_page(tmp_path, drawing_data={"panels": [1, 2]}))

    assert data == {
        "title": "Schedule",
        "settings": _SETTINGS,
        "options": [["show_labels", "Labels"]],
        "drawing": {"panels": [1, 2]},
        "drawing_name": "TestDrawing",
    }


def test_view_names_a_side_bar_only_when_asked() -> None:
    """Embed the side bar setting and label only for views that have a side bar."""
    with_sidebar = view_data(
        title="Schedule",
        settings=_SETTINGS,
        options=[],
        drawing=None,
        drawing_name="TestDrawing",
        sidebar=("show_notes", "Notes"),
    )

    assert with_sidebar["sidebar"] == ["show_notes", "Notes"]
    assert "sidebar" not in _view()


def test_page_hides_the_side_bar_and_its_button_until_a_view_names_one(tmp_path: Path) -> None:
    """Start with a closed side bar and a hidden button; the player shows them for a named side bar."""
    document = _page(tmp_path)

    sidebar = re.search(r'<aside\s+class="player-sidebar"[^>]*>', document)
    toggle = re.search(r'<button[^>]*class="player-button player-sidebar-toggle"[^>]*>', document)
    assert sidebar is not None
    assert "hidden" in sidebar.group(0)
    assert toggle is not None
    assert "hidden" in toggle.group(0)
    assert 'aria-expanded="false"' in toggle.group(0)


def test_page_data_cannot_break_out_of_the_document(tmp_path: Path) -> None:
    """Encode user-controlled text so that it cannot form markup."""
    hostile = "</script><script>alert(1)</script><img src=x>"

    document = _page(tmp_path, title=hostile, drawing_data={"name": hostile})

    assert document.count("</script>") == 2
    assert "<img" not in document
    assert _embedded_data(document)["drawing"]["name"] == hostile


def test_page_labels_the_time_slider_and_both_video_range_handles(tmp_path: Path) -> None:
    """Give the time slider and each end of the video range their own label."""
    document = _page(tmp_path)

    def label_of(name: str) -> str | None:
        element = re.search(rf'<input[^>]*class="{name}"[^>]*>', document)
        label = None if element is None else re.search(r'aria-label="([^"]*)"', element.group(0))
        return None if label is None else label.group(1)

    assert label_of("player-time") == "Schedule time"
    assert label_of("player-range-start") == "Video start timestep"
    assert label_of("player-range-end") == "Video end timestep"


def test_layout_editor_offers_a_size_slider_only_for_drawings_with_a_marker_size(tmp_path: Path) -> None:
    """Start with a hidden, labeled size slider; the player shows it when the settings have a marker size."""
    document = _page(tmp_path)
    script = _PLAYER.joinpath("player.js").read_text(encoding="utf-8")

    size = re.search(r'<label\s+class="player-size"[^>]*>', document)
    slider = re.search(r'<input[^>]*class="player-size-input"[^>]*>', document)
    assert size is not None
    assert "hidden" in size.group(0)
    assert slider is not None
    assert 'aria-label="Size of ions and junctions"' in slider.group(0)
    assert 'typeof settings.marker_size === "number"' in script
    assert "drawing.settingsText(view, settings)" in script


def test_player_draws_the_canvas_and_video_frames_with_the_drawing_function() -> None:
    """Draw playback, the time slider, steps, and video frames with one drawing call."""
    script = _PLAYER.joinpath("player.js").read_text(encoding="utf-8")

    assert script.count("drawing.drawAtTime(") == 2
    frame_call = r"drawing\.drawAtTime\(\s*context,\s*view,\s*times\[index\],\s*settings,\s*width,\s*height,?\s*\)"
    assert re.search(frame_call, script) is not None


def test_notebook_frame_contains_the_complete_page(tmp_path: Path) -> None:
    """Show the complete page in an isolated frame inside notebooks."""
    document = _page(tmp_path)

    frame = notebook_frame(document, 200)

    match = re.fullmatch(r'<iframe srcdoc="([^"]*)" title="Schedule view" style="[^"]*"></iframe>', frame)
    assert match is not None
    assert html.unescape(match.group(1)) == document


def test_video_frames_cover_the_range_at_the_playback_speed() -> None:
    """Show ``timesteps_per_second`` timesteps per video second, including both ends."""
    times = video_frame_times(10, 20, 5.0, 4)

    assert len(times) == 9
    assert (times[0], times[-1]) == (10, 20)
    assert times == sorted(times)
    assert video_frame_times(3, 3, 4.0, 30) == [3.0]


def test_video_range_must_lie_within_the_schedule() -> None:
    """Reject ranges that leave the schedule or run backwards."""
    assert video_range(1, 3, 0, 4) == (1, 3)
    with pytest.raises(ValueError, match="end_time must lie within the schedule"):
        video_range(1, 5, 0, 4)
    with pytest.raises(ValueError, match="start_time must not exceed end_time"):
        video_range(3, 1, 0, 4)
    with pytest.raises(TypeError, match="start_time must be an integer timestep"):
        video_range(cast("int", 1.5), 3, 0, 4)


_NODE_CHECK = r"""
const [playerPath, webmPath] = process.argv.slice(2);
const Player = require(playerPath);
const WebMWriter = require(webmPath);
const writer = new WebMWriter("V_VP9", 4, 2, 2);
writer.addFrame(Uint8Array.of(1, 2, 3), 0, true);
writer.addFrame(Uint8Array.of(4, 5), 500, false);
writer.addFrame(Uint8Array.of(6), 1000, true);
writer.finish().arrayBuffer().then((buffer) => {
  console.log(JSON.stringify({
    frames: Player.videoFrameTimes(10, 20, 5, 4),
    single: Player.videoFrameTimes(3, 3, 4, 30),
    rangeStart: Player.clampRange(7, 3, "start"),
    rangeEnd: Player.clampRange(7, 3, "end"),
    ordered: Player.clampRange(2, 5, "start"),
    next: Player.stepTarget([0, 2, 5], 2, 1, 0, 9),
    last: Player.stepTarget([0, 2, 5], 5, 1, 0, 9),
    previous: Player.stepTarget([0, 2, 5], 2, -1, 0, 9),
    webm: Buffer.from(buffer).toString("base64"),
  }));
});
"""


def _read_elements(data: bytes, start: int, end: int) -> list[tuple[int, int, int, int]]:
    """Split EBML bytes into ``(id, element start, payload start, payload end)`` entries."""
    elements = []
    position = start
    while position < end:
        element_start = position
        id_length = 9 - data[position].bit_length()
        element_id = int.from_bytes(data[position : position + id_length])
        position += id_length
        size_length = 9 - data[position].bit_length()
        size = int.from_bytes(data[position : position + size_length]) & ((1 << (7 * size_length)) - 1)
        position += size_length
        elements.append((element_id, element_start, position, position + size))
        position += size
    assert position == end
    return elements


def test_browser_player_matches_python_rules_and_writes_seekable_webm(tmp_path: Path) -> None:
    """Check the browser functions and the WebM writer with Node.js."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    check = tmp_path / "check.js"
    check.write_text(_NODE_CHECK, encoding="utf-8")

    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [node, str(check), str(_PLAYER.joinpath("player.js")), str(_PLAYER.joinpath("webm.js"))],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )

    output = json.loads(completed.stdout)
    assert output["frames"] == video_frame_times(10, 20, 5, 4)
    assert output["single"] == [3]
    assert (output["rangeStart"], output["rangeEnd"], output["ordered"]) == ([3, 3], [7, 7], [2, 5])
    assert (output["next"], output["last"], output["previous"]) == (5, 9, 0)
    webm = base64.b64decode(output["webm"])
    header, segment = _read_elements(webm, 0, len(webm))
    assert (header[0], segment[0]) == (0x1A45DFA3, 0x18538067)
    segment_data = segment[2]
    info, tracks, cues, *clusters = _read_elements(webm, segment_data, segment[3])
    assert (info[0], tracks[0], cues[0]) == (0x1549A966, 0x1654AE6B, 0x1C53BB6B)
    # A delta frame joins the cluster of its key frame, so two clusters hold three frames.
    assert [cluster[0] for cluster in clusters] == [0x1F43B675, 0x1F43B675]
    cue_positions = [
        int.from_bytes(webm[position[2] : position[3]])
        for cue in _read_elements(webm, cues[2], cues[3])
        for track in _read_elements(webm, cue[2], cue[3])
        if track[0] == 0xB7
        for position in _read_elements(webm, track[2], track[3])
        if position[0] == 0xF1
    ]
    assert [segment_data + position for position in cue_positions] == [cluster[1] for cluster in clusters]


# A minimal page for `Player.mount`: every element that the player looks up
# exists, records its children, style properties, and scrolling, and runs its
# listeners.
_PLAYER_NODE_CHECK = r"""
const [playerPath, dataJson] = process.argv.slice(2);

function makeElement(tag) {
  const listeners = {};
  const attributes = {};
  const element = {
    tag, hidden: false, textContent: "", value: "", className: "", children: [], dataset: {},
    styles: {},
    style: { setProperty: (name, value) => { element.styles[name] = value; } },
    classList: { toggle() {}, add() {}, remove() {} },
    clientWidth: 320, clientHeight: 200,
    setAttribute: (name, value) => { attributes[name] = String(value); },
    getAttribute: (name) => attributes[name],
    toggleAttribute: (name, on) => { if (name === "hidden") element.hidden = on; },
    addEventListener: (type, handler) => { (listeners[type] ??= []).push(handler); },
    click: () => (listeners.click ?? []).forEach((handler) => handler({ target: element })),
    dispatchEvent() {},
    append: (...items) => { element.children.push(...items); },
    replaceChildren: (...items) => { element.children = items; },
    getContext: () => new Proxy({}, { get: () => () => ({ width: 0 }), set: () => true }),
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 320, height: 200 }),
    scrolls: 0,
    scrollIntoView: () => { element.scrolls += 1; },
  };
  return element;
}

const elements = {};
const root = makeElement("main");
root.querySelector = (selector) => (elements[selector] ??= makeElement("div"));
root.querySelectorAll = () => [];
for (const name of ["sidebar", "sidebar-toggle", "edit-toggle", "edit-bar", "paint", "export"])
  root.querySelector(`.player-${name}`).hidden = true;
globalThis.document = { createElement: makeElement, body: { style: {} } };
globalThis.getComputedStyle = () => ({ backgroundColor: "#000000" });
globalThis.CustomEvent = class {};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.requestAnimationFrame = () => 0;

const Player = require(playerPath);
const drawing = {
  prepare: (data) => data,
  timeBounds: () => [0, 4],
  stepTimes: () => [0],
  drawAtTime() {},
  details: (view, time, settings) => [
    {
      title: "first",
      rows: [
        ["Theme", [settings.theme]],
        ["Ion", [["at", { text: "q0", border: "#ff0000" }, { text: "pz", fill: "#00ff00", ink: "#000000" }]]],
        ["Empty", []],
      ],
    },
  ],
};
const element = (name) => root.querySelector(`.player-${name}`);
const describe = (node) =>
  typeof node === "string"
    ? node
    : node.className === "player-pill"
      ? { pill: node.textContent, styles: node.styles }
      : { tag: node.tag, text: node.textContent, children: node.children.map(describe) };
Player.mount(root, drawing, JSON.parse(dataJson));
const output = { buttonHidden: element("sidebar-toggle").hidden, label: element("sidebar-label").textContent };
output.closed = { hidden: element("sidebar").hidden, body: element("sidebar-body").children.length };
element("sidebar-toggle").click();
output.open = {
  hidden: element("sidebar").hidden,
  expanded: element("sidebar-toggle").getAttribute("aria-expanded"),
  body: element("sidebar-body").children.map(describe),
};
element("sidebar-close").click();
output.closedAgain = element("sidebar").hidden;
const exportPanel = () => ({
  hidden: element("export").hidden,
  expanded: element("export-toggle").getAttribute("aria-expanded"),
  scrolls: element("export").scrolls,
});
element("export-toggle").click();
output.exportOpen = exportPanel();
element("export-toggle").click();
output.exportClosed = exportPanel();
console.log(JSON.stringify(output));
"""


def _run_player_check(tmp_path: Path, view: dict[str, object]) -> dict[str, Any]:
    """Mount the player in Node.js, open and close its panels, and return what it showed."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    check = tmp_path / "player-check.js"
    check.write_text(_PLAYER_NODE_CHECK, encoding="utf-8")
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [node, str(check), str(_PLAYER.joinpath("player.js")), json.dumps(view)],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )
    return cast("dict[str, Any]", json.loads(completed.stdout))


def _term(text: str) -> dict[str, object]:
    return {"tag": "dt", "text": text, "children": []}


def _value(*lines: object) -> dict[str, object]:
    return {"tag": "dd", "text": "", "children": list(lines)}


def _line(*parts: object) -> dict[str, object]:
    return {"tag": "div", "text": "", "children": list(parts)}


def _text(text: str) -> dict[str, object]:
    return {"tag": "span", "text": text, "children": []}


def test_player_side_bar_opens_closes_and_writes_the_drawing_details(tmp_path: Path) -> None:
    """Open the named side bar with its button, fill it with sections, lines, and pills, and close it again."""
    view = view_data(
        title="Schedule",
        settings={**_SETTINGS, "theme": "dark", "show_notes": False},
        options=[],
        drawing=None,
        drawing_name="TestDrawing",
        sidebar=("show_notes", "Notes"),
    )

    output = _run_player_check(tmp_path, view)

    assert (output["buttonHidden"], output["label"]) == (False, "Notes")
    assert output["closed"] == {"hidden": True, "body": 0}
    assert output["open"]["hidden"] is False
    assert output["open"]["expanded"] == "true"
    [section] = output["open"]["body"]
    heading, terms = section["children"]
    assert heading["text"] == "first"
    assert terms["children"] == [
        _term("Theme"),
        _value(_line(_text("dark"))),
        _term("Ion"),
        _value(
            _line(
                _text("at"),
                {"pill": "q0", "styles": {"--pill-border": "#ff0000"}},
                {"pill": "pz", "styles": {"--pill-fill": "#00ff00", "--pill-ink": "#000000"}},
            )
        ),
        _term("Empty"),
        _value(_line(_text("none"))),
    ]
    assert output["closedAgain"] is True


def test_player_without_a_named_side_bar_keeps_its_button_hidden(tmp_path: Path) -> None:
    """Show no side bar button for views that name no side bar."""
    output = _run_player_check(tmp_path, _view())

    assert output["buttonHidden"] is True
    assert output["closed"] == {"hidden": True, "body": 0}


def test_export_button_opens_the_export_panel_in_view(tmp_path: Path) -> None:
    """Open the export panel and scroll it into view; a second click closes it."""
    output = _run_player_check(tmp_path, _view())

    assert output["exportOpen"] == {"hidden": False, "expanded": "true", "scrolls": 1}
    assert output["exportClosed"] == {"hidden": True, "expanded": "false", "scrolls": 1}
