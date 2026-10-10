# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the interactive Grid view, its drawing data, and Matplotlib output."""

from __future__ import annotations

import base64
import html
import json
import math
import re
import shutil
import subprocess
from importlib.resources import files
from typing import TYPE_CHECKING, Any, cast

import pytest

from mqt.ionshuttler import visualize
from mqt.ionshuttler.core.gates import Rxx, Rz
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.grid import (
    GridArchitecture,
    GridMachineState,
    Junction,
    JunctionMove,
    ProcessingZone,
    Segment,
)
from mqt.ionshuttler.visualization import GridView, GridVisualizer, IonColor
from mqt.ionshuttler.visualization._player.settings import video_frame_times
from mqt.ionshuttler.visualization.grid._colors import (
    DARK_TEXT,
    ION_PALETTE,
    NEUTRAL_BORDER,
    WHITE,
    distinct_color,
    ion_colors,
)
from mqt.ionshuttler.visualization.grid._scene import build_scene, geometry, track_position

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.ionshuttler.core.actions import Action


def _result_from_layers(
    architecture: GridArchitecture,
    placement: dict[str, tuple[int, ...]],
    actions: list[tuple[int, Action, str | None]],
) -> CompilationResult:
    """Build a validated result from ``(start time, action, processing zone)`` entries."""
    scheduled: list[ScheduledAction[Action]] = [
        ScheduledAction(index, action, start, architecture.action_duration(action), zone)
        for index, (start, action, zone) in enumerate(actions)
    ]
    initial_state = architecture.initial_state(placement)
    schedule: Schedule[Action, GridMachineState] = Schedule(
        tuple(scheduled),
        max((item.end_time for item in scheduled), default=0),
        initial_state,
    )
    return CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=schedule,
        architecture=architecture,
        final_state=architecture.replay_schedule(schedule),
    )


def _line_result() -> CompilationResult:
    """Move one ion from the left segment to the right segment."""
    left = Segment("left")
    right = Segment("right")
    architecture = GridArchitecture(
        (left, right),
        (
            Junction("west", (left.start,)),
            Junction("center", (left.end, right.start)),
            Junction("east", (right.end,)),
        ),
    )
    return _result_from_layers(architecture, {"left": (0,)}, [(0, JunctionMove(left.end, right.start, (0,)), None)])


def _busy_result() -> CompilationResult:
    """Run two junction moves and a two-ion gate in one layer, then a virtual gate.

    The layout is a horizontal line ``left - middle - right - far`` with a
    ``top`` segment above the left junction. ``middle`` holds two processing
    zones.
    """
    left = Segment("left")
    middle = Segment("middle", capacity=2, processing_zones=(ProcessingZone("pz-a"), ProcessingZone("pz-b")))
    right = Segment("right")
    far = Segment("far")
    top = Segment("top")
    architecture = GridArchitecture(
        (left, middle, right, far, top),
        (
            Junction("west", (left.start,)),
            Junction("j1", (left.end, middle.start, top.start)),
            Junction("j2", (middle.end, right.start)),
            Junction("j3", (right.end, far.start)),
            Junction("north", (top.end,)),
            Junction("east", (far.end,)),
        ),
    )
    return _result_from_layers(
        architecture,
        {"left": (0,), "middle": (1, 2), "right": (3,)},
        [
            (0, JunctionMove(left.end, top.start, (0,)), None),
            (0, JunctionMove(right.end, far.start, (3,)), None),
            (0, Rxx(1, 2, 0.5), "pz-a"),
            (2, Rz(1, 0.25), None),
        ],
    )


_BUSY_COORDINATES: dict[str, tuple[float, float]] = {
    "west": (0, 0),
    "j1": (1, 0),
    "j2": (2, 0),
    "j3": (3, 0),
    "north": (1, 1),
    "east": (4, 0),
}


def _embedded(view: GridView) -> dict[str, Any]:
    """Decode the drawing data and settings embedded in a view."""
    match = re.search(r'class="player-data">\s*([^<]*?)\s*</script>', view.to_html())
    assert match is not None
    return cast("dict[str, Any]", json.loads(base64.b64decode(match.group(1)).decode("utf-8")))


def _close(figure: object) -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pyplot.close(cast("Any", figure))


def test_visualize_returns_an_interactive_view_for_grid_results() -> None:
    """Dispatch Grid results to the HTML view."""
    view = visualize(_line_result())

    assert isinstance(view, GridView)
    assert (view.start_time, view.end_time) == (0, 1)


def test_grid_visualizer_rejects_other_results() -> None:
    """Report a clear type error for results without Grid hardware."""
    result = _line_result()
    other = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=result.schedule,
        architecture=cast("GridArchitecture", object()),
        final_state=result.final_state,
    )

    with pytest.raises(TypeError, match="requires a Grid compilation result"):
        GridVisualizer().visualize(other)


def test_grid_visualizer_accepts_junction_objects_and_ids() -> None:
    """Place junctions through explicit coordinates keyed by IDs or junctions."""
    result = _line_result()
    junctions = {junction.junction_id: junction for junction in result.architecture.junctions}
    scene = build_scene(result, {junctions["west"]: (0, 0), "center": (1, 0), junctions["east"]: (2, 0)})

    positions = dict(scene.junctions)
    assert positions["west"][0] < positions["center"][0] < positions["east"][0]
    assert positions["west"][1] == positions["center"][1] == positions["east"][1]


def test_grid_visualizer_rejects_incomplete_coordinates() -> None:
    """Require explicit coordinates to cover exactly the architecture junctions."""
    with pytest.raises(ValueError, match="missing: center, east"):
        GridVisualizer({"west": (0, 0)}).visualize(_line_result())


def test_automatic_layout_recovers_generated_grid_rows_and_columns() -> None:
    """Draw generated rectangular grids in their row and column layout."""
    from mqt.ionshuttler.grid.layouts import rectangular_grid

    architecture = rectangular_grid(2, 3)
    result = _result_from_layers(architecture, {}, [])

    positions = dict(build_scene(result, None).junctions)

    assert positions["j:0:0"][1] == positions["j:0:2"][1] > positions["j:1:0"][1]
    assert positions["j:0:0"][0] == positions["j:1:0"][0] < positions["j:0:1"][0] < positions["j:0:2"][0]


def test_scene_stores_bounds_states_and_layer_timing() -> None:
    """Record the schedule bounds, initial and final ion positions, and layers."""
    result = _busy_result()
    scene = build_scene(result, _BUSY_COORDINATES)
    midpoints = {
        segment.segment_id: _midpoint(*ends)
        for segment, ends in zip(scene.segments, scene.geometry.segments, strict=True)
    }
    min_x, min_y, max_x, max_y = scene.geometry.bounds

    assert (scene.start_time, scene.end_time) == (0, 2)
    assert max_x - min_x > max_y - min_y > 0
    assert [(layer.start, layer.end) for layer in scene.layers] == [(0, 2), (2, 2)]
    initial = scene.ion_positions(0)
    final = scene.ion_positions(2)
    assert initial[0] == midpoints["left"]
    assert final[0] == midpoints["top"]
    assert initial[3] == midpoints["right"]
    assert final[3] == midpoints["far"]
    assert initial[1] == final[1]
    assert initial[2] == final[2]


def _midpoint(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    return ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)


def test_scene_moves_transported_ions_through_their_junction() -> None:
    """Move ions through the crossed junction, with all moves of a layer in step."""
    scene = build_scene(_busy_result(), _BUSY_COORDINATES)
    names = [junction_id for junction_id, _position in scene.junctions]
    junctions = dict(scene.junctions)
    left_ion = scene.ions[0]
    right_ion = scene.ions[3]

    assert [movement.via for movement in left_ion.movements] == [names.index("j1")]
    assert [movement.via for movement in right_ion.movements] == [names.index("j3")]
    halfway = scene.ion_positions(0.5)
    assert math.dist(halfway[0], junctions["j1"]) < 1e-9
    assert math.dist(halfway[3], junctions["j3"]) < 1e-9


def test_scene_positions_follow_moved_junctions() -> None:
    """Derive segment ends and ion positions from the junction coordinates alone."""
    scene = build_scene(_busy_result(), _BUSY_COORDINATES)
    names = [junction_id for junction_id, _position in scene.junctions]
    moved = [position for _junction_id, position in scene.junctions]
    moved[names.index("north")] = (0.25, 0.75)
    track = scene.ions[0]

    shape = geometry(scene, moved)

    assert track_position(track, 2, shape) == ((0.25 + 0.25) / 2, (0.0 + 0.75) / 2)
    assert shape.bounds[3] > scene.geometry.bounds[3]


def test_scene_highlights_running_gates_and_their_processing_zone() -> None:
    """Highlight gate ions and processing zones only while the gate runs."""
    scene = build_scene(_busy_result(), _BUSY_COORDINATES)

    assert scene.gate_ions(0) == scene.gate_ions(1.9) == frozenset({1, 2})
    assert scene.active_processing_zones(1) == frozenset({"pz-a"})
    assert scene.gate_ions(2) == frozenset({1})
    assert scene.active_processing_zones(2) == frozenset()


def test_scene_keeps_processing_zone_order_along_the_segment() -> None:
    """Draw processing zones in their segment order from start to end."""
    scene = build_scene(_busy_result(), _BUSY_COORDINATES)

    zones = [(zone.zone_id, scene.geometry.position(zone.place)[0]) for zone in scene.processing_zones]
    assert [zone_id for zone_id, _x in zones] == ["pz-a", "pz-b"]
    assert zones[0][1] < zones[1][1]


def test_simultaneous_moves_and_gates_form_one_layer() -> None:
    """Show all actions with one start time as one described layer."""
    scene = build_scene(_busy_result(), _BUSY_COORDINATES)
    first = scene.layers[0]

    assert first.description.count("move") == 2
    assert "rxx q1,q2 @ pz-a" in first.description
    assert scene.current_layer(1) == first
    assert scene.current_layer(2) == scene.layers[1]


def test_view_size_does_not_depend_on_frame_rate_or_speed() -> None:
    """Send the schedule once instead of one entry per drawn frame."""
    result = _busy_result()
    slow = _embedded(GridVisualizer(frames_per_second=1, timesteps_per_second=0.1).visualize(result))
    fast = _embedded(GridVisualizer(frames_per_second=240, timesteps_per_second=100).visualize(result))

    assert slow["drawing"]["panels"] == fast["drawing"]["panels"]


def test_view_embeds_every_setting() -> None:
    """Start the browser controls from the Python settings."""
    visualizer = GridVisualizer(
        theme="dark",
        show_ion_labels=False,
        show_processing_zone_labels=False,
        show_hardware_ids=True,
        timesteps_per_second=2.5,
        width=640,
        height=480,
        frames_per_second=24,
        video_start_time=1,
        video_end_time=2,
    )

    settings = _embedded(visualizer.visualize(_busy_result()))["settings"]

    assert settings == {
        "theme": "dark",
        "show_ion_labels": False,
        "show_processing_zone_labels": False,
        "show_hardware_ids": True,
        "timesteps_per_second": 2.5,
        "width": 640,
        "height": 480,
        "frames_per_second": 24,
        "video_start_time": 1,
        "video_end_time": 2,
        "ion_colors": "distinct",
    }


def test_views_start_in_the_dark_theme() -> None:
    """Use the dark theme unless the settings choose another one."""
    assert _embedded(GridVisualizer().visualize(_busy_result()))["settings"]["theme"] == "dark"


def test_video_range_defaults_to_the_schedule_bounds() -> None:
    """Select the whole schedule for export unless the settings say otherwise."""
    settings = _embedded(GridVisualizer().visualize(_busy_result()))["settings"]

    assert (settings["video_start_time"], settings["video_end_time"]) == (0, 2)


@pytest.mark.parametrize(
    ("settings", "error", "message"),
    [
        ({"theme": "blue"}, ValueError, "theme must be one of"),
        ({"show_ion_labels": 1}, TypeError, "show_ion_labels must be a boolean"),
        ({"show_processing_zone_labels": None}, TypeError, "show_processing_zone_labels must be a boolean"),
        ({"show_hardware_ids": "yes"}, TypeError, "show_hardware_ids must be a boolean"),
        ({"timesteps_per_second": 0}, ValueError, "timesteps_per_second must be positive"),
        ({"timesteps_per_second": math.inf}, ValueError, "timesteps_per_second must be positive and finite"),
        ({"timesteps_per_second": True}, TypeError, "timesteps_per_second must be a number"),
        ({"frames_per_second": -1}, ValueError, "frames_per_second must be positive"),
        ({"frames_per_second": 29.97}, TypeError, "frames_per_second must be an integer"),
        ({"width": 0}, ValueError, "width must be positive"),
        ({"height": 2.5}, TypeError, "height must be an integer"),
        ({"video_start_time": -1}, ValueError, "video_start_time must be non-negative"),
        ({"video_end_time": 1.5}, TypeError, "video_end_time must be an integer timestep"),
        ({"video_start_time": 3, "video_end_time": 2}, ValueError, "must not exceed"),
        ({"ion_colors": "rainbow"}, ValueError, "ion_colors must be 'distinct', 'single', or a mapping"),
        ({"ion_colors": {-1: IonColor()}}, TypeError, "keys must be non-negative ion IDs"),
        ({"ion_colors": {0: "red"}}, TypeError, "must be an IonColor or"),
        ({"ion_colors": {0: [(5, IonColor()), (2, IonColor())]}}, ValueError, "color times of ion 0 must increase"),
        ({"processing_zone_colors": {"pz": "not a color"}}, ValueError, "is not a known color"),
    ],
)
def test_grid_visualizer_validates_settings(settings: dict[str, object], error: type[Exception], message: str) -> None:
    """Reject invalid settings when the visualizer is created."""
    with pytest.raises(error, match=message):
        GridVisualizer(**cast("Any", settings))


def test_video_range_must_lie_within_the_schedule() -> None:
    """Reject export ranges that leave the schedule or run backwards."""
    result = _busy_result()
    view = GridVisualizer().visualize(result)

    with pytest.raises(ValueError, match="end_time must lie within the schedule"):
        GridVisualizer(video_end_time=3).visualize(result)
    with pytest.raises(ValueError, match="end_time must lie within the schedule"):
        view.export_video("out.gif", end_time=5)
    with pytest.raises(ValueError, match="start_time must not exceed end_time"):
        view.export_video("out.gif", start_time=2, end_time=1)


def test_document_is_a_standalone_player_with_the_grid_drawing() -> None:
    """Combine the shared player with the Grid drawing in one offline page."""
    document = GridVisualizer().visualize(_busy_result()).to_html()

    assert document.startswith("<!doctype html>")
    assert "const GridDrawing" in document
    assert "Player.start(GridDrawing);" in document
    assert not re.search(r"<script[^>]*\bsrc=|<link\b|https?://", document)


def test_document_offers_the_grid_label_options() -> None:
    """Offer one checkbox for each Grid label setting and the ion color modes."""
    options = _embedded(GridVisualizer().visualize(_busy_result()))["options"]

    assert options == [
        ["show_ion_labels", "Ion labels"],
        ["show_processing_zone_labels", "Processing-zone labels"],
        ["show_hardware_ids", "Segment and junction IDs"],
        ["ion_colors", "Ion colors", [["distinct", "Distinct"], ["single", "Single"]]],
    ]


def test_custom_ion_colors_reach_the_view() -> None:
    """Embed the color changes of chosen ions and offer the custom mode."""
    visualizer = GridVisualizer(
        ion_colors={
            0: IonColor(border="crimson", label="data"),
            1: [(0, IonColor(border="#999999", label="idle ancilla")), (2, IonColor(border="tab:green", fill="black"))],
        },
        processing_zone_colors={"pz-a": "teal"},
    )

    data = _embedded(visualizer.visualize(_busy_result()))

    assert data["settings"]["ion_colors"] == "custom"
    assert data["options"][-1][2][-1] == ["custom", "Custom"]
    assert data["drawing"]["ion_colors"] == {
        "0": [[0, "#dc143c", None, "data"]],
        "1": [[0, "#999999", None, "idle ancilla"], [2, "#2ca02c", "#000000", None]],
    }
    assert data["drawing"]["zone_colors"] == {"pz-a": "#008080"}


def test_ion_colors_follow_the_mode_and_switch_at_their_times() -> None:
    """Pick distinct, single, or custom colors, and switch custom colors at their times."""
    changes = {1: [(0, "#999999", None, "idle"), (2, "#2ca02c", "#000000", None)]}

    assert ion_colors(0, 0, "distinct", "#123456", changes) == (ION_PALETTE[0], WHITE, DARK_TEXT)
    assert ion_colors(30, 0, "distinct", "#123456", changes)[0] == distinct_color(30, ION_PALETTE)
    assert ion_colors(30, 0, "single", "#123456", changes)[0] == "#123456"
    assert ion_colors(1, 1.9, "custom", "#123456", changes) == ("#999999", WHITE, DARK_TEXT)
    assert ion_colors(1, 2, "custom", "#123456", changes) == ("#2ca02c", "#000000", WHITE)
    assert ion_colors(5, 2, "custom", "#123456", changes) == (NEUTRAL_BORDER, WHITE, DARK_TEXT)
    assert ion_colors(5, 2, "custom", "#123456", changes, "#1b1f25") == (NEUTRAL_BORDER, "#1b1f25", WHITE)
    assert len({distinct_color(index, ION_PALETTE) for index in range(60)}) == 60


def test_ion_color_validates_colors_and_labels() -> None:
    """Reject unknown colors and labels that are not text."""
    with pytest.raises(ValueError, match="border 'nope' is not a known color"):
        IonColor(border="nope")
    with pytest.raises(TypeError, match="label must be a string"):
        IonColor(label=cast("Any", 3))


def test_plot_draws_custom_ion_colors_and_a_legend() -> None:
    """Color ion rings and fills in Matplotlib figures and list labeled colors in a legend."""
    visualizer = GridVisualizer(
        ion_colors={1: [(0, IonColor(border="#999999", label="idle")), (2, IonColor(border="#2ca02c", fill="black"))]}
    )

    early = visualizer.plot(_busy_result(), 1)
    late = visualizer.plot(_busy_result(), 2)

    def ion_ring(figure: Any) -> tuple[float, ...]:
        rings = [collection for collection in figure.axes[0].collections if len(collection.get_offsets()) == 4]
        return tuple(round(value, 3) for value in rings[-1].get_edgecolor()[1][:3])

    assert ion_ring(early) == (0.6, 0.6, 0.6)
    assert ion_ring(late) == (round(0x2C / 255, 3), round(0xA0 / 255, 3), round(0x2C / 255, 3))
    assert any(text.get_text() == "idle" for text in early.axes[0].texts)
    _close(early)
    _close(late)


def test_embedded_identifiers_cannot_break_out_of_the_document() -> None:
    """Encode user-controlled IDs and titles so they cannot form markup."""
    hostile = "</script><script>alert(1)</script>"
    segment = Segment(hostile, processing_zones=(ProcessingZone("<b>zone</b>"),))
    architecture = GridArchitecture((segment,), (Junction("<img src=x>", (segment.start,)),))
    result = _result_from_layers(architecture, {hostile: (0,)}, [])

    view = GridVisualizer().compare({"<i>first</i>": result, "second": result})
    document = view.to_html()

    assert document.count("</script>") == 2
    assert "<img" not in document
    assert "<i>" not in document
    data = _embedded(view)
    assert data["drawing"]["panels"][0]["title"] == "<i>first</i>"
    assert data["drawing"]["panels"][0]["segments"][0][0] == hostile


def test_notebook_display_wraps_the_document_in_a_frame() -> None:
    """Show the complete document in an isolated frame inside notebooks."""
    view = GridVisualizer().visualize(_line_result())

    fragment = view._repr_html_()

    match = re.fullmatch(r'<iframe srcdoc="([^"]*)" title="Schedule view" style="[^"]*"></iframe>', fragment)
    assert match is not None
    assert html.unescape(match.group(1)) == view.to_html()


def test_save_writes_the_standalone_document(tmp_path: Path) -> None:
    """Write the same document that the view returns."""
    view = GridVisualizer().visualize(_line_result())

    path = view.save(tmp_path / "schedule.html")

    assert path.read_text(encoding="utf-8") == view.to_html()


def test_compare_shows_results_side_by_side() -> None:
    """Create one titled panel per result with its own schedule bounds."""
    view = GridVisualizer().compare({"line": _line_result(), "busy": _busy_result()})

    data = _embedded(view)
    assert [(panel["title"], panel["start"], panel["end"]) for panel in data["drawing"]["panels"]] == [
        ("line", 0, 1),
        ("busy", 0, 2),
    ]
    assert (view.start_time, view.end_time) == (0, 2)


def test_compare_needs_two_titled_results() -> None:
    """Reject comparisons without a second result or without titles."""
    with pytest.raises(ValueError, match="at least two results"):
        GridVisualizer().compare({"only": _line_result()})
    with pytest.raises(ValueError, match="non-empty"):
        GridVisualizer().compare({"": _line_result(), "other": _line_result()})


def test_compare_accepts_coordinates_for_each_architecture() -> None:
    """Draw differently shaped architectures with their own explicit coordinates."""
    line = _line_result()
    busy = _busy_result()

    view = GridVisualizer().compare(
        {"line": line, "busy": busy},
        junction_coordinates={"busy": _BUSY_COORDINATES},
    )

    panels = _embedded(view)["drawing"]["panels"]
    busy_junctions = {junction_id: (x, y) for junction_id, x, y in panels[1]["junctions"]}
    assert busy_junctions["north"][1] > busy_junctions["j1"][1]
    assert busy_junctions["north"][0] == busy_junctions["j1"][0]
    with pytest.raises(ValueError, match="names unknown results: missing"):
        GridVisualizer().compare({"line": line, "busy": busy}, junction_coordinates={"missing": {}})


def test_empty_and_zero_duration_schedules_remain_viewable(tmp_path: Path) -> None:
    """Show schedules without actions or with only instant gates."""
    pytest.importorskip("PIL")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture((processor,), ())
    empty = _result_from_layers(architecture, {"processor": (0,)}, [])
    instant = _result_from_layers(architecture, {"processor": (0,)}, [(0, Rz(0, 0.5), None)])

    for result in (empty, instant):
        view = GridVisualizer().visualize(result)
        assert (view.start_time, view.end_time) == (0, 0)
        assert view.to_html()
        view.export_video(tmp_path / "frame.gif")
        _close(GridVisualizer().plot(result, 0))
    assert build_scene(instant, None).gate_ions(0) == frozenset({0})


def test_plot_draws_one_schedule_time() -> None:
    """Draw a Matplotlib figure of the configured size at the requested time."""
    figure = GridVisualizer(width=500, height=300).plot(_line_result(), 0.5)

    assert tuple(figure.get_size_inches() * figure.dpi) == (500, 300)
    assert any(text.get_text() == "t 0.5 / 1" for text in figure.axes[0].texts)
    _close(figure)


def test_plot_rejects_times_outside_the_schedule() -> None:
    """Accept only times within the schedule bounds."""
    with pytest.raises(ValueError, match="time must lie within the schedule"):
        GridVisualizer().plot(_line_result(), 2)


def test_video_export_updates_artists_without_rebuilding_the_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write one frame per frame time; more frames add no artists."""
    image = pytest.importorskip("PIL.Image")
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    created_axes: list[Axes] = []
    cleared_axes: list[Axes] = []
    original_add_axes = Figure.add_axes
    original_clear = Axes.clear

    def record_axes(figure: Figure, *args: Any, **kwargs: Any) -> Axes:
        axis = original_add_axes(figure, *args, **kwargs)
        created_axes.append(axis)
        return axis

    def record_clear(axis: Axes) -> None:
        cleared_axes.append(axis)
        original_clear(axis)

    monkeypatch.setattr(Figure, "add_axes", record_axes)
    monkeypatch.setattr(Axes, "clear", record_clear)
    view = GridVisualizer(width=200, height=120, timesteps_per_second=2).visualize(_busy_result())

    view.export_video(tmp_path / "one.gif", end_time=0)
    path = view.export_video(tmp_path / "busy.gif", frames_per_second=4)

    with image.open(path) as video:
        assert video.n_frames == len(video_frame_times(0, 2, 2, 4)) == 5
    one_frame, all_frames = created_axes
    assert len(all_frames.get_children()) == len(one_frame.get_children())
    assert cleared_axes == created_axes


def test_video_export_explains_missing_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Name the missing program when a format needs FFmpeg."""
    from matplotlib.animation import FFMpegWriter

    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda _cls: False))
    view = GridVisualizer().visualize(_line_result())

    with pytest.raises(RuntimeError, match="requires FFmpeg"):
        view.export_video(tmp_path / "line.mp4")


def test_video_export_rejects_unknown_formats(tmp_path: Path) -> None:
    """Name the supported video file suffixes."""
    with pytest.raises(ValueError, match="unsupported video file suffix"):
        GridVisualizer().visualize(_line_result()).export_video(tmp_path / "line.txt")


_NODE_CHECK = r"""
const fs = require("fs");
const [scriptPath, dataPath, timesJson] = process.argv.slice(2);
const GridDrawing = require(scriptPath);
const data = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const view = GridDrawing.prepare(data.drawing);
const panel = view.panels[0];
const times = JSON.parse(timesJson);

const calls = [];
const context = new Proxy({}, {
  get: (target, name) => (name in target ? target[name] : (...args) => calls.push([name, ...args])),
  set: (target, name, value) => { target[name] = value; return true; },
});
const drawnText = (settings) => {
  calls.length = 0;
  GridDrawing.drawAtTime(context, view, 1, settings, 400, 300);
  return calls.filter((call) => call[0] === "fillText").map((call) => call[1]);
};
const settings = { ...data.settings, theme: "light", editing: false, highlight: null };
const shape = GridDrawing.geometry(panel);
const output = {
  bounds: GridDrawing.timeBounds(view),
  steps: GridDrawing.stepTimes(view),
  shape: { bounds: shape.bounds, ionRadius: shape.ionRadius },
  positions: times.map((time) => panel.ions.map((track) => GridDrawing.trackPosition(track, time, shape))),
  gates: times.map((time) => GridDrawing.runningGates(panel, time).map((gate) => gate.ions)),
  labels: drawnText({ ...settings, show_ion_labels: true }),
  unlabeled: drawnText({ ...settings, show_ion_labels: false }),
  hardware: drawnText({ ...settings, show_hardware_ids: true }),
  editing: drawnText({ ...settings, editing: true }),
};

// Drag junction "north" to the drawn position of junction "j2", as a user would.
const north = panel.junctionIds.indexOf("north");
const target = panel.layout.point(panel.junctions[panel.junctionIds.indexOf("j2")]);
const handle = GridDrawing.pick(view, ...panel.layout.point(panel.junctions[north]));
GridDrawing.beginDrag(view, handle);
GridDrawing.dragTo(view, handle, ...target);
GridDrawing.endDrag(view, handle);
output.picked = handle === null ? null : panel.junctionIds[handle.junction];
output.dragged = panel.junctions[north];
output.movedIon = GridDrawing.trackPosition(panel.ions[0], 2, GridDrawing.geometry(panel));
GridDrawing.paint(view, { kind: "ion", panel, ion: 0 }, "border", "#ff0000");
GridDrawing.paint(view, { kind: "zone", panel, zone: "pz-b" }, "color", "#00ff00");
output.painted = GridDrawing.ionColors(view, 0, 1, "distinct", "#000000");
output.layout = GridDrawing.settingsText(view);
GridDrawing.resetLayout(view);
output.reset = panel.junctions[north];
output.unpainted = GridDrawing.ionColors(view, 0, 1, "distinct", "#000000");
output.colors = Object.fromEntries(
  ["distinct", "single", "custom"].map((mode) => [
    mode,
    [0, 1, 2, 20].flatMap((ion) => [0, 1, 2].map((time) => GridDrawing.ionColors(view, ion, time, mode, "#123456"))),
  ]),
);
console.log(JSON.stringify(output));
"""


def test_browser_drawing_matches_the_python_drawing_data(tmp_path: Path) -> None:
    """Check the browser drawing against the Python scene with Node.js."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    result = _busy_result()
    scene = build_scene(result, _BUSY_COORDINATES)
    view = GridVisualizer(_BUSY_COORDINATES, show_processing_zone_labels=False).visualize(result)
    data = tmp_path / "data.json"
    data.write_text(json.dumps(_embedded(view)), encoding="utf-8")
    check = tmp_path / "check.js"
    check.write_text(_NODE_CHECK, encoding="utf-8")
    script = str(files("mqt.ionshuttler.visualization.grid").joinpath("draw.js"))
    times = [0, 0.25, 0.5, 0.75, 1, 1.5, 2]

    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [node, str(check), script, str(data), json.dumps(times)],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )

    output = json.loads(completed.stdout)
    for time, positions in zip(times, output["positions"], strict=True):
        expected = scene.ion_positions(time)
        for track, position in zip(scene.ions, positions, strict=True):
            assert math.dist(expected[track.ion], position) < 1e-9
    assert [sorted(ion for ions in gates for ion in ions) for gates in output["gates"]] == [
        sorted(scene.gate_ions(time)) for time in times
    ]
    assert output["bounds"] == [0, 2]
    assert output["steps"] == [0, 2]
    ions = {"q0", "q1", "q2", "q3"}
    hardware_ids = {"left", "middle", "far", "j1", "north"}
    assert ions.issubset(output["labels"])
    assert not ions.intersection(output["unlabeled"])
    assert hardware_ids.issubset(output["hardware"])
    assert not hardware_ids.intersection(output["labels"])
    assert math.dist(output["shape"]["bounds"][:2], scene.geometry.bounds[:2]) < 1e-9
    assert math.dist(output["shape"]["bounds"][2:], scene.geometry.bounds[2:]) < 1e-9
    assert abs(output["shape"]["ionRadius"] - scene.geometry.ion_radius) < 1e-9


def test_browser_ion_colors_match_the_python_colors(tmp_path: Path) -> None:
    """Pick the same ion colors in the browser and in Matplotlib for every mode."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    visualizer = GridVisualizer(
        _BUSY_COORDINATES,
        ion_colors={1: [(0, IonColor(border="#999999")), (2, IonColor(border="#2ca02c", fill="black"))]},
    )
    data = tmp_path / "data.json"
    embedded = _embedded(visualizer.visualize(_busy_result()))
    data.write_text(json.dumps(embedded), encoding="utf-8")
    check = tmp_path / "check.js"
    check.write_text(_NODE_CHECK, encoding="utf-8")
    script = str(files("mqt.ionshuttler.visualization.grid").joinpath("draw.js"))

    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [node, str(check), script, str(data), "[0]"],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )

    output = json.loads(completed.stdout)
    changes = {
        int(ion): [tuple(change) for change in timeline] for ion, timeline in embedded["drawing"]["ion_colors"].items()
    }
    for mode in ("distinct", "single", "custom"):
        expected = [
            dict(zip(("border", "fill", "text"), ion_colors(ion, time, mode, "#123456", changes), strict=True))
            for ion in (0, 1, 2, 20)
            for time in (0, 1, 2)
        ]
        assert output["colors"][mode] == expected


def test_browser_layout_editing_moves_junctions_and_reports_coordinates(tmp_path: Path) -> None:
    """Drag a junction in the browser drawing, copy the coordinates, and reset them."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    result = _busy_result()
    scene = build_scene(result, _BUSY_COORDINATES)
    view = GridVisualizer(_BUSY_COORDINATES).visualize(result)
    data = tmp_path / "data.json"
    data.write_text(json.dumps(_embedded(view)), encoding="utf-8")
    check = tmp_path / "check.js"
    check.write_text(_NODE_CHECK, encoding="utf-8")
    script = str(files("mqt.ionshuttler.visualization.grid").joinpath("draw.js"))

    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [node, str(check), script, str(data), "[0]"],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )

    output = json.loads(completed.stdout)
    junctions = dict(scene.junctions)
    assert {"j1", "north", "west"}.issubset(output["editing"])
    assert output["picked"] == "north"
    assert math.dist(output["dragged"], junctions["j2"]) < 1e-3
    top_after_drag = ((junctions["j1"][0] + junctions["j2"][0]) / 2, (junctions["j1"][1] + junctions["j2"][1]) / 2)
    assert math.dist(output["movedIon"], top_after_drag) < 1e-3
    assert output["layout"].startswith("junction_coordinates={\n")
    assert '    "north": (0.5, 0),' in output["layout"]
    assert 'ion_colors={\n    0: IonColor(border="#ff0000"),\n},' in output["layout"]
    assert 'processing_zone_colors={\n    "pz-b": "#00ff00",\n},' in output["layout"]
    assert output["painted"]["border"] == "#ff0000"
    assert output["unpainted"]["border"] == ION_PALETTE[0]
    assert output["reset"] == list(junctions["north"])
