# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the interactive Linear view, its drawing data, and Matplotlib output."""

from __future__ import annotations

import base64
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from importlib.resources import files
from math import pi
from typing import TYPE_CHECKING, Any, cast

import pytest

from mqt.ionshuttler import visualize
from mqt.ionshuttler.core.gates import GlobalGate, Rx, Rzz
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES, PhysicalSwap, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.result import LinearDiagnostics, SearchDiagnostics
from mqt.ionshuttler.linear.schedule import schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state
from mqt.ionshuttler.visualization import IonColor, LinearView, LinearVisualizer
from mqt.ionshuttler.visualization._player.settings import video_frame_times
from mqt.ionshuttler.visualization.linear._scene import (
    SWAP_LIFT,
    build_scene,
    panel_layout,
    rail_path,
    rail_runs,
    site_span,
    track_position,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.ionshuttler.linear.state import SearchTransition
    from mqt.ionshuttler.visualization.linear._scene import PanelLayout


def _result(
    path: list[SearchTransition],
    architecture: LinearArchitecture,
    positions: list[int],
    diagnostics: LinearDiagnostics | None = None,
) -> CompilationResult:
    schedule = schedule_from_path(
        path, create_initial_state(len(positions), architecture, initial_positions=positions), architecture
    )
    return CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=schedule,
        architecture=architecture,
        final_state=architecture.replay_schedule(schedule),
        diagnostics=diagnostics,
    )


def _busy_result() -> CompilationResult:
    """Swap two ions and shuttle a third, then run a two-ion, a one-ion, and a global gate.

    Four sites with a processing zone on sites 1 and 2 and a higher field at
    both ends. The swap takes three timesteps: ``rzz`` runs from 3 to 5, ``rx``
    from 5 to 6, and the global ``rx`` from 6 to 7. The compiler planned
    ``rx`` for zone ``pz-b``, but it ran in ``pz``.
    """
    architecture = LinearArchitecture(
        num_sites=4,
        processing_zones={"pz": [1, 2]},
        field_profile=FieldProfile(num_sites=4, site_field=((0, 2.0), (3, 2.0))),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    return _result(
        [
            PhysicalSwap(ion_a=0, ion_b=1, pos_a=0, pos_b=1),
            Shuttle(ion=2, src=3, dst=2),
            AdvanceTime(),
            AdvanceTime(),
            AdvanceTime(),
            Rzz(ion_a=0, ion_b=2, theta=0.2, gate_id=4),
            AdvanceTime(),
            AdvanceTime(),
            Rx(ion=0, theta=0.5, gate_id=5),
            AdvanceTime(),
            GlobalGate(gate_name="rx", theta=pi, ions=(0, 1, 2), gate_id=6),
            AdvanceTime(),
        ],
        architecture,
        [0, 1, 3],
        SearchDiagnostics(score=7, explored_nodes=42, preferred_gate_zones=((4, "pz"), (5, "pz-b"))),
    )


def _plain_result() -> CompilationResult:
    """Shuttle one ion on two sites with the same field everywhere."""
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [1]})
    return _result([Shuttle(ion=0, src=0, dst=1), AdvanceTime()], architecture, [0])


def _ring_result() -> CompilationResult:
    """Swap two ions across the edge between site 4 and site 0 of a five-site ring, then run gates there.

    The processing zone ``wrap`` holds sites 4 and 0. The swap takes three
    timesteps, ``rzz`` runs from 3 to 5, and the global ``rx`` from 5 to 6.
    """
    architecture = LinearArchitecture(
        num_sites=5,
        processing_zones={"wrap": [4, 0]},
        topology="ring",
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    return _result(
        [
            PhysicalSwap(ion_a=0, ion_b=1, pos_a=0, pos_b=4),
            Shuttle(ion=2, src=2, dst=3),
            AdvanceTime(),
            AdvanceTime(),
            AdvanceTime(),
            Rzz(ion_a=0, ion_b=1, theta=0.2),
            AdvanceTime(),
            AdvanceTime(),
            GlobalGate(gate_name="rx", theta=pi, ions=(0, 1, 2)),
            AdvanceTime(),
        ],
        architecture,
        [0, 4, 2],
    )


# A zigzag through the four sites of the busy result.
_ZIGZAG = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
# The corners of a square, clockwise from the lower left.
_SQUARE = {0: (0.0, 0.0), 1: (0.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 0.0)}


def _square_ring_result() -> CompilationResult:
    """Keep one ion on site 0 of a four-site ring."""
    architecture = LinearArchitecture(num_sites=4, processing_zones={"pz": [0, 1]}, topology="ring")
    return _result([AdvanceTime()], architecture, [0])


def _embedded(view: LinearView) -> dict[str, Any]:
    """Decode the drawing data and settings embedded in a view."""
    match = re.search(r'class="player-data">\s*([^<]*?)\s*</script>', view.to_html())
    assert match is not None
    return cast("dict[str, Any]", json.loads(base64.b64decode(match.group(1)).decode("utf-8")))


def test_visualize_returns_an_interactive_view_for_linear_results() -> None:
    """Show Linear results in the shared player with the Linear drawing."""
    view = visualize(_busy_result())

    assert isinstance(view, LinearView)
    document = view.to_html()
    assert "const LinearDrawing" in document
    assert "const PlayerStyle" in document
    assert "Player.start(LinearDrawing);" in document
    assert not re.search(r"<script[^>]*\bsrc=|<link\b|https?://", document)


def test_linear_visualizer_rejects_other_results() -> None:
    """Name the expected result type."""
    other = _busy_result()
    other = CompilationResult(
        status=other.status,
        schedule=other.schedule,
        architecture=cast("Any", object()),
        final_state=other.final_state,
    )

    with pytest.raises(TypeError, match="requires a Linear compilation result"):
        LinearVisualizer().visualize(other)


def test_scene_records_swaps_shuttles_and_gates() -> None:
    """Pass swapping ions above and below the chain and keep each gate's zones."""
    scene = build_scene(_busy_result(), None, "busy")

    tracks = {track.ion: track for track in scene.ions}
    assert [(move.start, move.end, move.target, move.side) for move in tracks[0].movements] == [(0, 3, 1, 1)]
    assert [(move.start, move.end, move.target, move.side) for move in tracks[1].movements] == [(0, 3, 0, -1)]
    assert [(move.start, move.end, move.target, move.side) for move in tracks[2].movements] == [(0, 1, 2, 0)]
    assert scene.layers[0].description == "swap q0 q1 0 ↔ 1; shuttle q2 3 → 2"
    assert [(gate.name, gate.ions, gate.processing_zone, gate.planned_zone) for gate in scene.gates] == [
        ("rzz", (0, 2), "pz", "pz"),
        ("rx", (0,), "pz", "pz-b"),
        ("global rx", (0, 1, 2), None, None),
    ]
    assert scene.field == (2.0, 1.0, 1.0, 2.0)
    assert scene.summary.swaps == 1
    assert scene.summary.shuttles == 1
    assert scene.summary.gates == 3
    assert scene.summary.statistics == (("Score", "7"), ("Makespan lower bound", "0"), ("Explored nodes", "42"))


@dataclass(frozen=True)
class _OtherMethodDiagnostics:
    """Diagnostics of another Linear compiler method with a zone plan and its own statistics."""

    preferred_gate_zones: tuple[tuple[int, str], ...] = ()
    reported: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return no data."""
        return {}

    def statistics(self) -> tuple[tuple[str, int | float | str], ...]:
        """Return the statistics given at construction."""
        return cast("tuple[tuple[str, int | float | str], ...]", self.reported)


def test_scene_reads_the_statistics_and_zone_plan_of_any_compiler_method() -> None:
    """Show the statistics and planned zones of diagnostics from another class."""
    diagnostics = _OtherMethodDiagnostics(((5, "pz-b"),), (("Rounds", 3), ("Gap", 0.123456), ("Order", "greedy")))

    scene = build_scene(replace(_busy_result(), diagnostics=diagnostics))

    assert scene.summary.statistics == (("Rounds", "3"), ("Gap", "0.1235"), ("Order", "greedy"))
    assert [gate.planned_zone for gate in scene.gates] == [None, "pz-b", None]
    assert build_scene(_plain_result()).summary.statistics == ()


@pytest.mark.parametrize(
    ("reported", "message"),
    [
        pytest.param((("Rounds",),), "must be \\(label, value\\) pairs", id="no-value"),
        pytest.param((("Rounds", None),), "needs a text label and a number or text value", id="no-number"),
        pytest.param((("Converged", True),), "needs a text label and a number or text value", id="flag"),
    ],
)
def test_scene_rejects_malformed_statistics(reported: tuple[Any, ...], message: str) -> None:
    """Name the expected shape of each statistic."""
    result = replace(_busy_result(), diagnostics=_OtherMethodDiagnostics(reported=reported))

    with pytest.raises(TypeError, match=message):
        build_scene(result)


def test_swapping_ions_pass_each_other_halfway() -> None:
    """Meet at the middle of the swap, one ion above the chain and one below."""
    scene = build_scene(_busy_result())
    tracks = {track.ion: track for track in scene.ions}

    assert track_position(tracks[0], 0) == (0.0, 0.0)
    assert track_position(tracks[0], 1.5) == pytest.approx((0.5, 1.0))
    assert track_position(tracks[1], 1.5) == pytest.approx((0.5, -1.0))
    assert track_position(tracks[0], 3) == (1.0, 0.0)
    assert track_position(tracks[2], 0.5) == (2.5, 0.0)


def test_running_gates_follow_the_schedule() -> None:
    """Report each gate while it runs and the layer that started last."""
    scene = build_scene(_busy_result())

    assert [gate.name for gate in scene.running_gates(4)] == ["rzz"]
    assert [gate.name for gate in scene.running_gates(6.5)] == ["global rx"]
    assert scene.running_gates(2) == ()
    layer = scene.current_layer(5.5)
    assert layer is not None
    assert layer.description == "rx q0 @ pz"


def test_uniform_fields_draw_no_profile() -> None:
    """Leave out the field profile and its switch when every site has the same field."""
    plain = _embedded(LinearVisualizer().visualize(_plain_result()))
    busy = _embedded(LinearVisualizer().visualize(_busy_result()))

    assert plain["drawing"]["panels"][0]["field"] is None
    assert ["show_field_profile", "Field profile"] not in plain["options"]
    assert ["show_field_profile", "Field profile"] in busy["options"]


def test_view_embeds_every_setting_and_offers_the_schedule_side_bar() -> None:
    """Start the browser controls from the Python settings."""
    visualizer = LinearVisualizer(
        theme="light",
        show_ion_labels=False,
        show_site_numbers=False,
        show_schedule_details=True,
        timesteps_per_second=2.5,
        width=640,
        height=300,
        marker_size=0.8,
        video_end_time=6,
    )

    data = _embedded(visualizer.visualize(_busy_result()))

    assert data["settings"] == {
        "theme": "light",
        "show_ion_labels": False,
        "show_processing_zone_labels": True,
        "show_site_numbers": False,
        "show_field_profile": False,
        "show_schedule_details": True,
        "marker_size": 0.8,
        "timesteps_per_second": 2.5,
        "width": 640,
        "height": 300,
        "frames_per_second": 30,
        "video_start_time": 0,
        "video_end_time": 6,
        "ion_colors": "distinct",
    }
    assert data["sidebar"] == ["show_schedule_details", "Schedule details"]
    assert data["drawing_name"] == "LinearDrawing"


@pytest.mark.parametrize(
    ("settings", "error", "message"),
    [
        ({"theme": "blue"}, ValueError, "theme must be one of"),
        ({"show_site_numbers": 1}, TypeError, "show_site_numbers must be a boolean"),
        ({"show_field_profile": None}, TypeError, "show_field_profile must be a boolean"),
        ({"show_schedule_details": "yes"}, TypeError, "show_schedule_details must be a boolean"),
        ({"marker_size": 0}, ValueError, "marker_size must be positive"),
        ({"ion_colors": "rainbow"}, ValueError, "ion_colors must be 'distinct', 'single', or a mapping"),
        ({"processing_zone_colors": {"pz": "not a color"}}, ValueError, "is not a known color"),
    ],
)
def test_linear_visualizer_validates_settings(
    settings: dict[str, object], error: type[Exception], message: str
) -> None:
    """Reject invalid settings when the visualizer is created."""
    with pytest.raises(error, match=message):
        LinearVisualizer(**cast("Any", settings))


def test_compare_stacks_results_with_their_titles() -> None:
    """Show one panel per result and refuse a single result."""
    view = LinearVisualizer().compare({"busy": _busy_result(), "plain": _plain_result()})

    data = _embedded(view)
    assert [panel["title"] for panel in data["drawing"]["panels"]] == ["busy", "plain"]
    assert data["title"] == "busy vs. plain"
    assert (view.start_time, view.end_time) == (0, 7)
    with pytest.raises(ValueError, match="at least two results"):
        LinearVisualizer().compare({"busy": _busy_result()})


def test_marker_size_scales_ions_without_moving_sites() -> None:
    """Keep every site in place and scale the ion radius."""
    scene = build_scene(_busy_result())
    full = panel_layout(scene, 0, 0, 960, 420)
    half = panel_layout(scene, 0, 0, 960, 420, 0.5)

    assert half.ion_radius == pytest.approx(full.ion_radius / 2)
    assert [half.point(site) for site in range(4)] == [full.point(site) for site in range(4)]
    assert 2 * full.ion_radius < full.spacing
    assert full.point(1, 1) == pytest.approx((full.sites[1][0], full.sites[1][1] - SWAP_LIFT * full.ion_radius))


def test_plot_draws_one_schedule_time() -> None:
    """Draw ions, gate names, the header, and the field profile only when asked."""
    pyplot = pytest.importorskip("matplotlib.pyplot")
    visualizer = LinearVisualizer(ion_colors={0: IonColor(border="crimson", label="data")}, show_field_profile=True)

    figure = visualizer.plot(_busy_result(), 4)
    plain = LinearVisualizer().plot(_busy_result(), 4)

    texts = [text.get_text() for text in figure.axes[0].texts if text.get_visible()]
    assert "rzz" in texts
    assert "t 4.0 / 7" in texts
    assert "rzz q0,q2 @ pz" in texts
    assert "data" in texts
    assert "field" in texts
    assert "field" not in [text.get_text() for text in plain.axes[0].texts]
    pyplot.close(plain)
    with pytest.raises(ValueError, match="time must lie within the schedule"):
        visualizer.plot(_busy_result(), 8)
    pyplot.close(figure)


def test_video_export_writes_one_frame_per_step(tmp_path: Path) -> None:
    """Write a GIF with the frames of the playback speed."""
    image = pytest.importorskip("PIL.Image")
    view = LinearVisualizer(width=200, height=120, timesteps_per_second=2).visualize(_busy_result())

    path = view.export_video(tmp_path / "busy.gif", end_time=2, frames_per_second=4)

    with image.open(path) as video:
        assert video.n_frames == len(video_frame_times(0, 2, 2, 4))


def test_ring_moves_cross_the_edge_between_the_last_site_and_site_zero() -> None:
    """Move ions over the short edge of the ring and keep a wrapping zone in one piece."""
    scene = build_scene(_ring_result())
    tracks = {track.ion: track for track in scene.ions}

    assert scene.ring
    # The ion that moves from site 4 to site 0 moves to the next site.
    assert [(move.target, move.side) for move in tracks[0].movements] == [(4, -1)]
    assert [(move.target, move.side) for move in tracks[1].movements] == [(0, 1)]
    assert scene.ion_positions(1.5)[0] == pytest.approx((4.5, -1.0))
    assert scene.ion_positions(1.5)[1] == pytest.approx((4.5, 1.0))
    assert scene.ion_positions(0.75)[0][0] == pytest.approx(4.75)
    assert scene.ion_positions(0.75)[1][0] == pytest.approx(4.25)
    assert rail_runs(scene) == [(-1, 0, "wrap"), (1, 3, None)]


def test_site_span_leaves_out_the_largest_gap_of_a_ring() -> None:
    """Cover a set of sites with the shortest stretch of the chain."""
    ring = build_scene(_ring_result())

    assert site_span([4, 0], ring) == (4, 5)
    assert site_span([0, 2, 4], ring) == (2, 5)
    assert site_span([1, 3], ring) == (1, 3)
    assert site_span([3, 1], build_scene(_busy_result())) == (1, 3)


def test_ring_fills_the_panel_with_an_ellipse() -> None:
    """Draw site 0 at the top, the other sites clockwise, and swapping ions on both sides of the ring."""
    layout = panel_layout(build_scene(_ring_result()), 0, 0, 960, 420)

    assert layout.ellipse is not None
    center_x, center_y, radius_x, radius_y = layout.ellipse
    assert radius_x > 2 * radius_y
    assert layout.sites[0] == pytest.approx((center_x, center_y - radius_y))
    assert layout.sites[1][0] > center_x
    assert layout.normal(0) == pytest.approx((0.0, -1.0))
    center = (center_x, center_y)
    assert (
        math.dist(layout.point(4.5, -1), center)
        < math.dist(layout.point(4.5), center)
        < math.dist(layout.point(4.5, 1), center)
    )
    name_distance = (SWAP_LIFT + 1) * layout.ion_radius + 0.5 * 9 + 2
    for site in range(5):
        x, y = layout.beside(site, name_distance)
        assert 0 <= x <= 960
        assert layout.header_height <= y <= 420


def test_site_coordinates_keep_their_shape() -> None:
    """Scale explicit coordinates evenly and turn ring normals outward in either direction."""
    result = _square_ring_result()
    scene = build_scene(result, _SQUARE)
    layout = panel_layout(scene, 0, 0, 960, 420)
    xs = [x for x, _y in layout.sites]
    ys = [y for _x, y in layout.sites]

    assert scene.coordinates == ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    assert build_scene(result, list(_SQUARE.values())).coordinates == scene.coordinates
    assert layout.ellipse is None
    assert max(xs) - min(xs) == pytest.approx(max(ys) - min(ys))
    # The left side of the square, from the lower left to the upper left corner.
    assert layout.normal(0.5) == pytest.approx((-1.0, 0.0))
    assert layout.normal(0) == pytest.approx((-math.sqrt(0.5), math.sqrt(0.5)))
    reversed_square = panel_layout(build_scene(result, [(0, 0), (1, 0), (1, 1), (0, 1)]), 0, 0, 960, 420)
    # The lower side of the square points down on the screen.
    assert reversed_square.normal(0.5) == pytest.approx((0.0, 1.0))


@pytest.mark.parametrize(
    ("coordinates", "error", "message"),
    [
        pytest.param({0: (0, 0)}, ValueError, r"missing: 1, 2, 3\)", id="missing"),
        pytest.param({**_SQUARE, 4: (2, 2)}, ValueError, r"unknown: 4\)", id="unknown"),
        pytest.param({**_SQUARE, 0: (0, math.nan)}, ValueError, "site 0 must be finite", id="not-finite"),
        pytest.param({**_SQUARE, 0: ("a", 0)}, TypeError, "site 0 must contain two numbers", id="text"),
        pytest.param({**_SQUARE, 0: (0,)}, ValueError, "site 0 must contain two numbers", id="one-number"),
        pytest.param({"0": (0, 0)}, TypeError, "keys must be site numbers", id="text-key"),
        pytest.param("abcd", TypeError, "mapping from site to coordinate", id="text-coordinates"),
    ],
)
def test_site_coordinates_must_name_every_site(coordinates: object, error: type[Exception], message: str) -> None:
    """Name the site and the problem of invalid coordinates."""
    visualizer = LinearVisualizer(site_coordinates=cast("Any", coordinates))

    with pytest.raises(error, match=message):
        visualizer.visualize(_square_ring_result())


def test_compare_uses_coordinates_per_result() -> None:
    """Give chosen panels their own coordinates and reject coordinates for unknown panels."""
    results = {"ring": _ring_result(), "zigzag": _busy_result()}

    data = _embedded(LinearVisualizer().compare(results, site_coordinates={"zigzag": _ZIGZAG}))

    ring, zigzag = data["drawing"]["panels"]
    assert (ring["ring"], ring["coordinates"]) == (True, None)
    assert (zigzag["ring"], zigzag["coordinates"]) == (False, [[0, 0], [0.3333, 0.3333], [0.6667, 0], [1, 0.3333]])
    with pytest.raises(ValueError, match="site_coordinates names unknown results: missing"):
        LinearVisualizer().compare(results, site_coordinates={"missing": _ZIGZAG})


def test_field_profile_needs_the_automatic_straight_chain() -> None:
    """Leave out the field profile and its switch for explicit coordinates."""
    data = _embedded(LinearVisualizer(site_coordinates=_ZIGZAG).visualize(_busy_result()))

    assert data["drawing"]["panels"][0]["field"] is None
    assert ["show_field_profile", "Field profile"] not in data["options"]


def test_plot_draws_rings_and_explicit_coordinates() -> None:
    """Draw zone names and gate names for a ring and for a zigzag chain."""
    pyplot = pytest.importorskip("matplotlib.pyplot")

    ring = LinearVisualizer().plot(_ring_result(), 4)
    zigzag = LinearVisualizer(site_coordinates=_ZIGZAG).plot(_busy_result(), 4)

    for figure, zone in ((ring, "wrap"), (zigzag, "pz")):
        texts = [text.get_text() for text in figure.axes[0].texts if text.get_visible()]
        assert {zone, "rzz", "q0", "0"}.issubset(texts)
        pyplot.close(figure)


def test_trajectory_plot_stays_available() -> None:
    """Draw every ion's site over the whole schedule."""
    pyplot = pytest.importorskip("matplotlib.pyplot")

    figure = LinearVisualizer.plot_trajectories(_busy_result())

    assert figure.axes[0].get_ylabel() == "site"
    pyplot.close(figure)


_NODE_CHECK = r"""
const fs = require("fs");
const [scriptPath, dataPath, timesJson] = process.argv.slice(2);
const LinearDrawing = require(scriptPath);
const data = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const view = LinearDrawing.prepare(data.drawing);
const panel = view.panels[0];
const times = JSON.parse(timesJson);
const calls = [];
const context = new Proxy({}, {
  get: (target, name) => {
    if (name in target) return target[name];
    if (name === "measureText") return () => ({ width: 10 });
    // Gate halos fill with radial gradients.
    if (name === "createRadialGradient") return () => ({ addColorStop() {} });
    return (...args) => calls.push([name, ...args]);
  },
  set: (target, name, value) => { target[name] = value; return true; },
});
const settings = { ...data.settings, theme: "dark", editing: false, highlight: null };
const drawnText = (time) => {
  calls.length = 0;
  LinearDrawing.drawAtTime(context, view, time, settings, 960, 420);
  return calls.filter((call) => call[0] === "fillText").map((call) => call[1]);
};
const layout = LinearDrawing.panelLayout(panel, 0, 0, 960, 420, 0.7);
const ringSize = panel.ring ? panel.sites : null;
const halfSites = Array.from({ length: panel.sites }, (_, site) => site + 0.5);
const output = {
  bounds: LinearDrawing.timeBounds(view),
  steps: LinearDrawing.stepTimes(view),
  positions: times.map((time) => panel.ions.map((track) => LinearDrawing.trackPosition(track, time, ringSize))),
  gates: times.map((time) => LinearDrawing.runningGates(panel, time).map((gate) => gate.name)),
  layout: {
    radius: layout.ionRadius,
    site: layout.point(2),
    lifted: layout.point(1, 1),
    sites: layout.sites,
    between: halfSites.map((site) => layout.point(site, 1)),
    labels: Array.from({ length: panel.sites }, (_, site) => layout.beside(site, -20)),
    rail: LinearDrawing.railPath(layout, -0.5, 1.5),
  },
  field: LinearDrawing.fieldPoints(panel, layout),
  text: drawnText(4),
  details: Object.fromEntries(
    times.map((time) => [time, LinearDrawing.details(view, time, settings)[0].rows]),
  ),
};
drawnText(4);
const pickedIon = LinearDrawing.pick(view, ...panel.ionPoints[0]);
// Between two ions of the first processing zone, next to the chain.
const zone = panel.zones[0].sites;
const zonePoint = panel.layout.point(zone.length > 1 ? zone[0] + 0.6 : zone[0]);
const pickedZone = LinearDrawing.pick(view, zonePoint[0], zonePoint[1] + 1);
output.picked = [pickedIon && pickedIon.ion, pickedZone && pickedZone.zone];
LinearDrawing.paint(view, pickedIon, "fill", "#000000");
LinearDrawing.paint(view, pickedZone, "color", "#00ff00");
output.settingsText = LinearDrawing.settingsText(view, { marker_size: 0.5 });
LinearDrawing.resetLayout(view);
output.resetText = LinearDrawing.settingsText(view, { marker_size: 1 });
console.log(JSON.stringify(output));
"""


def _run_node_check(tmp_path: Path, view: LinearView, times: list[float]) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    data = tmp_path / "data.json"
    data.write_text(json.dumps(_embedded(view)), encoding="utf-8")
    check = tmp_path / "check.js"
    check.write_text(_NODE_CHECK, encoding="utf-8")
    script = str(files("mqt.ionshuttler.visualization.linear").joinpath("draw.js"))
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [node, str(check), script, str(data), json.dumps(times)],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )
    return cast("dict[str, Any]", json.loads(completed.stdout))


def test_browser_drawing_matches_the_python_drawing_data(tmp_path: Path) -> None:
    """Check positions, gates, and layout of the browser drawing against the Python scene."""
    result = _busy_result()
    scene = build_scene(result)
    times = [0, 1.5, 3, 4, 5.5, 6.5, 7]

    output = _run_node_check(tmp_path, LinearVisualizer(show_field_profile=True).visualize(result), times)

    for time, positions in zip(times, output["positions"], strict=True):
        expected = scene.ion_positions(time)
        for track, position in zip(scene.ions, positions, strict=True):
            assert math.dist(expected[track.ion], position) < 1e-9
    assert output["gates"] == [[gate.name for gate in scene.running_gates(time)] for time in times]
    assert output["bounds"] == [0, 7]
    assert output["steps"] == [0, 3, 5, 6]
    _assert_same_layout(output["layout"], panel_layout(scene, 0, 0, 960, 420, 0.7))
    assert {"q0", "q1", "q2", "rzz", "field", "pz", "0", "3"}.issubset(output["text"])


def _assert_same_layout(browser: dict[str, Any], layout: PanelLayout) -> None:
    """Compare the browser layout with the Python layout at sites, between sites, beside sites, and on a rail."""

    def flat(points: list[Any]) -> list[float]:
        return [value for point in points for value in point]

    count = len(layout.sites)
    assert browser["radius"] == pytest.approx(layout.ion_radius)
    assert browser["site"] == pytest.approx(list(layout.point(2)))
    assert browser["lifted"] == pytest.approx(list(layout.point(1, 1)))
    assert flat(browser["sites"]) == pytest.approx(flat(list(layout.sites)))
    assert flat(browser["between"]) == pytest.approx(flat([layout.point(site + 0.5, 1) for site in range(count)]))
    assert flat(browser["labels"]) == pytest.approx(flat([layout.beside(site, -20) for site in range(count)]))
    assert flat(browser["rail"]) == pytest.approx(flat(rail_path(layout, -0.5, 1.5)))


def test_browser_ring_drawing_matches_the_python_drawing_data(tmp_path: Path) -> None:
    """Check positions, layout, labels, picking, and the side bar of a ring against the Python scene."""
    result = _ring_result()
    scene = build_scene(result)
    times = [0, 0.75, 1.5, 3, 4, 5.5]

    output = _run_node_check(tmp_path, LinearVisualizer().visualize(result), times)

    for time, positions in zip(times, output["positions"], strict=True):
        expected = scene.ion_positions(time)
        for track, position in zip(scene.ions, positions, strict=True):
            assert math.dist(expected[track.ion], position) < 1e-9
    _assert_same_layout(output["layout"], panel_layout(scene, 0, 0, 960, 420, 0.7))
    assert {"q0", "q1", "q2", "rzz", "wrap", "0", "4"}.issubset(output["text"])
    assert output["picked"] == [0, "wrap"]
    moving = dict(output["details"]["1.5"])["Moving"]
    assert [line[1:] for line in moving] == [["0 → 4", "swap"], ["4 → 0", "swap"]]
    assert dict(output["details"]["1.5"])["Ions"][0][1] == "site 4.5"


def test_browser_drawing_follows_explicit_coordinates(tmp_path: Path) -> None:
    """Place sites, lifted ions, labels, and rails at the same pixels as the Python layout."""
    result = _busy_result()

    output = _run_node_check(tmp_path, LinearVisualizer(site_coordinates=_ZIGZAG).visualize(result), [0])

    _assert_same_layout(output["layout"], panel_layout(build_scene(result, _ZIGZAG), 0, 0, 960, 420, 0.7))
    assert output["picked"] == [0, "pz"]


def test_browser_side_bar_lists_running_and_next_gates(tmp_path: Path) -> None:
    """List what runs, what comes next with its zone, the moving ions, and the outcome."""
    output = _run_node_check(tmp_path, LinearVisualizer().visualize(_busy_result()), [0.5, 4])

    early = dict(output["details"]["0.5"])
    assert early["Running"] == []
    assert early["Next"][0][:2] == ["t 3", "g4 rzz"]
    assert "planned pz-b" in early["Next"][1]
    assert [line[1] for line in early["Moving"]] == ["0 → 1", "1 → 0", "3 → 2"]
    assert [line[2] for line in early["Moving"]] == ["swap", "swap", "shuttle"]
    late = dict(output["details"]["4"])
    assert late["Running"][0][0] == "g4 rzz"
    assert late["Running"][0][-1] == "until t 5"
    first_ion = late["Ions"][0]
    assert (first_ion[1], first_ion[2]["text"], first_ion[3]) == ("site 1", "pz", "rzz")
    assert late["Run"][:2] == ["completed", "0 of 3 gates done"]
    assert late["Planned zones"] == ["1 of 2 gates ran in their planned zone"]
    assert late["Run"][3:6] == ["Score: 7", "Makespan lower bound: 0", "Explored nodes: 42"]


def test_browser_editor_colors_ions_and_zones(tmp_path: Path) -> None:
    """Pick ions and zones on the chain, report the changed colors, and undo them."""
    output = _run_node_check(tmp_path, LinearVisualizer().visualize(_busy_result()), [0])

    assert output["picked"] == [0, "pz"]
    assert output["settingsText"] == (
        'marker_size=0.5,\nion_colors={\n    0: IonColor(fill="#000000"),\n},\n'
        'processing_zone_colors={\n    "pz": "#00ff00",\n},'
    )
    assert output["resetText"] == "# The view uses the original settings."
