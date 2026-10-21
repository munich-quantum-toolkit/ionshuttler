# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the colors and text sizes that every drawing shares."""

from __future__ import annotations

import json
import shutil
import subprocess
from importlib.resources import files
from typing import TYPE_CHECKING, Any, cast

import pytest

from mqt.ionshuttler.visualization import IonColor
from mqt.ionshuttler.visualization._player.style import (
    DARK_TEXT,
    ION_PALETTE,
    NEUTRAL_BORDER,
    WHITE,
    color_changes,
    color_mode_option,
    distinct_color,
    ion_colors,
    ion_label_font_size,
    shade,
    truncate,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_custom_colors_become_time_ordered_changes_and_a_color_choice() -> None:
    """List each ion's colors in time order and offer the custom mode only for custom colors."""
    custom = {2: [(0, IonColor(border="red", label="data")), (4, IonColor(fill="black"))]}

    assert color_changes(custom) == {2: [(0, "#ff0000", None, "data"), (4, None, "#000000", None)]}
    assert color_changes("single") == {}
    assert [value for value, _label in color_mode_option(custom)[2]] == ["distinct", "single", "custom"]
    assert [value for value, _label in color_mode_option("distinct")[2]] == ["distinct", "single"]


def test_shade_darkens_toward_black() -> None:
    """Keep a color at amount 0 and reach black at amount 1."""
    assert shade("#ffffff", 0) == "#ffffff"
    assert shade("#ffffff", 0.14) == "#dbdbdb"
    assert shade("#2563eb", 1) == "#000000"


def test_text_fits_inside_ions_and_long_text_is_shortened() -> None:
    """Shrink long ion labels and shorten text to its width."""
    assert ion_label_font_size(10, "q1") == pytest.approx(9.5)
    assert ion_label_font_size(10, "q100") == pytest.approx(6.25)
    assert truncate("short", 100, 10) == "short"
    assert truncate("a long description of the layer", 55, 10) == "a long de…"


_NODE_CHECK = r"""
const PlayerStyle = require(process.argv[2]);
const changes = JSON.parse(process.argv[3]);
const view = { ionChanges: changes, ionEdits: {} };
const output = {};
for (const mode of ["distinct", "single", "custom"]) {
  output[mode] = [0, 1, 2, 30].flatMap((ion) =>
    [0, 1.9, 2].map((time) => PlayerStyle.ionColors(view, ion, time, mode, "#123456", "#1b1f25")),
  );
}
output.sizes = [["q1", 10], ["q100", 10]].map(([label, radius]) => PlayerStyle.ionLabelFontSize(radius, label));
output.truncated = PlayerStyle.truncate("a long description of the layer", 55, 10);
console.log(JSON.stringify(output));
"""


def test_browser_style_matches_the_python_style(tmp_path: Path) -> None:
    """Pick the same ion colors and text sizes in the browser and in Python."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    changes = {1: [(0, "#999999", None, "idle"), (2, "#2ca02c", "#000000", None)]}
    check = tmp_path / "check.js"
    check.write_text(_NODE_CHECK, encoding="utf-8")
    script = str(files("mqt.ionshuttler.visualization._player").joinpath("style.js"))

    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [node, str(check), script, json.dumps(changes)],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )

    output = json.loads(completed.stdout)
    for mode in ("distinct", "single", "custom"):
        expected = [
            dict(
                zip(("border", "fill", "text"), ion_colors(ion, time, mode, "#123456", changes, "#1b1f25"), strict=True)
            )
            for ion in (0, 1, 2, 30)
            for time in (0, 1.9, 2)
        ]
        assert output[mode] == expected
    assert output["sizes"] == pytest.approx([ion_label_font_size(10, "q1"), ion_label_font_size(10, "q100")])
    assert output["truncated"] == truncate("a long description of the layer", 55, 10)
