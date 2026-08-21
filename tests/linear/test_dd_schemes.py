# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for reusable dynamical-decoupling pulse schemes."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import GateSpec
from mqt.ionshuttler.linear.dd.schemes import (
    CDD_1,
    CDD_2,
    CDD_3,
    HAHN_ECHO,
    XY4,
    XY8,
    XY16,
    DDScheme,
    available_dd_schemes,
    get_dd_scheme,
    make_arbitrary_x_scheme,
    make_cdd_scheme,
    make_cpmg_scheme,
    make_xy4_scheme,
)


def test_hahn_uses_midpoint_and_closing_boundary() -> None:
    """Preserve the source half-open-window convention."""
    assert HAHN_ECHO.gate_specs == (GateSpec("Rx", pi), GateSpec("Rx", pi))
    assert HAHN_ECHO.gate_timesteps((3, 9)) == (6, 9)
    assert HAHN_ECHO.resolved_gate_timesteps((3, 9)) == (6, 9)
    assert HAHN_ECHO.resolved_gate_timesteps((0, 3)) is None


def test_scheme_resolution_handles_rounding_and_duplicate_boundaries() -> None:
    """Reject collisions after optional discrete-time rounding."""
    rounded = DDScheme(
        "rounded",
        (0.25, 0.5),
        (GateSpec("Rx", pi), GateSpec("Rx", pi)),
        allow_rounding=True,
    )
    assert rounded.resolved_gate_timesteps((0, 4)) == (1, 2)
    assert rounded.resolved_gate_timesteps((0, 1)) is None


@pytest.mark.parametrize("scheme", [XY4, XY8, XY16, CDD_1, CDD_2, CDD_3])
def test_registered_sequences_resolve_on_matching_windows(scheme: DDScheme) -> None:
    """Resolve every built-in sequence at its natural discrete length."""
    assert scheme.resolved_gate_timesteps((0, scheme.num_pulses)) == tuple(range(1, scheme.num_pulses + 1))
    assert get_dd_scheme(scheme.name) is scheme
    assert available_dd_schemes()[scheme.name] is scheme


def test_factories_preserve_source_axes_names_and_validation() -> None:
    """Construct source-compatible CPMG, arbitrary-X, XY, and CDD families."""
    assert make_cpmg_scheme(4).name == "cpmg4"
    assert make_arbitrary_x_scheme(3).name == "arbitrary_x_3"
    assert make_xy4_scheme(2).name == "xy4^2"
    assert tuple((spec.gate_name, spec.theta) for spec in XY4.gate_specs) == (
        ("Rx", pi),
        ("Ry", pi),
        ("Rx", pi),
        ("Ry", pi),
    )
    with pytest.raises(ValueError, match="even integer"):
        make_cpmg_scheme(3)
    with pytest.raises(ValueError, match="between 1 and 3"):
        make_cdd_scheme(4)


def test_scheme_validation_and_registry_copy() -> None:
    """Reject malformed schemes and prevent registry mutation."""
    with pytest.raises(ValueError, match="same length"):
        DDScheme("bad", (0.5,), (GateSpec("Rx", pi), GateSpec("Rx", pi)))
    registry = available_dd_schemes()
    registry.clear()
    assert get_dd_scheme("hahn") is HAHN_ECHO
    with pytest.raises(ValueError, match="unknown DD scheme"):
        get_dd_scheme("missing")
