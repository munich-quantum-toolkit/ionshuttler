# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear hardware timing and compiler settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from mqt.ionshuttler.linear.config import (
    GateTiming,
    HardwareTiming,
    HeuristicMode,
    LinearCompilerConfig,
    SearchConfig,
    TransportTiming,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_compiler_config_uses_ready_to_run_defaults() -> None:
    """Provide ready-to-use timing and search settings."""
    config = LinearCompilerConfig()

    assert config.hardware_timing == HardwareTiming(
        transport=TransportTiming(shuttle=1, swap=3),
        gates=GateTiming(rx=1, ry=1, rz=0, rxx=2, ryy=2, rzz=2),
    )
    assert config.search == SearchConfig(
        horizon=3,
        committed_gates=2,
        iterative_diving_search=True,
        informed_action_prioritization=False,
        num_solutions=1,
        max_frontier_size=1000,
        max_compile_time=1800.0,
        use_dependencies=True,
        heuristic_mode="quality",
    )


def test_gate_timing_supports_alternate_single_qubit_implementations() -> None:
    """Describe virtual and physical rotations independently of their axes."""
    timing = GateTiming(
        rx=0,
        ry=0,
        rz=1,
        virtual_single_qubit_gates=frozenset({"RX", "ry"}),
    )

    assert timing.virtual_single_qubit_gates == frozenset({"rx", "ry"})
    assert timing.is_virtual("rx")
    assert not timing.is_virtual("rz")
    assert timing.duration_for("RZ") == 1


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: GateTiming(rx=-1), "rx duration"),
        (lambda: GateTiming(rxx=0), "rxx duration"),
        (lambda: GateTiming(rz=1), "virtual gate 'rz'"),
        (
            lambda: GateTiming(virtual_single_qubit_gates=frozenset({"x"})),
            "unknown virtual",
        ),
    ],
)
def test_gate_timing_rejects_inconsistent_hardware_settings(
    factory: Callable[[], GateTiming],
    message: str,
) -> None:
    """Reject impossible durations and unknown native gates."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_transport_timing_requires_positive_integer_durations(value: object) -> None:
    """Reject transport durations that cannot occupy a positive timestep count."""
    with pytest.raises(ValueError, match="integer >= 1"):
        TransportTiming(swap=cast("int", value))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SearchConfig(horizon=0), "horizon"),
        (
            lambda: SearchConfig(horizon=2, committed_gates=3),
            "committed_gates must be <= horizon",
        ),
        (lambda: SearchConfig(committed_gates=0), "committed_gates"),
        (lambda: SearchConfig(num_solutions=0), "num_solutions"),
        (lambda: SearchConfig(max_frontier_size=0), "max_frontier_size"),
        (lambda: SearchConfig(max_compile_time=-0.1), "max_compile_time"),
    ],
)
def test_search_config_rejects_invalid_bounds(
    factory: Callable[[], SearchConfig],
    message: str,
) -> None:
    """Reject search settings that cannot define a bounded compilation."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_search_config_accepts_zero_heuristic() -> None:
    """Allow an admissible zero estimate for exact search profiles."""
    assert SearchConfig(heuristic_mode="zero").heuristic_mode == "zero"


def test_search_config_rejects_unknown_heuristic() -> None:
    """Reject heuristic selectors the compiler does not understand."""
    with pytest.raises(ValueError, match="'quality' or 'zero'"):
        SearchConfig(heuristic_mode=cast("HeuristicMode", "distance"))


def test_search_config_requires_a_string_heuristic_selector() -> None:
    """Report a clear type error for malformed heuristic selectors."""
    with pytest.raises(TypeError, match="must be a string"):
        SearchConfig(heuristic_mode=cast("HeuristicMode", 0))
