# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear hardware timing and compiler settings."""

from __future__ import annotations

from inspect import isabstract
from typing import TYPE_CHECKING, cast

import pytest

from mqt.ionshuttler.core.gates import Rx
from mqt.ionshuttler.linear import GateTiming, TransportTiming
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.config import LinearCompilerConfig, PermutationConfig, SearchConfig
from mqt.ionshuttler.linear.cost import zero_heuristic

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.linear.cost import HeuristicFn


def test_architecture_and_compiler_config_use_ready_to_run_defaults() -> None:
    """Keep hardware data on the architecture and policy in compiler config."""
    search = SearchConfig()
    permutation = PermutationConfig()
    architecture = LinearArchitecture(num_sites=1)

    assert architecture.gate_timing == GateTiming(rx=1, ry=1, rz=0, rxx=2, ryy=2, rzz=2)
    assert architecture.gate_timing.virtual_single_qubit_gates == frozenset({"rz"})
    assert architecture.transport_timing == TransportTiming(shuttle=1, swap=3)
    assert search == SearchConfig(
        horizon=3,
        committed_gates=2,
        iterative_diving_search=True,
        informed_action_prioritization=False,
        num_solutions=1,
        max_frontier_size=1000,
        use_dependencies=True,
        max_compile_time=1800.0,
        pre_partition_config=None,
        pre_partition=False,
        action_types=None,
    )
    assert permutation == PermutationConfig(
        lookahead_gates=32,
        record_planning_trace=False,
        max_compile_time=1800.0,
        pre_partition=True,
        pre_partition_config=None,
        action_types=None,
    )


def test_strategy_configs_share_linear_settings() -> None:
    """Keep common settings on a base that cannot select a strategy itself."""
    assert isabstract(LinearCompilerConfig)
    assert issubclass(SearchConfig, LinearCompilerConfig)
    assert issubclass(PermutationConfig, LinearCompilerConfig)


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


def test_transport_timing_round_trips_and_rejects_malformed_data() -> None:
    """Restore transport timing and use defaults for missing fields."""
    timing = TransportTiming(shuttle=2, swap=5)

    assert TransportTiming.from_dict(timing.to_dict()) == timing
    assert TransportTiming.from_dict({"swap": 4}) == TransportTiming(shuttle=1, swap=4)
    with pytest.raises(TypeError, match="shuttle duration must be an integer"):
        TransportTiming.from_dict({"shuttle": "1"})
    with pytest.raises(TypeError, match="transport_timing must be a JSON object"):
        TransportTiming.from_dict(None)
    with pytest.raises(ValueError, match="unknown transport_timing fields: shutle"):
        TransportTiming.from_dict({"shutle": 2})


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
    ],
)
def test_search_config_rejects_invalid_bounds(
    factory: Callable[[], SearchConfig],
    message: str,
) -> None:
    """Reject search settings that cannot define a bounded compilation."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        pytest.param(lambda: SearchConfig(max_compile_time=-0.1), ValueError, "max_compile_time", id="negative-time"),
        pytest.param(
            lambda: PermutationConfig(max_compile_time=cast("float", "60")),
            ValueError,
            "max_compile_time",
            id="text-time",
        ),
        pytest.param(
            lambda: SearchConfig(pre_partition=cast("bool", 1)),
            TypeError,
            "pre_partition",
            id="integer-partition-flag",
        ),
        pytest.param(lambda: PermutationConfig(lookahead_gates=0), ValueError, "lookahead_gates", id="no-lookahead"),
        pytest.param(
            lambda: PermutationConfig(record_planning_trace=cast("bool", 1)),
            TypeError,
            "record_planning_trace",
            id="integer-trace-flag",
        ),
    ],
)
def test_compiler_and_permutation_configs_reject_invalid_settings(
    factory: Callable[[], object],
    error: type[Exception],
    message: str,
) -> None:
    """Reject settings that cannot describe a compilation."""
    with pytest.raises(error, match=message):
        factory()


def test_permutation_config_requires_pre_partitioning() -> None:
    """Do not allow the permutation strategy to skip its required assignment."""
    pre_partition = False

    with pytest.raises(ValueError, match="requires pre_partition=True"):
        PermutationConfig(pre_partition=cast("Literal[True]", pre_partition))


def test_compiler_configs_reject_invalid_action_types() -> None:
    """Reject entries that are not actions and repeated action classes."""
    with pytest.raises(TypeError, match="Action subclasses"):
        SearchConfig(action_types=(cast("type[Action]", object),))
    with pytest.raises(ValueError, match="must not contain duplicates"):
        PermutationConfig(action_types=(Rx, Rx))


def test_search_config_accepts_the_zero_heuristic() -> None:
    """Allow the admissible zero estimate for exact search profiles."""
    assert SearchConfig(heuristic=zero_heuristic).heuristic is zero_heuristic


def test_search_config_defaults_to_the_built_in_heuristic() -> None:
    """Leave the built-in estimate in place unless one is supplied."""
    assert SearchConfig().heuristic is None


def test_search_config_accepts_a_callable_heuristic() -> None:
    """Store a supplied heuristic for the search to consult."""

    def estimate(*_args: object, **_kwargs: object) -> int:
        return 0

    assert SearchConfig(heuristic=estimate).heuristic is estimate


def test_search_config_rejects_a_non_callable_heuristic() -> None:
    """Report a clear type error when the heuristic cannot be called."""
    with pytest.raises(TypeError, match="heuristic must be callable or None"):
        SearchConfig(heuristic=cast("HeuristicFn", 5))
