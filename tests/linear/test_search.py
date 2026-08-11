# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for exhaustive and rolling-horizon Linear search."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

import mqt.ionshuttler.linear.search as search_module
from mqt.ionshuttler.linear.actions import AdvanceTime, Rx, Ry, Rzz, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.config import GateTiming, HeuristicMode, LinearCompilerConfig, SearchConfig
from mqt.ionshuttler.linear.expand import replay_path
from mqt.ionshuttler.linear.parser import parse_qasm_file
from mqt.ionshuttler.linear.result import (
    CompilationResult,
    CompilationStatus,
    DDInsertionRecord,
    GlobalDDRecord,
)
from mqt.ionshuttler.linear.state import State, create_initial_state, has_pending_timed_work

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from itertools import count

    from mqt.ionshuttler.linear.actions import Action, GateAction


def exhaustive_config(
    *,
    informed_action_prioritization: bool = False,
    iterative_diving_search: bool = False,
    num_solutions: int = 1,
    max_frontier_size: int | None = None,
    max_compile_time: float | None = None,
    heuristic_mode: HeuristicMode = "quality",
) -> LinearCompilerConfig:
    """Build an exhaustive-search configuration for focused tests."""
    return LinearCompilerConfig(
        search=SearchConfig(
            horizon=None,
            committed_gates=1,
            iterative_diving_search=iterative_diving_search,
            informed_action_prioritization=informed_action_prioritization,
            num_solutions=num_solutions,
            max_frontier_size=max_frontier_size,
            max_compile_time=max_compile_time,
            heuristic_mode=heuristic_mode,
        )
    )


def assert_replays(
    result: CompilationResult,
    initial_state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: Mapping[int, GateAction],
    predecessors: Mapping[int, frozenset[int]] | None = None,
) -> None:
    """Check that a returned path legally reaches its reported state."""
    assert result.final_state is not None
    assert (
        replay_path(
            initial_state,
            architecture,
            result.path,
            gate_order,
            gates,
            predecessors=predecessors,
        )
        == result.final_state
    )


def test_zero_heuristic_compiles_with_exact_search_profile() -> None:
    """Compile a small circuit with every quality-oriented shortcut disabled."""
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gate = Rx(ion=0, theta=0.5)

    result = search_module.search(
        initial_state,
        [0],
        {0: gate},
        architecture,
        config=exhaustive_config(heuristic_mode="zero"),
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.score == 1
    assert result.path == [gate, AdvanceTime()]


def test_exhaustive_search_schedules_and_completes_a_gate() -> None:
    """Start an available gate and wait until it finishes."""
    architecture = Architecture(num_sites=2)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0)}

    result = search_module.search(
        initial_state,
        [0],
        gates,
        architecture,
        config=exhaustive_config(),
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == [gates[0], AdvanceTime()]
    assert result.num_timesteps == 1
    assert result.score == 1
    assert_replays(result, initial_state, architecture, [0], gates)
    assert initial_state.completed_gates == frozenset()


def test_exhaustive_search_preserves_additional_result_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep result extensions when public search metadata is attached."""
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    dd_record = DDInsertionRecord(
        ion=0,
        window=(0, 1),
        scheme_name="echo",
        gate_timesteps=(0,),
    )
    global_record = GlobalDDRecord(
        scheme_name="periodic",
        pulse_timesteps=(0,),
        spacing=1,
    )
    internal_result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[],
        num_timesteps=0,
        dd_insertions=(dd_record,),
        global_dd_records=(global_record,),
    )
    monkeypatch.setattr(search_module, "_search_with_budget", lambda *_args: internal_result)

    result = search_module.exhaustive_search(
        initial_state,
        architecture,
        [],
        {},
        config=exhaustive_config(),
    )

    assert result.dd_insertions == (dd_record,)
    assert result.global_dd_records == (global_record,)
    assert result.architecture is architecture
    assert result.initial_state is initial_state


def test_two_qubit_gate_routes_ions_into_one_processing_zone() -> None:
    """Move separated ions into a shared zone before running their gate."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [2, 3]})
    initial_state = State(
        positions=((0, 0), (1, 4)),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=((0, 0), (1, 0)),
        pzs_busy_until=(("pz", 0),),
        time=0,
    )
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0)}

    result = search_module.search(
        initial_state,
        [0],
        gates,
        architecture,
        config=exhaustive_config(),
    )

    assert result.status is CompilationStatus.SUCCESS
    assert any(isinstance(action, Shuttle) for action in result.path)
    assert_replays(result, initial_state, architecture, [0], gates)


def test_independent_gates_share_a_timestep() -> None:
    """Run gates concurrently when they use separate ions and zones."""
    architecture = Architecture(
        num_sites=2,
        processing_zones={"left": [0], "right": [1]},
    )
    initial_state = create_initial_state(2, architecture)
    gates = {0: Rx(ion=0, theta=1.0), 1: Ry(ion=1, theta=0.5)}
    predecessors = {0: frozenset(), 1: frozenset()}

    result = search_module.search(
        initial_state,
        [0, 1],
        gates,
        architecture,
        predecessors,
        exhaustive_config(),
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path[:2] == [gates[0], gates[1]]
    assert result.num_timesteps == 1
    assert_replays(result, initial_state, architecture, [0, 1], gates, predecessors)


def test_dependencies_wait_for_gate_completion() -> None:
    """Start a dependent gate only after its predecessor has finished."""
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0, duration=2), 1: Ry(ion=0, theta=0.5)}
    predecessors = {0: frozenset(), 1: frozenset({0})}

    result = search_module.search(
        initial_state,
        [0, 1],
        gates,
        architecture,
        predecessors,
        exhaustive_config(),
    )

    assert result.path.index(gates[1]) > result.path.index(gates[0])
    assert result.path[:3] == [gates[0], AdvanceTime(), AdvanceTime()]
    assert_replays(result, initial_state, architecture, [0, 1], gates, predecessors)


@pytest.mark.parametrize(("horizon", "committed"), [(1, 1), (2, 1), (2, 2)])
def test_rolling_horizon_completes_serial_gates(horizon: int, committed: int) -> None:
    """Combine planning windows into one valid complete schedule."""
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0), 1: Ry(ion=0, theta=0.5)}
    predecessors = {0: frozenset(), 1: frozenset({0})}
    config = LinearCompilerConfig(
        search=SearchConfig(
            horizon=horizon,
            committed_gates=committed,
            iterative_diving_search=False,
            max_frontier_size=None,
            max_compile_time=None,
        )
    )

    result = search_module.search(
        initial_state,
        [0, 1],
        gates,
        architecture,
        predecessors,
        config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.final_state is not None
    assert result.final_state.completed_gates == frozenset({0, 1})
    assert_replays(result, initial_state, architecture, [0, 1], gates, predecessors)


def test_informed_prioritization_falls_back_to_broader_actions() -> None:
    """Recover when the most directed moves alone cannot finish routing."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [2, 3]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 4])
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0)}
    config = exhaustive_config(informed_action_prioritization=True)

    result = search_module.search(
        initial_state,
        [0],
        gates,
        architecture,
        config=config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert_replays(result, initial_state, architecture, [0], gates)


def test_iterative_diving_and_bounded_frontier_find_a_valid_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep deferred alternatives within the configured memory bound."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [2, 3]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 4])
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0)}
    observed_sizes: list[int] = []
    original_push = search_module._push_frontier

    def recording_push(
        frontier: search_module.Frontier,
        node: search_module._SearchNode,
        tie_breaker: count,
        max_size: int | None,
    ) -> None:
        original_push(frontier, node, tie_breaker, max_size)
        observed_sizes.append(len(frontier))

    monkeypatch.setattr(search_module, "_push_frontier", recording_push)
    config = exhaustive_config(
        iterative_diving_search=True,
        max_frontier_size=2,
    )

    result = search_module.search(
        initial_state,
        [0],
        gates,
        architecture,
        config=config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert observed_sizes
    assert max(observed_sizes) <= 2
    assert_replays(result, initial_state, architecture, [0], gates)


@pytest.mark.parametrize("search_style", ["astar", "iterative_diving"])
def test_multiple_solution_search_returns_the_lowest_cost_goal(
    search_style: str,
) -> None:
    """Continue after one goal and retain the best schedule found."""
    iterative_diving = search_style == "iterative_diving"
    architecture = Architecture(num_sites=3)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0)}
    one = search_module.search(
        initial_state,
        [0],
        gates,
        architecture,
        config=exhaustive_config(
            iterative_diving_search=iterative_diving,
            num_solutions=1,
        ),
    )
    multiple = search_module.search(
        initial_state,
        [0],
        gates,
        architecture,
        config=exhaustive_config(
            iterative_diving_search=iterative_diving,
            num_solutions=2,
        ),
    )

    assert one.status is CompilationStatus.SUCCESS
    assert multiple.status is CompilationStatus.SUCCESS
    assert multiple.score == one.score
    assert (multiple.explored_nodes or 0) > (one.explored_nodes or 0)


def test_timeout_keeps_a_complete_goal_found_while_seeking_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the best complete schedule when a wider search runs out of time."""
    goal_found = False
    original_better_solution = search_module._better_solution

    def record_goal(
        current: search_module._FoundSolution | None,
        candidate: search_module._FoundSolution,
    ) -> search_module._FoundSolution:
        nonlocal goal_found
        goal_found = True
        return original_better_solution(current, candidate)

    monkeypatch.setattr(search_module, "_better_solution", record_goal)
    monkeypatch.setattr(
        search_module._TimeBudget,
        "expired",
        lambda _budget: goal_found,
    )
    architecture = Architecture(num_sites=3)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0)}

    result = search_module.search(
        initial_state,
        [0],
        gates,
        architecture,
        config=exhaustive_config(num_solutions=2, max_compile_time=1.0),
    )

    assert result.status is CompilationStatus.TIMEOUT
    assert result.path == [gates[0], AdvanceTime()]
    assert result.final_state is not None
    assert result.final_state.completed_gates == frozenset({0})


def test_timeout_returns_the_best_partial_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return useful progress when the time budget expires between expansions."""
    checks = iter([False, True])
    monkeypatch.setattr(
        search_module._TimeBudget,
        "expired",
        lambda _budget: next(checks, True),
    )
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0)}

    result = search_module.search(
        initial_state,
        [0],
        gates,
        architecture,
        config=exhaustive_config(max_compile_time=1.0),
    )

    assert result.status is CompilationStatus.TIMEOUT
    assert result.path == [gates[0]]
    assert result.final_state is not None
    assert result.final_state.in_progress_gates == ((0, 1),)


def test_interruption_returns_the_best_state_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn a keyboard interruption into a structured partial result."""

    def interrupt(*_args: object, **_kwargs: object) -> list[object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(search_module, "expand", interrupt)
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)

    result = search_module.search(
        initial_state,
        [0],
        {0: Rx(ion=0, theta=1.0)},
        architecture,
        config=exhaustive_config(),
    )

    assert result.status is CompilationStatus.INTERRUPTED
    assert result.path == []
    assert result.final_state == initial_state


def test_impossible_gate_returns_failed_without_idle_search() -> None:
    """Stop when no action can make progress toward the requested gate."""
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)

    result = search_module.search(
        initial_state,
        [0],
        {0: Rx(ion=1, theta=1.0)},
        architecture,
        config=exhaustive_config(),
    )

    assert result.status is CompilationStatus.FAILED
    assert result.path == []
    assert result.final_state == initial_state


def test_rolling_windows_share_one_time_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use one deadline across every planning window."""
    budget_ids: list[int] = []
    original_search = search_module._search_with_budget

    def recording_search(
        initial_state: State,
        context: search_module._SearchContext,
        budget: search_module._TimeBudget,
    ) -> CompilationResult:
        budget_ids.append(id(budget))
        return original_search(initial_state, context, budget)

    monkeypatch.setattr(search_module, "_search_with_budget", recording_search)
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0), 1: Ry(ion=0, theta=0.5)}
    predecessors = {0: frozenset(), 1: frozenset({0})}
    config = LinearCompilerConfig(
        search=SearchConfig(
            horizon=1,
            committed_gates=1,
            iterative_diving_search=False,
            max_frontier_size=None,
            max_compile_time=None,
        )
    )

    result = search_module.search(
        initial_state,
        [0, 1],
        gates,
        architecture,
        predecessors,
        config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert len(budget_ids) == 2
    assert len(set(budget_ids)) == 1


def test_rolling_search_times_out_before_starting_a_window() -> None:
    """Return immediately when no time remains for the first planning window."""
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    config = LinearCompilerConfig(search=SearchConfig(horizon=1, committed_gates=1, max_compile_time=0.0))

    result = search_module.search(
        initial_state,
        [0],
        {0: Rx(ion=0, theta=1.0)},
        architecture,
        config=config,
    )

    assert result.status is CompilationStatus.TIMEOUT
    assert result.path == []
    assert result.final_state == initial_state


def test_rolling_search_reports_an_interruption(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the completed window prefix when rolling planning is interrupted."""

    def interrupt(*_args: object, **_kwargs: object) -> CompilationStatus:
        raise KeyboardInterrupt

    monkeypatch.setattr(search_module, "_run_rolling_search", interrupt)
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)

    result = search_module.search(
        initial_state,
        [0],
        {0: Rx(ion=0, theta=1.0)},
        architecture,
    )

    assert result.status is CompilationStatus.INTERRUPTED
    assert result.path == []
    assert result.final_state == initial_state


def test_rolling_search_without_dependency_maps_uses_circuit_order() -> None:
    """Schedule gates in circuit order when no dependency maps are supplied."""
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0), 1: Ry(ion=0, theta=0.5)}
    config = LinearCompilerConfig(
        search=SearchConfig(
            horizon=1,
            committed_gates=1,
            iterative_diving_search=False,
            max_frontier_size=None,
            max_compile_time=None,
        )
    )

    result = search_module.search(
        initial_state,
        [0, 1],
        gates,
        architecture,
        config=config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == [gates[0], AdvanceTime(), gates[1], AdvanceTime()]
    assert_replays(result, initial_state, architecture, [0, 1], gates)


def test_rolling_horizon_entry_point_requires_a_finite_horizon() -> None:
    """Reject the rolling entry point when complete-circuit search was selected."""
    architecture = Architecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)

    with pytest.raises(ValueError, match="finite horizon"):
        search_module.rolling_horizon_search(
            initial_state,
            architecture,
            [0],
            {0: Rx(ion=0, theta=1.0)},
            predecessors=None,
            config=exhaustive_config(),
        )


def test_rolling_window_accepts_a_completed_global_predecessor() -> None:
    """Treat dependencies completed before the current window as satisfied."""
    architecture = Architecture(num_sites=1)
    initial_state = State(
        positions=((0, 0),),
        completed_gates=frozenset({0}),
        in_progress_gates=(),
        ions_busy_until=((0, 0),),
        pzs_busy_until=(("all_sites", 0),),
        time=0,
    )
    gates = {0: Rx(ion=0, theta=1.0), 1: Ry(ion=0, theta=0.5)}
    predecessors = {0: frozenset(), 1: frozenset({0})}
    config = LinearCompilerConfig(
        search=SearchConfig(
            horizon=1,
            committed_gates=1,
            iterative_diving_search=False,
            max_frontier_size=None,
            max_compile_time=None,
        )
    )

    result = search_module.search(
        initial_state,
        [0, 1],
        gates,
        architecture,
        predecessors,
        config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == [gates[1], AdvanceTime()]
    assert_replays(result, initial_state, architecture, [0, 1], gates, predecessors)


def test_production_defaults_match_the_frozen_schedule(
    production_default_golden: dict[str, object],
) -> None:
    """Keep the compact production schedule deterministic."""
    architecture = Architecture(
        num_sites=9,
        processing_zones={"pz1": [2, 3], "pz2": [5, 6]},
    )
    initial_state = create_initial_state(2, architecture)
    gates = {
        0: Rx(ion=0, theta=0.1),
        1: Ry(ion=1, theta=0.2),
        2: Rzz(ion_a=0, ion_b=1, theta=0.3, duration=2),
    }
    predecessors = {0: frozenset(), 1: frozenset(), 2: frozenset({0, 1})}

    result = search_module.search(
        initial_state,
        [0, 1, 2],
        gates,
        architecture,
        predecessors,
    )

    expected_result = cast("dict[str, object]", production_default_golden["expected_result"])
    expected_actions = cast("list[dict[str, object]]", expected_result["actions"])
    normalized_actions = []
    for action in expected_actions:
        normalized = {key: value for key, value in action.items() if key != "start_time"}
        if normalized["type"] == "AdvanceTime":
            normalized.pop("duration")
        normalized_actions.append(normalized)

    assert result.status is CompilationStatus.SUCCESS
    assert result.num_timesteps == expected_result["num_timesteps"]
    assert result.score == expected_result["score"]
    assert [action.to_dict() for action in result.path] == normalized_actions
    assert_replays(result, initial_state, architecture, [0, 1, 2], gates, predecessors)


def test_larger_schedule_remains_deterministic_and_replayable() -> None:
    """Cover concurrent work, routing, and successive time advances."""
    architecture = Architecture(
        num_sites=12,
        processing_zones={"left": [2, 3, 4], "right": [7, 8, 9]},
    )
    initial_state = create_initial_state(4, architecture)
    gates = {
        0: Rx(ion=0, theta=0.1, duration=2),
        1: Ry(ion=3, theta=0.2, duration=2),
        2: Rzz(ion_a=0, ion_b=1, theta=0.3, duration=2),
        3: Rzz(ion_a=2, ion_b=3, theta=0.4, duration=2),
    }
    predecessors = {
        0: frozenset(),
        1: frozenset(),
        2: frozenset({0}),
        3: frozenset({1}),
    }
    expected: list[Action] = [
        Shuttle(ion=0, src=4, dst=3),
        Shuttle(ion=1, src=5, dst=4),
        gates[1],
        AdvanceTime(),
        gates[0],
        AdvanceTime(),
        AdvanceTime(),
        Shuttle(ion=3, src=7, dst=8),
        Shuttle(ion=2, src=6, dst=7),
        gates[2],
        AdvanceTime(),
        gates[3],
        AdvanceTime(),
        AdvanceTime(),
    ]

    result = search_module.search(
        initial_state,
        [0, 1, 2, 3],
        gates,
        architecture,
        predecessors,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == expected
    assert result.num_timesteps == 6
    assert_replays(result, initial_state, architecture, [0, 1, 2, 3], gates, predecessors)


def test_six_qubit_qft_matches_frozen_schedule() -> None:
    """Keep a substantial production schedule exactly reproducible."""
    qasm_path = Path(__file__).parent / "fixtures" / "qft_6.qasm"
    num_qubits, gate_list, predecessors, _ = parse_qasm_file(
        qasm_path,
        gate_timing=GateTiming(),
    )
    architecture = Architecture(
        num_sites=9,
        processing_zones={"pz1": [2, 3], "pz2": [5, 6]},
    )
    initial_state = create_initial_state(num_qubits, architecture)
    gates = dict(enumerate(gate_list))

    result = search_module.search(
        initial_state,
        list(gates),
        gates,
        architecture,
        predecessors,
    )

    serialized_actions = [action.to_dict() for action in result.path]
    encoded_actions = json.dumps(
        serialized_actions,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert result.status is CompilationStatus.SUCCESS
    assert len(gates) == 143
    assert len(result.path) == 426
    assert result.num_timesteps == 219
    assert Counter(type(action).__name__ for action in result.path) == {
        "AdvanceTime": 219,
        "PhysicalSwap": 36,
        "Rx": 5,
        "Ry": 54,
        "Rz": 45,
        "Rzz": 39,
        "Shuttle": 28,
    }
    assert hashlib.sha256(encoded_actions, usedforsecurity=False).hexdigest() == (
        "1557943f721c4019943a4a768cf1598e0f58f986d910b59a068a79c5e03d93b8"
    )
    assert result.final_state is not None
    assert result.final_state.completed_gates == frozenset(gates)
    replayed_state = initial_state
    for action in result.path:
        if isinstance(action, AdvanceTime):
            assert has_pending_timed_work(replayed_state)
        replayed_state = replay_path(
            replayed_state,
            architecture,
            [action],
            list(gates),
            gates,
            predecessors=predecessors,
        )
    assert replayed_state == result.final_state
    assert initial_state.completed_gates == frozenset()
