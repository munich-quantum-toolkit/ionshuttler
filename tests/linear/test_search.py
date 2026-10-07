# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import mqt.ionshuttler.linear.search as search_module
from mqt.ionshuttler.circuit import Circuit, parse_circuit
from mqt.ionshuttler.linear import GateTiming
from mqt.ionshuttler.linear.actions import Rx, Ry, Rzz, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.config import LinearCompilerConfig, SearchConfig
from mqt.ionshuttler.linear.cost import zero_heuristic
from mqt.ionshuttler.linear.expand import GenerationMode
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus, LinearDiagnostics
from mqt.ionshuttler.linear.schedule import (
    LinearMachineState,
)
from mqt.ionshuttler.linear.state import State, create_initial_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.linear.actions import GateAction
    from mqt.ionshuttler.linear.cost import HeuristicFn
    from mqt.ionshuttler.linear.state import SearchTransition


def exhaustive_config(
    *,
    informed_action_prioritization: bool = False,
    iterative_diving_search: bool = False,
    num_solutions: int = 1,
    max_frontier_size: int | None = None,
    max_compile_time: float | None = None,
    heuristic: HeuristicFn | None = None,
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
            heuristic=heuristic,
        )
    )


def diagnostics(result: CompilationResult) -> LinearDiagnostics:
    """Return the Linear search statistics of a result."""
    assert isinstance(result.diagnostics, LinearDiagnostics)
    return result.diagnostics


def make_circuit(
    gates: Mapping[int, GateAction],
    predecessors: Mapping[int, frozenset[int]] | None = None,
) -> Circuit:
    """Build a circuit from concise test mappings."""
    assert tuple(gates) == tuple(range(len(gates)))
    gate_ids = range(len(gates))
    dependencies = (
        tuple(frozenset((gate_id - 1,)) if gate_id else frozenset() for gate_id in gate_ids)
        if predecessors is None
        else tuple(predecessors.get(gate_id, frozenset()) for gate_id in gate_ids)
    )
    return Circuit(
        num_ions=8,
        gates=tuple(gates[gate_id] for gate_id in gate_ids),
        predecessors=dependencies,
    )


def test_zero_heuristic_compiles_with_exact_search_profile() -> None:
    """Compile a small circuit with every quality-oriented shortcut disabled."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gate = Rx(ion=0, theta=0.5, gate_id=0)

    result = search_module.search(
        initial_state,
        make_circuit({0: gate}),
        architecture,
        config=exhaustive_config(heuristic=zero_heuristic),
    )

    assert result.status is CompilationStatus.SUCCESS
    assert diagnostics(result).score == 1
    assert result.path == [gate]
    assert result.schedule.scheduled_actions[0].start_time == 0


def test_exhaustive_search_schedules_and_completes_a_gate() -> None:
    """Start an available gate and wait until it finishes."""
    architecture = LinearArchitecture(num_sites=2)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0)}

    result = search_module.search(
        initial_state,
        make_circuit(gates),
        architecture,
        config=exhaustive_config(),
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == [gates[0]]
    assert result.end_time == 1
    assert diagnostics(result).score == 1
    result.validate()
    assert initial_state.completed_gates == frozenset()


def test_exhaustive_search_serializes_actions_from_the_entry_time() -> None:
    """Derive schedule timestamps from the search entry rather than its final state."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = State(
        positions=((0, 0),),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=((0, 5),),
        pzs_busy_until=(("all_sites", 5),),
        time=5,
    )

    result = search_module.exhaustive_search(
        initial_state,
        architecture,
        make_circuit({0: Rx(ion=0, theta=0.5, gate_id=0)}),
        config=exhaustive_config(),
    )

    serialized_actions = result.schedule.to_dict()["actions"]
    assert isinstance(serialized_actions, list)
    assert isinstance(serialized_actions[0], dict)
    assert serialized_actions[0]["start_time"] == 5


def test_exhaustive_search_attaches_the_public_scheduled_program(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attach initial hardware metadata at the public search boundary."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    internal_outcome = search_module._SearchOutcome(
        path=(),
        final_state=initial_state,
        status=CompilationStatus.SUCCESS,
        explored_nodes=1,
    )
    monkeypatch.setattr(search_module, "_search_with_budget", lambda *_args: internal_outcome)

    result = search_module.exhaustive_search(
        initial_state,
        architecture,
        make_circuit({}),
        config=exhaustive_config(),
    )

    assert result.architecture is architecture
    assert result.initial_state.positions == initial_state.positions


def test_two_qubit_gate_routes_ions_into_one_processing_zone() -> None:
    """Move separated ions into a shared zone before running their gate."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2, 3]})
    initial_state = State(
        positions=((0, 0), (1, 4)),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=((0, 0), (1, 0)),
        pzs_busy_until=(("pz", 0),),
        time=0,
    )
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0, gate_id=0)}

    result = search_module.search(
        initial_state,
        make_circuit(gates),
        architecture,
        config=exhaustive_config(),
    )

    assert result.status is CompilationStatus.SUCCESS
    assert any(isinstance(action, Shuttle) for action in result.path)
    result.validate()


def test_independent_gates_share_a_timestep() -> None:
    """Run gates concurrently when they use separate ions and zones."""
    architecture = LinearArchitecture(
        num_sites=2,
        processing_zones={"left": [0], "right": [1]},
    )
    initial_state = create_initial_state(2, architecture)
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0), 1: Ry(ion=1, theta=0.5, gate_id=1)}
    predecessors = {0: frozenset(), 1: frozenset()}

    result = search_module.search(
        initial_state,
        make_circuit(gates, predecessors),
        architecture,
        exhaustive_config(),
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path[:2] == [gates[0], gates[1]]
    assert result.end_time == 1
    result.validate()


def test_dependencies_wait_for_gate_completion() -> None:
    """Start a dependent gate only after its predecessor has finished."""
    architecture = LinearArchitecture(num_sites=1, gate_timing=GateTiming(rx=2))
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0), 1: Ry(ion=0, theta=0.5, gate_id=1)}
    predecessors = {0: frozenset(), 1: frozenset({0})}

    result = search_module.search(
        initial_state,
        make_circuit(gates, predecessors),
        architecture,
        exhaustive_config(),
    )

    assert result.path.index(gates[1]) > result.path.index(gates[0])
    assert result.path == [gates[0], gates[1]]
    assert [item.start_time for item in result.schedule.scheduled_actions] == [0, 2]
    result.validate()


@pytest.mark.parametrize(("horizon", "committed"), [(1, 1), (2, 1), (2, 2)])
def test_rolling_horizon_completes_serial_gates(horizon: int, committed: int) -> None:
    """Combine planning windows into one valid complete schedule."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0), 1: Ry(ion=0, theta=0.5, gate_id=1)}
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
        make_circuit(gates, predecessors),
        architecture,
        config,
    )

    assert result.status is CompilationStatus.SUCCESS
    result.validate()


def test_informed_prioritization_falls_back_to_broader_actions() -> None:
    """Recover when the most directed moves alone cannot finish routing."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2, 3]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 4])
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0, gate_id=0)}
    config = exhaustive_config(informed_action_prioritization=True)

    result = search_module.search(
        initial_state,
        make_circuit(gates),
        architecture,
        config=config,
    )

    assert result.status is CompilationStatus.SUCCESS
    result.validate()


def test_iterative_diving_and_bounded_frontier_find_a_valid_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep deferred alternatives within the configured memory bound."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2, 3]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 4])
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0, gate_id=0)}
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
        make_circuit(gates),
        architecture,
        config=config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert observed_sizes
    assert max(observed_sizes) <= 2
    result.validate()


@pytest.mark.parametrize("search_style", ["astar", "iterative_diving"])
def test_multiple_solution_search_returns_the_lowest_cost_goal(
    search_style: str,
) -> None:
    """Continue after one goal and retain the best schedule found."""
    iterative_diving = search_style == "iterative_diving"
    architecture = LinearArchitecture(num_sites=3)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0)}
    one = search_module.search(
        initial_state,
        make_circuit(gates),
        architecture,
        config=exhaustive_config(
            iterative_diving_search=iterative_diving,
            num_solutions=1,
        ),
    )
    multiple = search_module.search(
        initial_state,
        make_circuit(gates),
        architecture,
        config=exhaustive_config(
            iterative_diving_search=iterative_diving,
            num_solutions=2,
        ),
    )

    assert one.status is CompilationStatus.SUCCESS
    assert multiple.status is CompilationStatus.SUCCESS
    assert diagnostics(multiple).score == diagnostics(one).score
    assert diagnostics(multiple).explored_nodes > diagnostics(one).explored_nodes


def test_timeout_keeps_a_complete_goal_found_while_seeking_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the best complete schedule when a wider search runs out of time."""
    goal_found = False
    original_better_solution = search_module._better_solution

    def record_goal(
        current: search_module._SearchOutcome | None,
        candidate: search_module._SearchOutcome,
    ) -> search_module._SearchOutcome:
        nonlocal goal_found
        goal_found = True
        return original_better_solution(current, candidate)

    monkeypatch.setattr(search_module, "_better_solution", record_goal)
    monkeypatch.setattr(
        search_module._TimeBudget,
        "expired",
        lambda _budget: goal_found,
    )
    architecture = LinearArchitecture(num_sites=3)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0)}

    result = search_module.search(
        initial_state,
        make_circuit(gates),
        architecture,
        config=exhaustive_config(num_solutions=2, max_compile_time=1.0),
    )

    assert result.status is CompilationStatus.TIMEOUT
    assert result.path == [gates[0]]
    result.validate()


def test_timeout_returns_the_best_partial_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return useful progress when the time budget expires between expansions."""
    checks = iter([False, True])
    monkeypatch.setattr(
        search_module._TimeBudget,
        "expired",
        lambda _budget: next(checks, True),
    )
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0)}

    result = search_module.search(
        initial_state,
        make_circuit(gates),
        architecture,
        config=exhaustive_config(max_compile_time=1.0),
    )

    assert result.status is CompilationStatus.TIMEOUT
    assert result.path == [gates[0]]
    assert result.final_state.time == 1
    result.validate()


def test_interruption_returns_the_best_state_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn a keyboard interruption into a structured partial result."""

    def interrupt(*_args: object, **_kwargs: object) -> list[object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(search_module, "expand", interrupt)
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)

    result = search_module.search(
        initial_state,
        make_circuit({0: Rx(ion=0, theta=1.0, gate_id=0)}),
        architecture,
        config=exhaustive_config(),
    )

    assert result.status is CompilationStatus.INTERRUPTED
    assert result.path == []
    assert result.final_state == LinearMachineState.from_compiler_state(initial_state)


def test_impossible_gate_returns_failed_without_idle_search() -> None:
    """Stop when no action can make progress toward the requested gate."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)

    result = search_module.search(
        initial_state,
        make_circuit({0: Rx(ion=1, theta=1.0, gate_id=0)}),
        architecture,
        config=exhaustive_config(),
    )

    assert result.status is CompilationStatus.FAILED
    assert result.path == []
    assert result.final_state == LinearMachineState.from_compiler_state(initial_state)


def test_rolling_windows_share_one_time_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use one deadline across every planning window."""
    budget_ids: list[int] = []
    original_search = search_module._search_with_budget

    def recording_search(
        initial_state: State,
        context: search_module._SearchContext,
        budget: search_module._TimeBudget,
    ) -> search_module._SearchOutcome:
        budget_ids.append(id(budget))
        return original_search(initial_state, context, budget)

    monkeypatch.setattr(search_module, "_search_with_budget", recording_search)
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0), 1: Ry(ion=0, theta=0.5, gate_id=1)}
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
        make_circuit(gates, predecessors),
        architecture,
        config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert len(budget_ids) == 2
    assert len(set(budget_ids)) == 1


def test_rolling_search_times_out_before_starting_a_window() -> None:
    """Return immediately when no time remains for the first planning window."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    config = LinearCompilerConfig(search=SearchConfig(horizon=1, committed_gates=1, max_compile_time=0.0))

    result = search_module.search(
        initial_state,
        make_circuit({0: Rx(ion=0, theta=1.0, gate_id=0)}),
        architecture,
        config=config,
    )

    assert result.status is CompilationStatus.TIMEOUT
    assert result.path == []
    assert result.final_state == LinearMachineState.from_compiler_state(initial_state)


def test_rolling_search_reports_an_interruption(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the completed window prefix when rolling planning is interrupted."""

    def interrupt(*_args: object, **_kwargs: object) -> CompilationStatus:
        raise KeyboardInterrupt

    monkeypatch.setattr(search_module, "_run_rolling_search", interrupt)
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)

    result = search_module.search(
        initial_state,
        make_circuit({0: Rx(ion=0, theta=1.0, gate_id=0)}),
        architecture,
    )

    assert result.status is CompilationStatus.INTERRUPTED
    assert result.path == []
    assert result.final_state == LinearMachineState.from_compiler_state(initial_state)


def test_rolling_search_serial_policy_uses_circuit_order() -> None:
    """Schedule gates in circuit order when dependency scheduling is disabled."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0), 1: Ry(ion=0, theta=0.5, gate_id=1)}
    config = LinearCompilerConfig(
        search=SearchConfig(
            horizon=1,
            committed_gates=1,
            iterative_diving_search=False,
            max_frontier_size=None,
            max_compile_time=None,
            use_dependencies=False,
        )
    )

    result = search_module.search(
        initial_state,
        make_circuit(gates, {0: frozenset(), 1: frozenset()}),
        architecture,
        config=config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == [gates[0], gates[1]]
    assert [item.start_time for item in result.schedule.scheduled_actions] == [0, 1]
    result.validate()


def test_rolling_horizon_entry_point_requires_a_finite_horizon() -> None:
    """Reject the rolling entry point when complete-circuit search was selected."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)

    with pytest.raises(ValueError, match="finite horizon"):
        search_module.rolling_horizon_search(
            initial_state,
            architecture,
            make_circuit({0: Rx(ion=0, theta=1.0, gate_id=0)}),
            config=exhaustive_config(),
        )


def test_rolling_window_accepts_a_completed_global_predecessor() -> None:
    """Treat dependencies completed before the current window as satisfied."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = State(
        positions=((0, 0),),
        completed_gates=frozenset({0}),
        in_progress_gates=(),
        ions_busy_until=((0, 0),),
        pzs_busy_until=(("all_sites", 0),),
        time=0,
    )
    gates = {0: Rx(ion=0, theta=1.0, gate_id=0), 1: Ry(ion=0, theta=0.5, gate_id=1)}
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
        make_circuit(gates, predecessors),
        architecture,
        config,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == [gates[1]]
    result.validate()


def test_production_defaults_build_the_expected_compact_schedule() -> None:
    """Keep the compact production schedule deterministic."""
    architecture = LinearArchitecture(
        num_sites=9,
        processing_zones={"pz1": [2, 3], "pz2": [5, 6]},
    )
    initial_state = create_initial_state(2, architecture)
    gates = {
        0: Rx(ion=0, theta=0.1, gate_id=0),
        1: Ry(ion=1, theta=0.2, gate_id=1),
        2: Rzz(ion_a=0, ion_b=1, theta=0.3, gate_id=2),
    }
    predecessors = {0: frozenset(), 1: frozenset(), 2: frozenset({0, 1})}

    result = search_module.search(
        initial_state,
        make_circuit(gates, predecessors),
        architecture,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.end_time == 5
    assert diagnostics(result).score == 5
    assert result.path == [
        Shuttle(ion=0, src=3, dst=2),
        Shuttle(ion=1, src=4, dst=3),
        gates[0],
        gates[1],
        gates[2],
    ]
    assert [item.start_time for item in result.schedule.scheduled_actions] == [0, 0, 1, 2, 3]
    result.validate()


def test_larger_schedule_remains_deterministic_and_replayable() -> None:
    """Cover concurrent work, routing, and successive time advances."""
    architecture = LinearArchitecture(
        num_sites=12,
        processing_zones={"left": [2, 3, 4], "right": [7, 8, 9]},
        gate_timing=GateTiming(rx=2, ry=2),
    )
    initial_state = create_initial_state(4, architecture)
    gates = {
        0: Rx(ion=0, theta=0.1, gate_id=0),
        1: Ry(ion=3, theta=0.2, gate_id=1),
        2: Rzz(ion_a=0, ion_b=1, theta=0.3, gate_id=2),
        3: Rzz(ion_a=2, ion_b=3, theta=0.4, gate_id=3),
    }
    predecessors = {
        0: frozenset(),
        1: frozenset(),
        2: frozenset({0}),
        3: frozenset({1}),
    }
    expected: list[SearchTransition] = [
        Shuttle(ion=0, src=4, dst=3),
        Shuttle(ion=1, src=5, dst=4),
        gates[1],
        gates[0],
        Shuttle(ion=3, src=7, dst=8),
        Shuttle(ion=2, src=6, dst=7),
        gates[2],
        gates[3],
    ]

    result = search_module.search(
        initial_state,
        make_circuit(gates, predecessors),
        architecture,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == expected
    assert result.end_time == 6
    result.validate()


def test_six_qubit_qft_matches_frozen_schedule() -> None:
    """Keep a substantial production schedule exactly reproducible."""
    qasm_path = Path(__file__).parent / "fixtures" / "qft_6.qasm"
    circuit = parse_circuit(qasm_path)
    architecture = LinearArchitecture(
        num_sites=9,
        processing_zones={"pz1": [2, 3], "pz2": [5, 6]},
    )
    initial_state = create_initial_state(circuit.num_ions, architecture)
    result = search_module.search(
        initial_state,
        circuit,
        architecture,
    )

    serialized_actions = [action.to_dict() for action in result.path]
    encoded_actions = json.dumps(
        serialized_actions,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert result.status is CompilationStatus.SUCCESS
    assert len(circuit.gates) == 143
    assert len(result.path) == 196
    assert result.end_time == 219
    assert Counter(type(action).__name__ for action in result.path) == {
        "PhysicalSwap": 36,
        "Rx": 5,
        "Ry": 54,
        "Rz": 45,
        "Rzz": 39,
        "Shuttle": 17,
    }
    assert hashlib.sha256(encoded_actions, usedforsecurity=False).hexdigest() == (
        "1717a9a774d55a141bc5af2a586837bcf4f3419e7f261cd3b4174c4bb6f9a191"
    )
    result.validate()
    assert initial_state.completed_gates == frozenset()


@pytest.mark.parametrize("max_frontier_size", [None, 1, 2, 5, 64])
def test_frontier_returns_nodes_in_best_first_order(max_frontier_size: int | None) -> None:
    """Take frontier nodes cheapest-first, whether or not the frontier is bounded."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2, 3]})
    state = create_initial_state(2, architecture, initial_positions=[0, 4])
    tie_breaker = count()
    frontier: search_module.Frontier = []

    priorities = [7, 2, 9, 2, 0, 5, 3, 8, 1, 6, 4, 9, 1]
    for priority in priorities:
        node = search_module._SearchNode(
            state=state,
            path=(),
            cost_value=priority,
            heuristic_value=0,
            generation_mode=GenerationMode.FULL,
        )
        search_module._push_frontier(frontier, node, tie_breaker, max_frontier_size)
        if max_frontier_size is not None:
            assert len(frontier) <= max_frontier_size

    taken = []
    while frontier:
        node, _ = search_module._take_node(None, frontier, max_frontier_size)
        taken.append(node.cost_value)

    assert taken == sorted(taken)
    if max_frontier_size is None:
        assert sorted(taken) == sorted(priorities)
    else:
        # A bounded frontier keeps the cheapest entries it was offered.
        assert taken == sorted(priorities)[: len(taken)]


def test_custom_heuristic_replaces_the_built_in_estimate() -> None:
    """Route a two-qubit gate using a supplied heuristic instead of the default."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2, 3]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 4])
    gates: dict[int, GateAction] = {0: Rzz(ion_a=0, ion_b=1, theta=1.0, gate_id=0)}
    calls: list[int] = []

    def custom_heuristic(
        state: State,
        architecture_: LinearArchitecture,
        circuit: Circuit,
        active_gate_ids: Sequence[int],
        predecessors: Sequence[frozenset[int]],
        *,
        use_dependencies: bool = True,
        gate_zone: Mapping[int, str] | None = None,
        zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
    ) -> int:
        del architecture_, circuit, predecessors, use_dependencies, gate_zone, zone_site_pairs
        calls.append(state.time)
        return len([gate_id for gate_id in active_gate_ids if gate_id not in state.completed_gates])

    config = LinearCompilerConfig(
        search=SearchConfig(
            horizon=None,
            committed_gates=1,
            iterative_diving_search=False,
            max_frontier_size=None,
            max_compile_time=None,
            heuristic=custom_heuristic,
        )
    )

    result = search_module.search(initial_state, make_circuit(gates), architecture, config=config)

    assert result.status is CompilationStatus.SUCCESS
    assert calls, "the supplied heuristic was never consulted"
    result.validate()


def test_custom_heuristic_is_used_by_rolling_horizon_windows() -> None:
    """Carry the supplied heuristic into every rolling planning window."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates: dict[int, GateAction] = {0: Rx(ion=0, theta=1.0, gate_id=0), 1: Ry(ion=0, theta=0.5, gate_id=1)}
    predecessors = {0: frozenset[int](), 1: frozenset({0})}
    windows: list[tuple[int, ...]] = []

    def custom_heuristic(
        state: State,
        architecture_: LinearArchitecture,
        circuit: Circuit,
        active_gate_ids: Sequence[int],
        predecessors_: Sequence[frozenset[int]],
        *,
        use_dependencies: bool = True,
        gate_zone: Mapping[int, str] | None = None,
        zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
    ) -> int:
        del state, architecture_, circuit, predecessors_, use_dependencies, gate_zone, zone_site_pairs
        windows.append(tuple(active_gate_ids))
        return 0

    config = LinearCompilerConfig(
        search=SearchConfig(
            horizon=1,
            committed_gates=1,
            iterative_diving_search=False,
            max_frontier_size=None,
            max_compile_time=None,
            heuristic=custom_heuristic,
        )
    )

    result = search_module.search(initial_state, make_circuit(gates, predecessors), architecture, config)

    assert result.status is CompilationStatus.SUCCESS
    result.validate()
    assert set(windows) == {(0,), (1,)}


def test_zero_heuristic_yields_an_admissible_estimate() -> None:
    """Skip estimation entirely when the zero heuristic is selected."""
    architecture = LinearArchitecture(num_sites=1)
    initial_state = create_initial_state(1, architecture)
    gates: dict[int, GateAction] = {0: Rx(ion=0, theta=0.5, gate_id=0)}

    config = LinearCompilerConfig(
        search=SearchConfig(
            horizon=None,
            committed_gates=1,
            iterative_diving_search=False,
            max_frontier_size=None,
            max_compile_time=None,
            heuristic=zero_heuristic,
        )
    )

    result = search_module.search(initial_state, make_circuit(gates), architecture, config=config)

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == [gates[0]]


def test_omitting_a_custom_heuristic_keeps_the_built_in_schedule() -> None:
    """Produce the same schedule with an explicit ``None`` as with the default."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2, 3]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 4])
    gates: dict[int, GateAction] = {0: Rzz(ion_a=0, ion_b=1, theta=1.0, gate_id=0)}

    circuit = make_circuit(gates)
    default_result = search_module.search(initial_state, circuit, architecture, config=exhaustive_config())
    explicit_result = search_module.search(
        initial_state,
        circuit,
        architecture,
        config=LinearCompilerConfig(
            search=SearchConfig(
                horizon=None,
                committed_gates=1,
                iterative_diving_search=False,
                max_frontier_size=None,
                max_compile_time=None,
                heuristic=None,
            )
        ),
    )

    assert explicit_result.status is default_result.status
    assert diagnostics(explicit_result).score == diagnostics(default_result).score
    assert explicit_result.path == default_result.path


@pytest.mark.parametrize("horizon", [None, 1])
@pytest.mark.parametrize("assigned", [False, True])
def test_custom_heuristic_receives_partition_bias(horizon: int | None, *, assigned: bool) -> None:
    """Pass optional pre-partition data to custom estimates in both search modes."""
    architecture = LinearArchitecture(num_sites=2)
    initial_state = create_initial_state(2, architecture)
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=0.3, gate_id=0), 1: Rzz(ion_a=0, ion_b=1, theta=0.5, gate_id=1)}
    received: list[tuple[Mapping[int, str] | None, Mapping[str, tuple[tuple[int, int], ...]] | None]] = []

    def custom_heuristic(
        state: State,
        architecture_: LinearArchitecture,
        circuit: Circuit,
        active_gate_ids: Sequence[int],
        predecessors: Sequence[frozenset[int]],
        /,
        *,
        use_dependencies: bool = True,
        gate_zone: Mapping[int, str] | None = None,
        zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
    ) -> int:
        del state, architecture_, circuit, active_gate_ids, predecessors, use_dependencies
        received.append((gate_zone, zone_site_pairs))
        return 0

    config = LinearCompilerConfig(search=SearchConfig(horizon=horizon, committed_gates=1, heuristic=custom_heuristic))
    expected_gate_zone = {0: "missing", 1: "missing"} if assigned else {}
    expected_zone_site_pairs = {"other": ((0, 1),)}

    result = search_module.search(
        initial_state,
        make_circuit(gates),
        architecture,
        config=config,
        gate_zone=expected_gate_zone,
        zone_site_pairs=expected_zone_site_pairs,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert received
    assert all(item == (expected_gate_zone, expected_zone_site_pairs) for item in received)
