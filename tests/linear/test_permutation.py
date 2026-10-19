# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the permutation strategy of the Linear compiler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from math import pi
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit

import mqt.ionshuttler.linear.compiler as compiler_module
import mqt.ionshuttler.linear.permutation as permutation_module
from mqt.ionshuttler.circuit import Circuit, parse_circuit
from mqt.ionshuttler.core.gates import GateAction, Rx, Rz, Rzz
from mqt.ionshuttler.core.result import CompilationStatus
from mqt.ionshuttler.linear import (
    LinearArchitecture,
    LinearCompiler,
    PermutationConfig,
)
from mqt.ionshuttler.linear.actions import PhysicalSwap, Shuttle
from mqt.ionshuttler.linear.permutation import compile_with_permutations, route_to_placement
from mqt.ionshuttler.linear.result import PermutationDiagnostics
from mqt.ionshuttler.linear.state import State, create_initial_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.linear.result import LinearCompilationResult

TWO_ZONE_ARCHITECTURE = LinearArchitecture(
    num_sites=11,
    processing_zones={"left": [2, 3], "right": [7, 8]},
)


def _compile(
    circuit: QuantumCircuit,
    architecture: LinearArchitecture,
    *,
    action_types: tuple[type[Action], ...] | None = None,
    max_compile_time: float | None = None,
    record_planning_trace: bool = False,
) -> LinearCompilationResult:
    config = PermutationConfig(
        action_types=action_types,
        record_planning_trace=record_planning_trace,
        max_compile_time=max_compile_time,
    )
    return LinearCompiler(architecture, config).compile(circuit)


def _permutation_diagnostics(
    result: LinearCompilationResult,
) -> PermutationDiagnostics:
    assert isinstance(result.diagnostics, PermutationDiagnostics)
    return result.diagnostics


def _random_circuit(num_ions: int, num_gates: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_ions)
    for _ in range(num_gates):
        kind = int(rng.integers(4))
        angle = float(rng.uniform(-pi, pi))
        if kind == 3:
            ion_a, ion_b = (int(ion) for ion in rng.choice(num_ions, size=2, replace=False))
            circuit.rzz(angle, ion_a, ion_b)
        else:
            (circuit.rx, circuit.ry, circuit.rz)[kind](angle, int(rng.integers(num_ions)))
    return circuit


def _gate_time_windows(result: LinearCompilationResult) -> dict[int, tuple[int, int]]:
    windows: dict[int, tuple[int, int]] = {}
    for item in result.schedule.scheduled_actions:
        if isinstance(item.action, GateAction) and item.action.gate_id is not None:
            assert item.action.gate_id not in windows
            windows[item.action.gate_id] = (item.start_time, item.end_time)
    return windows


def _assert_dependencies_hold(result: LinearCompilationResult, circuit: QuantumCircuit) -> None:
    predecessors = parse_circuit(circuit).predecessors
    windows = _gate_time_windows(result)
    for gate_id, (start, _end) in windows.items():
        for predecessor in predecessors[gate_id]:
            assert windows[predecessor][1] <= start


def _apply_route(source: Sequence[int], actions: Sequence[Action]) -> list[int]:
    sites = list(source)
    for action in actions:
        if isinstance(action, Shuttle):
            assert sites[action.ion] == action.src
            assert action.dst not in sites
            assert abs(action.dst - action.src) == 1
            sites[action.ion] = action.dst
        else:
            assert isinstance(action, PhysicalSwap)
            assert (sites[action.ion_a], sites[action.ion_b]) == (action.pos_a, action.pos_b)
            assert abs(action.pos_a - action.pos_b) == 1
            sites[action.ion_a], sites[action.ion_b] = action.pos_b, action.pos_a
    return sites


def _minimum_move_count(
    source: tuple[int, ...],
    target: tuple[int, ...],
    num_sites: int,
) -> int:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        sites = queue.popleft()
        if sites == target:
            return distances[sites]
        ion_at_site = {site: ion for ion, site in enumerate(sites)}
        for ion, site in enumerate(sites):
            for neighbor in (site - 1, site + 1):
                if not 0 <= neighbor < num_sites:
                    continue
                moved = list(sites)
                other = ion_at_site.get(neighbor)
                moved[ion] = neighbor
                if other is not None:
                    moved[other] = site
                key = tuple(moved)
                if key not in distances:
                    distances[key] = distances[sites] + 1
                    queue.append(key)
    msg = "target is unreachable"
    raise AssertionError(msg)


def test_route_to_the_current_placement_is_empty() -> None:
    """Return no transports when every ion already sits on its target site."""
    assert route_to_placement((1, 2, 4), (1, 2, 4), num_sites=6) == []


def test_route_reverses_a_full_line_with_one_swap_per_ion_pair() -> None:
    """Use only physical swaps, one for each pair, when the line has no empty site."""
    source = (0, 1, 2, 3)
    target = (3, 2, 1, 0)

    actions = route_to_placement(source, target, num_sites=4)

    assert all(isinstance(action, PhysicalSwap) for action in actions)
    assert len(actions) == len(list(combinations(source, 2)))
    assert _apply_route(source, actions) == list(target)


def test_route_keeps_ion_order_with_shuttles_only() -> None:
    """Move ions with shuttles when the target keeps their order."""
    actions = route_to_placement((0, 1, 2), (3, 4, 5), num_sites=6)

    assert all(isinstance(action, Shuttle) for action in actions)
    assert len(actions) == 9
    assert _apply_route((0, 1, 2), actions) == [3, 4, 5]


@pytest.mark.parametrize("seed", range(40))
def test_route_uses_the_fewest_adjacent_moves(seed: int) -> None:
    """Match a breadth-first search over all placements on short lines."""
    rng = np.random.default_rng(seed)
    num_sites = int(rng.integers(2, 6))
    num_ions = int(rng.integers(1, num_sites + 1))
    source = tuple(int(site) for site in rng.choice(num_sites, size=num_ions, replace=False))
    target = tuple(int(site) for site in rng.choice(num_sites, size=num_ions, replace=False))

    actions = route_to_placement(source, target, num_sites=num_sites)

    assert _apply_route(source, actions) == list(target)
    assert len(actions) == _minimum_move_count(source, target, num_sites)


def test_route_rejects_repeated_sites() -> None:
    """Reject a placement that puts two ions on one site."""
    with pytest.raises(ValueError, match="distinct sites"):
        route_to_placement((0, 0), (0, 1), num_sites=3)


def test_route_rejects_placements_of_different_ions() -> None:
    """Reject a target that places a different number of ions than the source."""
    with pytest.raises(ValueError, match="same ions"):
        route_to_placement((0, 1), (0,), num_sites=3)


def test_permutation_runs_a_two_ion_gate_in_a_processing_zone() -> None:
    """Move both ions into the zone and run every gate once."""
    architecture = LinearArchitecture(num_sites=6, processing_zones={"zone": [4, 5]})
    circuit = QuantumCircuit(2)
    circuit.rx(0.1, 0)
    circuit.rz(0.2, 1)
    circuit.rzz(0.3, 0, 1)

    result = _compile(circuit, architecture)

    assert result.status is CompilationStatus.SUCCESS
    gates = [item for item in result.schedule.scheduled_actions if isinstance(item.action, GateAction)]
    assert [type(item.action) for item in gates] == [Rz, Rx, Rzz]
    assert {item.processing_zone_id for item in gates} == {None, "zone"}
    assert result.final_state.positions == ((0, 4), (1, 5))
    result.validate()


@pytest.mark.parametrize("seed", range(5))
def test_permutation_completes_random_circuits_on_two_zones(seed: int) -> None:
    """Complete random circuits, keep dependency order, and respect the lower bound."""
    circuit = _random_circuit(num_ions=7, num_gates=300, seed=seed)

    result = _compile(circuit, TWO_ZONE_ARCHITECTURE)

    assert result.status is CompilationStatus.SUCCESS
    assert len(_gate_time_windows(result)) == 300
    _assert_dependencies_hold(result, circuit)
    assert result.diagnostics is not None
    assert result.duration >= result.diagnostics.makespan_lower_bound > 0
    result.validate()


def test_permutation_respects_barriers() -> None:
    """Start gates after a barrier only when all gates before it have finished."""
    architecture = LinearArchitecture(num_sites=8, processing_zones={"left": [1, 2], "right": [5, 6]})
    circuit = QuantumCircuit(4)
    circuit.rzz(0.1, 0, 1)
    circuit.barrier()
    circuit.rx(0.2, 3)

    result = _compile(circuit, architecture)

    assert result.status is CompilationStatus.SUCCESS
    windows = _gate_time_windows(result)
    assert windows[1][0] >= windows[0][1]


def test_permutation_reads_ion_busy_times_by_ion() -> None:
    """Start each ion's gate after that ion's busy time, not after an earlier tuple entry."""
    architecture = LinearArchitecture(num_sites=2)
    circuit = QuantumCircuit(2)
    circuit.rx(0.1, 0)
    circuit.rx(0.2, 1)
    initial_state = State(
        positions=((0, 0), (1, 1)),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=((1, 5), (0, 0)),
        pzs_busy_until=(),
        time=2,
    )

    result = compile_with_permutations(
        initial_state,
        parse_circuit(circuit),
        architecture,
        PermutationConfig(),
    )

    assert result.status is CompilationStatus.SUCCESS
    windows = _gate_time_windows(result)
    assert windows[0][0] == 2
    assert windows[1][0] == 5


@dataclass(frozen=True)
class _ThreeIonGate(GateAction):
    """A gate on three ions, which no Linear processing zone can run."""

    @property
    def ions(self) -> tuple[int, ...]:
        return (0, 1, 2)


def test_permutation_rejects_gates_on_more_than_two_ions() -> None:
    """Reject a gate type that the strategy cannot place in a processing zone."""
    architecture = LinearArchitecture(num_sites=3)
    circuit = Circuit(num_ions=3, gates=(_ThreeIonGate(gate_id=0),), predecessors=(frozenset(),))

    with pytest.raises(TypeError, match="supports no _ThreeIonGate"):
        compile_with_permutations(
            create_initial_state(3, architecture),
            circuit,
            architecture,
            PermutationConfig(),
        )


def test_permutation_is_deterministic() -> None:
    """Return the same schedule and diagnostics for the same input."""
    circuit = _random_circuit(num_ions=6, num_gates=120, seed=3)

    assert _compile(circuit, TWO_ZONE_ARCHITECTURE) == _compile(
        circuit,
        TWO_ZONE_ARCHITECTURE,
    )


def test_permutation_prefers_the_pre_partitioned_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a gate in its preferred zone even when another zone is nearer."""
    monkeypatch.setattr(compiler_module, "compute_gate_zone_assignment", lambda *_args, **_kwargs: {0: "right"})
    circuit = QuantumCircuit(2)
    circuit.rzz(0.1, 0, 1)

    result = _compile(circuit, LinearArchitecture(num_sites=11, processing_zones={"left": [4, 5], "right": [9, 10]}))

    gate = next(item for item in result.schedule.scheduled_actions if isinstance(item.action, Rzz))
    assert gate.processing_zone_id == "right"
    assert _permutation_diagnostics(result).gates_in_preferred_zone == 1


def test_permutation_with_one_implicit_zone_needs_no_transport() -> None:
    """Run every gate in place when the whole line is one processing zone."""
    circuit = _random_circuit(num_ions=4, num_gates=50, seed=0)

    result = _compile(circuit, LinearArchitecture(num_sites=4))

    assert result.status is CompilationStatus.SUCCESS
    assert all(isinstance(action, GateAction) for action in result.path)
    diagnostics = _permutation_diagnostics(result)
    assert diagnostics.groups == diagnostics.groups_without_moves


def test_planning_trace_records_every_group_in_schedule_order() -> None:
    """Record each group once, with its gates, moves, and final placement."""
    circuit = _random_circuit(num_ions=6, num_gates=80, seed=1)

    result = _compile(
        circuit,
        TWO_ZONE_ARCHITECTURE,
        record_planning_trace=True,
    )
    untraced = _compile(circuit, TWO_ZONE_ARCHITECTURE)

    diagnostics = _permutation_diagnostics(result)
    trace = diagnostics.planning_trace
    assert trace is not None
    assert len(trace) == diagnostics.groups
    assert sorted(gate_id for group in trace for gate_id in group.gate_ids) == list(range(80))
    assert sum(group.swaps for group in trace) == sum(isinstance(action, PhysicalSwap) for action in result.path)
    assert sum(group.shuttles for group in trace) == sum(isinstance(action, Shuttle) for action in result.path)
    assert sum(group.swaps == group.shuttles == 0 for group in trace) == diagnostics.groups_without_moves
    assert trace[-1].target_site_of_ion == tuple(site for _ion, site in result.final_state.positions)
    assert _permutation_diagnostics(untraced).planning_trace is None
    assert untraced.schedule == result.schedule


def test_permutation_fails_without_a_zone_for_two_ions() -> None:
    """Return a valid empty result when no zone can hold a two-ion gate."""
    architecture = LinearArchitecture(num_sites=4, processing_zones={"left": [0], "right": [3]})
    circuit = QuantumCircuit(2)
    circuit.rzz(0.1, 0, 1)

    result = _compile(circuit, architecture)

    assert result.status is CompilationStatus.FAILED
    assert result.path == []
    result.validate()


def test_permutation_fails_when_a_needed_swap_is_unavailable() -> None:
    """Stop with a valid partial result when a target needs an unavailable swap."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"zone": [0, 1]})
    circuit = QuantumCircuit(3)
    circuit.rx(0.1, 0)
    circuit.rx(0.2, 1)
    circuit.rzz(0.3, 0, 2)

    result = _compile(circuit, architecture, action_types=(Shuttle, Rx, Rzz))

    assert result.status is CompilationStatus.FAILED
    assert [type(action) for action in result.path] == [Rx, Rx]
    result.validate()


def test_permutation_returns_an_empty_valid_result_without_time() -> None:
    """Report a timeout before the first group when the budget is zero."""
    circuit = _random_circuit(num_ions=3, num_gates=10, seed=0)

    result = _compile(circuit, LinearArchitecture(num_sites=6, processing_zones={"zone": [2, 3]}), max_compile_time=0.0)

    assert result.status is CompilationStatus.TIMEOUT
    assert result.path == []
    result.validate()


def test_interruption_keeps_the_completed_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the groups that finished before an interruption as a valid schedule."""
    schedule_gates = permutation_module._PermutationCompilation.schedule_gates
    calls = [0]

    def interrupt_second_group(
        compilation: permutation_module._PermutationCompilation,
        gate_ids: list[int],
    ) -> None:
        calls[0] += 1
        if calls[0] == 2:
            raise KeyboardInterrupt
        schedule_gates(compilation, gate_ids)

    monkeypatch.setattr(
        permutation_module._PermutationCompilation,
        "schedule_gates",
        interrupt_second_group,
    )
    circuit = _random_circuit(num_ions=6, num_gates=60, seed=2)

    result = _compile(circuit, TWO_ZONE_ARCHITECTURE)

    assert result.status is CompilationStatus.INTERRUPTED
    assert 0 < len(_gate_time_windows(result)) < 60
    assert _permutation_diagnostics(result).groups == 1
    result.validate()
