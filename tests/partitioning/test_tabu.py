# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the fine-grained tabu gate partitioner."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mqt.ionshuttler.partitioning import (
    FineGrainedTabuConfig,
    GateInfo,
    compute_fine_grained_gate_partition,
    tabu,
)


def _distance_matrix(num_pzs: int) -> list[list[float]]:
    """Create a symmetric unit-distance matrix.

    Args:
        num_pzs: Number of processing zones represented by the matrix.

    Returns:
        A square matrix with zero diagonal entries and unit distances elsewhere.
    """
    return [[0.0 if i == j else 1.0 for j in range(num_pzs)] for i in range(num_pzs)]


def _sample_gate_info() -> dict[int, GateInfo]:
    """Return representative one- and two-qubit gate metadata.

    Returns:
        A mapping from stable gate IDs to sample ``GateInfo`` entries.
    """
    return {
        0: GateInfo(qubits=(0, 1), qasm="cx q[0],q[1];"),
        1: GateInfo(qubits=(2, 3), qasm="cx q[2],q[3];"),
        2: GateInfo(qubits=(0, 2), qasm="cx q[0],q[2];"),
        3: GateInfo(qubits=(1,), qasm="x q[1];"),
        4: GateInfo(qubits=(3,), qasm="x q[3];"),
    }


def test_config_defaults_match_prototype_reference_values() -> None:
    config = FineGrainedTabuConfig()

    assert config.balance_penalty == pytest.approx(1.0)
    assert config.capacity_weight == pytest.approx(0.5)
    assert config.distance_weight_factor == pytest.approx(1.0)
    assert config.max_iterations_factor == pytest.approx(20.0)
    assert config.tabu_list_length == 200
    assert config.candidate_list_length == 200
    assert config.per_slice_quota is None
    assert config.slack_dropoff == pytest.approx(1.0)
    assert config.refresh_every is None
    assert config.randomize_initial is False
    assert config.seed == 0
    assert config.max_layer_depth is None


def test_config_rejects_non_positive_integer_limits() -> None:
    """Configuration should reject non-positive integer search limits."""
    factories = (
        lambda: FineGrainedTabuConfig(max_iterations=0),
        lambda: FineGrainedTabuConfig(tabu_list_length=0),
        lambda: FineGrainedTabuConfig(candidate_list_length=-1),
        lambda: FineGrainedTabuConfig(per_slice_quota=0),
        lambda: FineGrainedTabuConfig(refresh_every=-1),
        lambda: FineGrainedTabuConfig(max_layer_depth=0),
    )

    for factory in factories:
        with pytest.raises(ValueError, match="must be positive"):
            factory()


def test_config_rejects_invalid_objective_parameters() -> None:
    """Configuration should reject invalid objective weights and factors."""
    with pytest.raises(ValueError, match="balance_penalty must be non-negative"):
        FineGrainedTabuConfig(balance_penalty=-1.0)
    with pytest.raises(ValueError, match="max_iterations_factor must be positive"):
        FineGrainedTabuConfig(max_iterations_factor=0.0)
    with pytest.raises(ValueError, match="slack_dropoff must be positive"):
        FineGrainedTabuConfig(slack_dropoff=-1.0)


@pytest.mark.parametrize(
    "field",
    [
        "max_iterations",
        "tabu_list_length",
        "candidate_list_length",
        "per_slice_quota",
        "refresh_every",
        "max_layer_depth",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5])
def test_config_rejects_boolean_and_fractional_integer_limits(field: str, value: object) -> None:
    """Integer search limits should reject booleans and fractional values."""
    kwargs: dict[str, Any] = {field: value}

    with pytest.raises(TypeError, match="must be an integer"):
        FineGrainedTabuConfig(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "balance_penalty",
        "capacity_weight",
        "distance_weight_factor",
        "max_iterations_factor",
        "slack_dropoff",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_numeric_settings(field: str, value: float) -> None:
    """Objective settings and factors should contain only finite numbers."""
    kwargs: dict[str, Any] = {field: value}

    with pytest.raises(ValueError, match="must be a finite number"):
        FineGrainedTabuConfig(**kwargs)


def test_compute_partition_rejects_duplicate_gate_ids() -> None:
    """Partitioning should reject sequences containing duplicate gate IDs."""
    with pytest.raises(ValueError, match="must not contain duplicate gate ids"):
        compute_fine_grained_gate_partition(
            [0, 0],
            {0: GateInfo(qubits=(0,), qasm="x q[0];")},
            ["pz1"],
            _distance_matrix(1),
        )


def test_compute_partition_rejects_repeated_processing_zone_names() -> None:
    """Require a distinct name for every processing zone."""
    with pytest.raises(ValueError, match="names must be unique"):
        compute_fine_grained_gate_partition(
            [0],
            {0: GateInfo(qubits=(0,), qasm="x q[0];")},
            ["pz1", "pz1"],
            _distance_matrix(2),
        )


@pytest.mark.parametrize(
    ("capacities", "error", "message"),
    [
        ({"pz1": 2}, ValueError, "exactly the processing-zone names"),
        ({"pz1": 2, "pz2": 2, "other": 2}, ValueError, "exactly the processing-zone names"),
        ({"pz1": 0, "pz2": 2}, ValueError, "must be positive"),
        ({"pz1": True, "pz2": 2}, TypeError, "must be an integer"),
        ({"pz1": 1.5, "pz2": 2}, TypeError, "must be an integer"),
        ([2, 2], TypeError, "must be a mapping"),
    ],
)
def test_compute_partition_rejects_invalid_processing_zone_capacities(
    capacities: Any,
    error: type[Exception],
    message: str,
) -> None:
    """Require one positive integer capacity for every named processing zone."""
    with pytest.raises(error, match=message):
        compute_fine_grained_gate_partition(
            [0],
            {0: GateInfo(qubits=(0,), qasm="x q[0];")},
            ["pz1", "pz2"],
            _distance_matrix(2),
            processing_zone_capacities=capacities,
        )


def test_capacity_cost_uses_each_processing_zone_capacity() -> None:
    """Charge overflow only against the capacity of the occupied zone."""
    counts = [[3, 3]]
    loads = [[6, 3]]

    assert tabu._compute_capacity_cost(counts, loads, (2, 4)) == pytest.approx(6.0)
    assert tabu._compute_capacity_cost(counts, loads, (3, 3)) == pytest.approx(0.0)
    assert tabu._capacity_delta(counts[0], loads[0], 0, 1, 1, 2, (2, 4)) == pytest.approx(-6.0)


def test_omitted_capacities_match_the_uniform_default() -> None:
    """Preserve the inferred uniform capacities for callers that omit the mapping."""
    gate_info = _sample_gate_info()
    args = ([0, 1, 2, 3, 4], gate_info, ["pz1", "pz2"], _distance_matrix(2))

    inferred = compute_fine_grained_gate_partition(*args)
    explicit = compute_fine_grained_gate_partition(
        *args,
        processing_zone_capacities={"pz1": 2, "pz2": 2},
    )

    assert inferred.gate_assignment == explicit.gate_assignment
    assert inferred.qubit_assignments_by_slice == explicit.qubit_assignments_by_slice
    assert inferred.cost_before == pytest.approx(explicit.cost_before)
    assert inferred.cost_after == pytest.approx(explicit.cost_after)


def test_compute_partition_returns_runtime_neutral_result() -> None:
    gate_info = _sample_gate_info()
    result = compute_fine_grained_gate_partition(
        [0, 1, 2, 3, 4],
        gate_info,
        ["pz1", "pz2"],
        _distance_matrix(2),
        processing_zone_capacities={"pz1": 2, "pz2": 2},
    )

    assert set(result.gate_partition_by_pz) == {"pz1", "pz2"}
    assert set(result.gate_assignment) == {0, 1, 2, 3, 4}
    assert result.time_slices
    assert result.qubit_assignments_by_slice
    assert result.cost_before >= result.cost_after
    assert result.move_distance_total >= 0.0
    assert result.optimization_time >= 0.0
    assert not hasattr(result, "slice_plan")


def test_relaxed_slicing_groups_non_conflicting_two_qubit_gates() -> None:
    gate_info = {
        0: GateInfo(qubits=(0, 1), qasm="cx q[0],q[1];"),
        1: GateInfo(qubits=(2, 3), qasm="cx q[2],q[3];"),
        2: GateInfo(qubits=(0, 2), qasm="cx q[0],q[2];"),
    }

    result = compute_fine_grained_gate_partition(
        [0, 1, 2],
        gate_info,
        ["pz1", "pz2"],
        _distance_matrix(2),
        processing_zone_capacities={"pz1": 2, "pz2": 2},
    )

    assert result.time_slices[0] == [0, 1]
    assert result.time_slices[1] == [2]


def _scan_relaxed_slices(
    gate_ids: list[int],
    gate_info: dict[int, GateInfo],
    max_layer_depth: int | None,
) -> list[list[int]]:
    """Build relaxed slices by scanning every remaining gate for each slice.

    A gate with a blocked qubit or at the depth limit blocks all its qubits. A
    gate that would join a third qubit to a pair blocks its qubits and their
    partners. Every other gate joins the slice. A gate without qubits joins no
    slice, and a gate that no empty slice accepts forms a slice on its own.
    """
    pending_gate_ids = [gate_id for gate_id in gate_ids if gate_info[gate_id].qubits]
    time_slices: list[list[int]] = []
    while pending_gate_ids:
        blocked_qubits: set[int] = set()
        depth_by_qubit: dict[int, int] = {}
        component_by_qubit: dict[int, set[int]] = {}
        selected_gate_ids: list[int] = []
        for gate_id in pending_gate_ids:
            qubits = tuple(dict.fromkeys(gate_info[gate_id].qubits))
            component = set().union(*(component_by_qubit.get(qubit, {qubit}) for qubit in qubits))
            if blocked_qubits.intersection(qubits) or (
                max_layer_depth is not None and any(depth_by_qubit.get(qubit, 0) >= max_layer_depth for qubit in qubits)
            ):
                blocked_qubits.update(qubits)
            elif len(component) > 2:
                blocked_qubits.update(component if len(qubits) == 2 else qubits)
            else:
                selected_gate_ids.append(gate_id)
                for qubit in qubits:
                    depth_by_qubit[qubit] = depth_by_qubit.get(qubit, 0) + 1
                    component_by_qubit[qubit] = component
                    if max_layer_depth is not None and depth_by_qubit[qubit] >= max_layer_depth:
                        blocked_qubits.add(qubit)
        if not selected_gate_ids:
            selected_gate_ids = pending_gate_ids[:1]
        time_slices.append(selected_gate_ids)
        pending_gate_ids = [gate_id for gate_id in pending_gate_ids if gate_id not in selected_gate_ids]
    return time_slices


@pytest.mark.parametrize("seed", range(100))
def test_relaxed_slicing_matches_a_scan_of_all_remaining_gates(seed: int) -> None:
    """Skip only gates that cannot change a slice, so the slices match a full scan."""
    rng = np.random.default_rng(seed)
    num_qubits = int(rng.integers(3, 8))
    gate_info: dict[int, GateInfo] = {}
    for gate_id in range(int(rng.integers(1, 80))):
        arity = int(rng.choice([0, 1, 2, 2, 2, 3]))
        qubits = tuple(int(qubit) for qubit in rng.choice(num_qubits, size=arity, replace=False))
        gate_info[gate_id] = GateInfo(qubits=qubits, qasm="gate")
    gate_ids = list(gate_info)
    max_layer_depth = None if seed % 3 else int(rng.integers(1, 4))

    slices = tabu._build_time_slices_relaxed(gate_ids, gate_info, num_qubits, max_layer_depth=max_layer_depth)

    assert slices == _scan_relaxed_slices(gate_ids, gate_info, max_layer_depth)


def test_multi_qubit_projection_stays_within_one_cluster() -> None:
    gate_info = _sample_gate_info()
    sequence = [0, 1, 2, 3, 4]
    result = compute_fine_grained_gate_partition(
        sequence,
        gate_info,
        ["pz1", "pz2"],
        _distance_matrix(2),
        processing_zone_capacities={"pz1": 2, "pz2": 2},
    )

    for slice_gate_ids, qubit_assignment in zip(result.time_slices, result.qubit_assignments_by_slice, strict=True):
        for gate_id in slice_gate_ids:
            qubits = gate_info[gate_id].qubits
            if not qubits:
                continue
            cluster = qubit_assignment[qubits[0]]
            assert all(qubit_assignment[qubit] == cluster for qubit in qubits[1:])


def test_seeded_randomized_runs_are_deterministic() -> None:
    gate_info = _sample_gate_info()
    config = FineGrainedTabuConfig(randomize_initial=True, seed=7, max_iterations=12)

    first = compute_fine_grained_gate_partition(
        [0, 1, 2, 3, 4],
        gate_info,
        ["pz1", "pz2", "pz3"],
        _distance_matrix(3),
        processing_zone_capacities={"pz1": 2, "pz2": 2, "pz3": 2},
        config=config,
    )
    second = compute_fine_grained_gate_partition(
        [0, 1, 2, 3, 4],
        gate_info,
        ["pz1", "pz2", "pz3"],
        _distance_matrix(3),
        processing_zone_capacities={"pz1": 2, "pz2": 2, "pz3": 2},
        config=config,
    )

    assert first.gate_assignment == second.gate_assignment
    assert first.gate_partition_by_pz == second.gate_partition_by_pz
    assert first.time_slices == second.time_slices
    assert first.qubit_assignments_by_slice == second.qubit_assignments_by_slice


def test_empty_sequence_returns_empty_partition_result() -> None:
    result = compute_fine_grained_gate_partition(
        [],
        {},
        ["pz1", "pz2"],
        _distance_matrix(2),
    )

    assert result.gate_partition_by_pz == {"pz1": [], "pz2": []}
    assert result.gate_assignment == {}
    assert result.time_slices == []
    assert result.qubit_assignments_by_slice == []
    assert result.cost_before == pytest.approx(0.0)
    assert result.cost_after == pytest.approx(0.0)


def test_consider_supernode_moves_returns_pre_move_balance_delta() -> None:
    contraction = tabu._SliceContraction(
        supernodes=[tabu._Supernode(id=0, qubits=(0,), load=2)],
        qubit_to_supernode={0: 0},
        required_edges={},
        required_unary={0},
        cluster_assignment=None,
        cluster_loads=None,
    )
    slice_loads = [[5, 1]]

    best_move = tabu._consider_supernode_moves(
        contraction=contraction,
        slice_index=0,
        supernode_id=0,
        num_pzs=2,
        slice_counts=[[1, 0]],
        slice_loads=slice_loads,
        active_counts_per_slice=[{0: 1}],
        active_loads_per_slice=[{0: 2}],
        qubit_assignments_by_slice=[[0]],
        distance_matrix=None,
        slack_weights=None,
        capacities=None,
        config=FineGrainedTabuConfig(balance_penalty=1.0, capacity_weight=0.0, distance_weight_factor=0.0),
        current_cost=0.0,
        best_cost=0.0,
        tabu_set=set(),
        best_move_state=tabu._MoveEvaluation(),
    )

    assert best_move.move == (0, contraction.supernodes[0], 1)
    assert best_move.balance_delta == pytest.approx(tabu._balance_delta(slice_loads[0], 0, 1, 2, 2))

    moved_slice_loads = [3, 3]
    assert best_move.balance_delta != pytest.approx(tabu._balance_delta(moved_slice_loads, 0, 1, 2, 2))
