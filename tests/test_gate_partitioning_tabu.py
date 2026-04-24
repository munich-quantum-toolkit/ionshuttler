"""Tests for the fine-grained tabu gate partitioner."""

from __future__ import annotations

import importlib
from typing import Any, cast

from mqt.ionshuttler.multi_shuttler.circuit_types import GateInfo
from mqt.ionshuttler.multi_shuttler.gate_partitioning_tabu import (
    FineGrainedTabuConfig,
    compute_fine_grained_gate_partition,
)

pytest = cast("Any", importlib.import_module("pytest"))


def _distance_matrix(num_pzs: int) -> list[list[float]]:
    return [[0.0 if i == j else 1.0 for j in range(num_pzs)] for i in range(num_pzs)]


def _sample_gate_info() -> dict[int, GateInfo]:
    return {
        0: GateInfo(qubits=(0, 1), qasm="cx q[0],q[1];"),
        1: GateInfo(qubits=(2, 3), qasm="cx q[2],q[3];"),
        2: GateInfo(qubits=(0, 2), qasm="cx q[0],q[2];"),
        3: GateInfo(qubits=(1,), qasm="x q[1];"),
        4: GateInfo(qubits=(3,), qasm="x q[3];"),
    }


def test_config_defaults_match_prototype_reference_values() -> None:
    config = FineGrainedTabuConfig()

    assert config.balance_penalty == 1.0
    assert config.capacity_weight == 0.5
    assert config.distance_weight_factor == 1.0
    assert config.max_iterations_factor == 20.0
    assert config.tabu_list_length == 200
    assert config.candidate_list_length == 200
    assert config.per_slice_quota is None
    assert config.slack_dropoff == 1.0
    assert config.refresh_every is None
    assert config.randomize_initial is False
    assert config.seed == 0
    assert config.max_layer_depth is None


def test_compute_partition_returns_runtime_neutral_result() -> None:
    gate_info = _sample_gate_info()
    result = compute_fine_grained_gate_partition(
        [0, 1, 2, 3, 4],
        gate_info,
        ["pz1", "pz2"],
        _distance_matrix(2),
        capacity=2,
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
        capacity=2,
    )

    assert result.time_slices[0] == [0, 1]
    assert result.time_slices[1] == [2]


def test_multi_qubit_projection_stays_within_one_cluster() -> None:
    gate_info = _sample_gate_info()
    sequence = [0, 1, 2, 3, 4]
    result = compute_fine_grained_gate_partition(
        sequence,
        gate_info,
        ["pz1", "pz2"],
        _distance_matrix(2),
        capacity=2,
    )

    for slice_gate_ids, qubit_assignment in zip(result.time_slices, result.qubit_assignments_by_slice):
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
        capacity=2,
        config=config,
    )
    second = compute_fine_grained_gate_partition(
        [0, 1, 2, 3, 4],
        gate_info,
        ["pz1", "pz2", "pz3"],
        _distance_matrix(3),
        capacity=2,
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
    assert result.cost_before == 0.0
    assert result.cost_after == 0.0