# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dependency views shared by circuit schedulers."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dagdependency
from qiskit.qasm3 import loads as load_qasm3

from .model import Circuit
from .parser import qasm_version

if TYPE_CHECKING:
    from .parser import CircuitInput


def with_commutation_aware_dependencies(circuit: Circuit, source: CircuitInput) -> Circuit:
    """Return a scheduling view that permits commuting gate reordering.

    Args:
        circuit: Parsed circuit whose gate IDs define schedule identity.
        source: Original circuit used to derive commutation relations.

    Returns:
        A circuit with Qiskit's commutation-aware direct predecessors.

    Raises:
        ValueError: If the source operations do not match the parsed gates.
    """
    quantum_circuit = _as_quantum_circuit(source)
    dependency_graph = circuit_to_dagdependency(quantum_circuit)
    qubit_indices = {qubit: index for index, qubit in enumerate(dependency_graph.qubits)}
    operation_nodes = tuple(node for node in dependency_graph.get_nodes() if node.op.name not in {"barrier", "measure"})
    if len(operation_nodes) != len(circuit.gates):
        msg = "source operations do not match the parsed circuit"
        raise ValueError(msg)

    node_to_gate: dict[int, int] = {}
    for gate, node in zip(circuit.gates, operation_nodes, strict=True):
        ions = tuple(qubit_indices[qubit] for qubit in node.qargs)
        if gate.gate_id is None or gate.ions != ions or gate.circuit_name != node.op.name.lower():
            msg = "source operations do not match the parsed circuit"
            raise ValueError(msg)
        node_to_gate[node.node_id] = gate.gate_id

    @cache
    def gate_predecessors(node_id: int) -> frozenset[int]:
        predecessors: set[int] = set()
        for predecessor_id in dependency_graph.direct_predecessors(node_id):
            if predecessor_id in node_to_gate:
                predecessors.add(node_to_gate[predecessor_id])
            else:
                predecessors.update(gate_predecessors(predecessor_id))
        return frozenset(predecessors)

    predecessors = [frozenset[int]() for _gate in circuit.gates]
    for node_id, gate_id in node_to_gate.items():
        predecessors[gate_id] = gate_predecessors(node_id)
    return Circuit(circuit.num_ions, circuit.gates, tuple(predecessors))


def _as_quantum_circuit(source: CircuitInput) -> QuantumCircuit:
    """Return a Qiskit circuit for dependency analysis."""
    if isinstance(source, QuantumCircuit):
        return source
    qasm = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    if qasm_version(qasm) == 3:
        return load_qasm3(qasm)
    return QuantumCircuit.from_qasm_str(qasm)


__all__ = ["with_commutation_aware_dependencies"]
