# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dependency views shared by circuit schedulers."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import RXGate, RXXGate, RYGate, RYYGate, RZGate, RZZGate
from qiskit.converters import circuit_to_dagdependency

from .model import Circuit

if TYPE_CHECKING:
    from mqt.ionshuttler.core.gates import GateAction

_QISKIT_GATE_TYPES = {
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "rxx": RXXGate,
    "ryy": RYYGate,
    "rzz": RZZGate,
}


def with_commutation_aware_dependencies(circuit: Circuit) -> Circuit:
    """Return a scheduling view that permits commuting gate reordering.

    Args:
        circuit: Parsed circuit whose gate IDs define schedule identity.

    Returns:
        A circuit with Qiskit's commutation-aware direct predecessors.

    Raises:
        ValueError: If the dependency operations do not match the parsed gates.
    """
    quantum_circuit = _as_quantum_circuit(circuit)
    dependency_graph = circuit_to_dagdependency(quantum_circuit)
    qubit_indices = {qubit: index for index, qubit in enumerate(dependency_graph.qubits)}
    operation_nodes = tuple(node for node in dependency_graph.get_nodes() if node.op.name != "barrier")
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
    return Circuit(
        num_ions=circuit.num_ions,
        gates=circuit.gates,
        predecessors=tuple(predecessors),
        barriers=circuit.barriers,
    )


def _as_quantum_circuit(circuit: Circuit) -> QuantumCircuit:
    """Build a Qiskit circuit from the parsed gates and barriers.

    Returns:
        The circuit used for dependency analysis.
    """
    quantum_circuit = QuantumCircuit(circuit.num_ions)
    for gate_id, gate in enumerate(circuit.gates):
        for targets in circuit.barriers[gate_id]:
            quantum_circuit.barrier(*sorted(targets))
        quantum_circuit.append(_as_qiskit_gate(gate), gate.ions)
    for targets in circuit.barriers[-1]:
        quantum_circuit.barrier(*sorted(targets))
    return quantum_circuit


def _as_qiskit_gate(gate: GateAction) -> Gate:
    """Build one Qiskit gate while preserving its parsed name and operands.

    Returns:
        The gate used for dependency analysis.

    Raises:
        ValueError: If the parsed gate type has no circuit name.
    """
    name = gate.circuit_name
    if name is None:
        msg = f"gate type {type(gate).__name__} has no circuit name"
        raise ValueError(msg)
    parameters = [getattr(gate, parameter_name) for parameter_name in gate.parameter_names]
    gate_type = _QISKIT_GATE_TYPES.get(name)
    return Gate(name, len(gate.ions), parameters) if gate_type is None else gate_type(*parameters)


__all__ = ["with_commutation_aware_dependencies"]
