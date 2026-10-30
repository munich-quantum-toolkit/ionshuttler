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
    from qiskit.dagcircuit import DAGDependency

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

    A barrier still orders every gate on its qubits.

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
    left_out = _barrier_orders_left_out(dependency_graph)

    @cache
    def gate_predecessors(node_id: int) -> frozenset[int]:
        predecessors: set[int] = set()
        for predecessor_id in (*dependency_graph.direct_predecessors(node_id), *left_out.get(node_id, ())):
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


def _barrier_orders_left_out(dependency_graph: DAGDependency) -> dict[int, set[int]]:
    """Return the orders that a barrier requires and that the commutation analysis leaves out.

    Qiskit lets an operation that equals the identity, such as ``rz(2*pi)``,
    commute with a barrier. A barrier orders every operation on its qubits.

    Returns:
        Additional direct predecessors by node ID.
    """
    nodes = tuple(dependency_graph.get_nodes())
    left_out: dict[int, set[int]] = {}
    for barrier in nodes:
        if barrier.op.name != "barrier":
            continue
        qubits = set(barrier.qargs)
        before = set(dependency_graph.predecessors(barrier.node_id))
        after = set(dependency_graph.successors(barrier.node_id))
        for node in nodes:
            if node.node_id == barrier.node_id or qubits.isdisjoint(node.qargs):
                continue
            if node.node_id < barrier.node_id and node.node_id not in before:
                left_out.setdefault(barrier.node_id, set()).add(node.node_id)
            elif node.node_id > barrier.node_id and node.node_id not in after:
                left_out.setdefault(node.node_id, set()).add(barrier.node_id)
    return left_out


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
