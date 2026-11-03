# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dependency views shared by circuit schedulers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit import Gate
from qiskit.circuit.commutation_library import SessionCommutationChecker
from qiskit.circuit.library import RXGate, RXXGate, RYGate, RYYGate, RZGate, RZZGate

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

    Two gates keep their order when they share an ion and do not commute, as
    Qiskit's commutation checker decides. A barrier orders every gate on its
    qubits. The returned predecessors imply exactly these orders; they need
    not be the smallest sets that do.

    The analysis walks back along each ion of a gate and stops at a gate that
    already follows every earlier gate on that ion. For typical circuits, time
    and memory therefore grow about linearly with the number of gates.

    Args:
        circuit: Parsed circuit whose gate IDs define schedule identity.

    Returns:
        A circuit whose predecessors order every pair of gates that share an
        ion and do not commute, and every gate across a barrier on its ions.

    Raises:
        ValueError: If a gate type has no circuit name.
    """  # ruff: ignore[docstring-extraneous-exception] - Raised while building the Qiskit operations.
    ordering = _Ordering()
    gate_node: list[int] = []
    for gate_id, gate in enumerate(circuit.gates):
        for targets in circuit.barriers[gate_id]:
            ordering.add(tuple(sorted(targets)), None)
        gate_node.append(ordering.add(gate.ions, _as_qiskit_gate(gate)))
    predecessors = tuple(frozenset(ordering.gates_before(node)) for node in gate_node)
    return Circuit(
        num_ions=circuit.num_ions,
        gates=circuit.gates,
        predecessors=predecessors,
        barriers=circuit.barriers,
    )


class _Ordering:
    """Collect gates and barriers in circuit order with the nodes that each must follow.

    Nodes are numbered in circuit order. A barrier is a node without an
    operation; it commutes with nothing on its qubits. An identity, such as
    ``rz(2*pi)``, commutes with every gate, so only barriers order it.
    """

    def __init__(self) -> None:
        self.ions: list[tuple[int, ...]] = []
        self.operations: list[Gate | None] = []
        self.signatures: list[tuple[object, ...] | None] = []
        self.identity: list[bool] = []
        self.after: list[list[int]] = []
        # For each node and each of its ions: whether the node follows every earlier node on that ion,
        # and whether it follows every earlier node except identities.
        self.follows_all: list[dict[int, bool]] = []
        self.follows_gates: list[dict[int, bool]] = []
        self.on_ion: dict[int, list[int]] = {}
        self.barrier_gates: dict[int, frozenset[int]] = {}
        self.gate_number: dict[int, int] = {}
        self.known: dict[tuple[object, ...], bool] = {}
        self.known_identity: dict[tuple[object, ...], bool] = {}

    def add(self, ions: tuple[int, ...], operation: Gate | None) -> int:
        """Add one gate (with its Qiskit operation) or one barrier (``None``).

        Returns:
            The node number of the gate or barrier.
        """
        node = len(self.ions)
        barrier = operation is None
        signature = None if operation is None else (operation.name, tuple(operation.params), len(ions))
        self.ions.append(ions)
        self.operations.append(operation)
        self.signatures.append(signature)
        self.identity.append(operation is not None and signature is not None and self.is_identity(operation, signature))
        after: list[int] = []
        found: set[int] = set()
        follows_all: dict[int, bool] = {}
        follows_gates: dict[int, bool] = {}
        for ion in ions:
            complete = complete_for_gates = True
            # Walk back along the ion. A node that the new node already follows needs no entry; a node that
            # also follows every earlier node that matters ends the walk. A gate never follows an identity,
            # so it passes identities; a barrier follows every node.
            for earlier in reversed(self.on_ion.get(ion, ())):
                if not barrier and self.identity[earlier]:
                    complete = False
                    continue
                if not (earlier in found or self.follows_found(earlier, after)):
                    if self.commute(node, earlier):
                        complete = complete_for_gates = False
                        continue
                    after.append(earlier)
                    found.add(earlier)
                if barrier and self.follows_all[earlier][ion]:
                    break
                if not barrier and self.follows_gates[earlier][ion]:
                    complete = complete and self.follows_all[earlier][ion]
                    break
            follows_all[ion] = complete
            follows_gates[ion] = complete_for_gates
        for ion in ions:
            self.on_ion.setdefault(ion, []).append(node)
        self.after.append(after)
        self.follows_all.append(follows_all)
        self.follows_gates.append(follows_gates)
        if barrier:
            self.barrier_gates[node] = frozenset(self.gates_before(node))
        else:
            self.gate_number[node] = len(self.gate_number)
        return node

    def is_identity(self, operation: Gate, signature: tuple[object, ...]) -> bool:
        """Return whether an operation commutes with every rotation on each of its qubits, so equals the identity."""
        if signature not in self.known_identity:
            qubits = list(range(operation.num_qubits))
            self.known_identity[signature] = all(
                SessionCommutationChecker.commute(operation, qubits, [], probe, [qubit], [])
                for qubit in qubits
                for probe in (RXGate(0.7), RYGate(0.7), RZGate(0.7))
            )
        return self.known_identity[signature]

    def follows_found(self, earlier: int, after: list[int]) -> bool:
        """Return whether a node in ``after`` must follow ``earlier`` because they do not commute."""
        return any(
            earlier < found
            and not set(self.ions[earlier]).isdisjoint(self.ions[found])
            and not self.commute(earlier, found)
            for found in after
        )

    def gates_before(self, node: int) -> set[int]:
        """Return the gate numbers that ``node`` follows, looking through barriers."""
        gates: set[int] = set()
        for earlier in self.after[node]:
            if earlier in self.gate_number:
                gates.add(self.gate_number[earlier])
            else:
                gates.update(self.barrier_gates[earlier])
        return gates

    def commute(self, first: int, second: int) -> bool:
        """Return whether two nodes commute; nodes on disjoint ions always do, barriers never."""
        first_ions, second_ions = self.ions[first], self.ions[second]
        first_operation, second_operation = self.operations[first], self.operations[second]
        if first_operation is None or second_operation is None:
            return set(first_ions).isdisjoint(second_ions)
        # The answer depends only on the operations and on which ions they share.
        shared = tuple(second_ions.index(ion) if ion in second_ions else -1 for ion in first_ions)
        key = (self.signatures[first], self.signatures[second], shared)
        known = self.known.get(key)
        if known is None:
            local = {ion: position for position, ion in enumerate(dict.fromkeys((*first_ions, *second_ions)))}
            known = SessionCommutationChecker.commute(
                first_operation,
                [local[ion] for ion in first_ions],
                [],
                second_operation,
                [local[ion] for ion in second_ions],
                [],
            )
            self.known[key] = known
        return known


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
