# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared circuit dependency views."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.circuit import Gate
from qiskit.circuit.commutation_library import SessionCommutationChecker
from qiskit.circuit.library import RXGate, RXXGate, RYGate, RYYGate, RZGate, RZZGate

from mqt.ionshuttler.circuit import Circuit, parse_circuit, with_commutation_aware_dependencies
from mqt.ionshuttler.core.gates import Measure, Reset, Rx, Rxx, Ry, Ryy, Rz, Rzz

if TYPE_CHECKING:
    from mqt.ionshuttler.core.gates import GateAction

_ANGLES = (0.3, math.pi / 2, math.pi, 2 * math.pi, 0.0, -math.pi / 2)
_ROTATIONS = (Rx, Ry, Rz, Rx, Ry, Rz)
_PAIR_ROTATIONS = (Rxx, Ryy, Rzz, Rzz)
_QISKIT_ROTATIONS = {"rx": RXGate, "ry": RYGate, "rz": RZGate, "rxx": RXXGate, "ryy": RYYGate, "rzz": RZZGate}


def _qasm(*instructions: str) -> str:
    return "\n".join(("OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[2];", *instructions))


def _random_circuit(seed: int, num_ions: int, num_gates: int, barrier_chance: float = 0.08) -> Circuit:
    """Return a circuit of rotations (some equal to the identity), measurements, resets, and barriers."""
    rng = np.random.default_rng(seed)
    gates: list[GateAction] = []
    predecessors: list[frozenset[int]] = []
    barriers: list[tuple[frozenset[int], ...]] = []
    latest: dict[int, int] = {}
    for gate_id in range(num_gates):
        targets = frozenset(int(ion) for ion in rng.choice(num_ions, size=rng.integers(1, num_ions + 1), replace=False))
        barriers.append((targets,) if rng.random() < barrier_chance else ())
        choice = int(rng.integers(len(_ROTATIONS) + len(_PAIR_ROTATIONS) + 2))
        angle = _ANGLES[int(rng.integers(len(_ANGLES)))]
        gate: GateAction
        if choice < len(_ROTATIONS):
            gate = _ROTATIONS[choice](ion=int(rng.integers(num_ions)), theta=angle, gate_id=gate_id)
        elif choice < len(_ROTATIONS) + len(_PAIR_ROTATIONS):
            first, second = (int(ion) for ion in rng.choice(num_ions, size=2, replace=False))
            gate = _PAIR_ROTATIONS[choice - len(_ROTATIONS)](ion_a=first, ion_b=second, theta=angle, gate_id=gate_id)
        elif choice == len(_ROTATIONS) + len(_PAIR_ROTATIONS):
            gate = Measure(ion=int(rng.integers(num_ions)), gate_id=gate_id)
        else:
            gate = Reset(ion=int(rng.integers(num_ions)), gate_id=gate_id)
        gates.append(gate)
        predecessors.append(frozenset(latest[ion] for ion in gate.ions if ion in latest))
        for ion in gate.ions:
            latest[ion] = gate_id
    barriers.append(())
    return Circuit(num_ions, tuple(gates), tuple(predecessors), tuple(barriers))


def _alternating_blocks_circuit(num_gates: int) -> Circuit:
    """Return blocks of four gates on ion 0 that alternate between Z and X rotations.

    The gates of one block commute. On ion 0, the gates of neighboring blocks do not.
    """
    gates: list[GateAction] = []
    predecessors: list[frozenset[int]] = []
    latest: dict[int, int] = {}
    for gate_id in range(num_gates):
        partner = 1 + gate_id % 2
        gate: GateAction
        if (gate_id // 4) % 2 == 1:
            gate = Rx(ion=partner if gate_id % 2 else 0, theta=0.3, gate_id=gate_id)
        elif gate_id % 2:
            gate = Rzz(ion_a=0, ion_b=partner, theta=0.3, gate_id=gate_id)
        else:
            gate = Rz(ion=0, theta=0.3, gate_id=gate_id)
        gates.append(gate)
        predecessors.append(frozenset(latest[ion] for ion in gate.ions if ion in latest))
        for ion in gate.ions:
            latest[ion] = gate_id
    return Circuit(3, tuple(gates), tuple(predecessors), ((),) * (num_gates + 1))


def _rounds_with_identities_circuit(rounds: int) -> Circuit:
    """Return measurement rounds on ion 0, each with an ``rz(-2*pi)`` as MQT Core synthesis produces it."""
    gates: list[GateAction] = []
    for _ in range(rounds):
        gates.extend((
            Reset(ion=0, gate_id=len(gates)),
            Rzz(ion_a=0, ion_b=1, theta=-math.pi / 2, gate_id=len(gates) + 1),
            Rzz(ion_a=0, ion_b=2, theta=-math.pi / 2, gate_id=len(gates) + 2),
            Rz(ion=0, theta=-2 * math.pi, gate_id=len(gates) + 3),
            Rx(ion=0, theta=math.pi / 2, gate_id=len(gates) + 4),
            Measure(ion=0, gate_id=len(gates) + 5),
        ))
    predecessors: list[frozenset[int]] = []
    latest: dict[int, int] = {}
    for gate_id, gate in enumerate(gates):
        predecessors.append(frozenset(latest[ion] for ion in gate.ions if ion in latest))
        for ion in gate.ions:
            latest[ion] = gate_id
    return Circuit(3, tuple(gates), tuple(predecessors), ((),) * (len(gates) + 1))


def _qiskit_operation(gate: GateAction) -> Gate:
    rotation = _QISKIT_ROTATIONS.get(gate.circuit_name or "")
    if rotation is None:
        return Gate(gate.circuit_name or "", len(gate.ions), [])
    return rotation(*(getattr(gate, name) for name in gate.parameter_names))


def _orders(predecessors: tuple[frozenset[int], ...]) -> list[int]:
    """Return, for each gate, a bit set of every gate that must run before it."""
    before: list[int] = []
    for gate_predecessors in predecessors:
        bits = 0
        for predecessor in gate_predecessors:
            bits |= before[predecessor] | (1 << predecessor)
        before.append(bits)
    return before


def _pairwise_orders(circuit: Circuit) -> list[int]:
    """Order each pair of gates or barriers that share an ion, unless both are gates that commute."""
    nodes: list[tuple[tuple[int, ...], Gate | None]] = []
    gate_nodes: list[int] = []
    for gate_id, gate in enumerate(circuit.gates):
        nodes.extend((tuple(sorted(targets)), None) for targets in circuit.barriers[gate_id])
        gate_nodes.append(len(nodes))
        nodes.append((gate.ions, _qiskit_operation(gate)))
    before: list[int] = []
    for later, (later_ions, later_operation) in enumerate(nodes):
        bits = 0
        for earlier, (earlier_ions, earlier_operation) in enumerate(nodes[:later]):
            if set(earlier_ions).isdisjoint(later_ions):
                continue
            if (
                earlier_operation is None
                or later_operation is None
                or not SessionCommutationChecker.commute(
                    earlier_operation, list(earlier_ions), [], later_operation, list(later_ions), []
                )
            ):
                bits |= before[earlier] | (1 << earlier)
        before.append(bits)
    gate_of_node = {node: gate_id for gate_id, node in enumerate(gate_nodes)}
    return [
        sum(1 << gate_of_node[earlier] for earlier in gate_of_node if (before[node] >> earlier) & 1)
        for node in gate_nodes
    ]


@pytest.mark.parametrize("seed", range(60))
def test_commutation_aware_dependencies_order_exactly_the_noncommuting_pairs(seed: int) -> None:
    """Imply the same order as ordering every non-commuting pair on a shared ion and every pair across a barrier."""
    circuit = _random_circuit(seed, num_ions=2 + seed % 4, num_gates=10 + seed % 50)

    dependencies = with_commutation_aware_dependencies(circuit)

    assert _orders(dependencies.predecessors) == _pairwise_orders(circuit)


def test_commutation_aware_dependencies_scale_to_long_circuits() -> None:
    """Analyze a 20 000-gate circuit in seconds; checking all pairs would take minutes and gigabytes."""
    circuit = _random_circuit(1, num_ions=24, num_gates=20_000, barrier_chance=0.001)

    began = time.perf_counter()
    dependencies = with_commutation_aware_dependencies(circuit)

    assert time.perf_counter() - began < 60
    assert len(dependencies.predecessors) == 20_000


def test_commutation_aware_dependencies_order_alternating_blocks_like_the_pairwise_check() -> None:
    """Order blocks of commuting rotations like the check of every pair."""
    circuit = _alternating_blocks_circuit(200)

    dependencies = with_commutation_aware_dependencies(circuit)

    assert _orders(dependencies.predecessors) == _pairwise_orders(circuit)


def test_commutation_aware_dependencies_scale_to_rounds_with_identity_rotations() -> None:
    """Analyze 30 000 gates in seconds when each measurement round contains a rotation that equals the identity."""
    circuit = _rounds_with_identities_circuit(5_000)

    began = time.perf_counter()
    dependencies = with_commutation_aware_dependencies(circuit)

    assert time.perf_counter() - began < 60
    assert len(dependencies.predecessors) == 30_000


def test_commutation_aware_dependencies_release_commuting_rotations() -> None:
    """Permit equal-axis rotations on one ion to change order."""
    source = _qasm("rx(0.5) q[0];", "rx(0.25) q[0];")

    circuit = with_commutation_aware_dependencies(parse_circuit(source))

    assert circuit.predecessors == (frozenset(), frozenset())


def test_commutation_aware_dependencies_retain_noncommuting_rotations() -> None:
    """Keep the order of rotations that do not commute."""
    source = _qasm("rx(0.5) q[0];", "rz(0.25) q[0];")

    circuit = with_commutation_aware_dependencies(parse_circuit(source))

    assert circuit.predecessors == (frozenset(), frozenset({0}))


def test_commutation_aware_dependencies_retain_barrier_order() -> None:
    """Do not move commuting gates through a circuit barrier."""
    source = _qasm("rx(0.5) q[0];", "barrier q;", "rx(0.25) q[1];")

    circuit = with_commutation_aware_dependencies(parse_circuit(source))

    assert circuit.predecessors == (frozenset(), frozenset({0}))


def test_commutation_aware_dependencies_keep_identity_rotations_on_their_side_of_a_barrier() -> None:
    """Order rotations that equal the identity at a barrier, like every other gate."""
    source = _qasm("rz(6.283185307179586) q[0];", "barrier q;", "rx(0.25) q[1];", "rz(-6.283185307179586) q[1];")

    circuit = with_commutation_aware_dependencies(parse_circuit(source))

    assert circuit.predecessors == (frozenset(), frozenset({0}), frozenset({0}))


def test_commutation_aware_dependencies_preserve_gate_ids_across_independent_operations() -> None:
    """Map dependency nodes to stable IDs in source order."""
    source = _qasm("rx(0.5) q[1];", "rx(0.25) q[0];", "rz(0.125) q[1];")

    circuit = with_commutation_aware_dependencies(parse_circuit(source))

    assert circuit.predecessors == (frozenset(), frozenset(), frozenset({0}))


@pytest.mark.parametrize("version", ["3", "3.0"])
def test_commutation_aware_dependencies_support_qasm3_text(version: str) -> None:
    """Construct the dependency view from parsed OpenQASM 3 operations."""
    source = f"""// generated circuit
    // the version header follows leading comments
    OPENQASM {version};
    include "stdgates.inc";
    qubit[2] q;
    rx(0.5) q[0];
    rx(0.25) q[0];
    """

    circuit = with_commutation_aware_dependencies(parse_circuit(source))

    assert circuit.predecessors == (frozenset(), frozenset())
