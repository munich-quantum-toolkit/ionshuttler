# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the internal circuit model."""

import pytest

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.circuit.parser import parse_circuit
from mqt.ionshuttler.linear.actions import Rx, Rxx


def test_circuit_exposes_stable_gate_ids() -> None:
    """Gate identifiers follow immutable circuit order."""
    circuit = Circuit(
        num_ions=2,
        gates=(
            Rx(ion=0, theta=0.5, gate_id=0),
            Rxx(ion_a=0, ion_b=1, theta=1.0, gate_id=1),
        ),
        predecessors=(frozenset(), frozenset({0})),
    )

    assert circuit.gate_ids == range(2)


def test_circuit_rejects_forward_dependencies() -> None:
    """A gate can depend only on an earlier gate occurrence."""
    gate = Rx(ion=0, theta=0.5, gate_id=0)

    with pytest.raises(ValueError, match="each predecessor must identify an earlier gate"):
        Circuit(num_ions=1, gates=(gate,), predecessors=(frozenset({0}),))


def test_parsed_circuit_always_retains_dependencies() -> None:
    """Parsing preserves the dependency graph for compiler policies."""
    circuit = parse_circuit(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        rx(0.5) q[0];
        ry(0.25) q[1];
        rzz(1.0) q[0], q[1];
        """
    )

    assert circuit.predecessors == (frozenset(), frozenset(), frozenset({0, 1}))
    assert tuple(gate.gate_id for gate in circuit.gates) == (0, 1, 2)
