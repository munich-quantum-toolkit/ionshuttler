# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared circuit dependency views."""

from __future__ import annotations

import pytest

from mqt.ionshuttler.circuit import parse_circuit, with_commutation_aware_dependencies


def _qasm(*instructions: str) -> str:
    return "\n".join(("OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[2];", *instructions))


def test_commutation_aware_dependencies_release_commuting_rotations() -> None:
    """Permit equal-axis rotations on one ion to change order."""
    source = _qasm("rx(0.5) q[0];", "rx(0.25) q[0];")

    circuit = with_commutation_aware_dependencies(parse_circuit(source), source)

    assert circuit.predecessors == (frozenset(), frozenset())


def test_commutation_aware_dependencies_retain_noncommuting_rotations() -> None:
    """Keep the order of rotations that do not commute."""
    source = _qasm("rx(0.5) q[0];", "rz(0.25) q[0];")

    circuit = with_commutation_aware_dependencies(parse_circuit(source), source)

    assert circuit.predecessors == (frozenset(), frozenset({0}))


def test_commutation_aware_dependencies_retain_barrier_order() -> None:
    """Do not move commuting gates through a circuit barrier."""
    source = _qasm("rx(0.5) q[0];", "barrier q;", "rx(0.25) q[1];")

    circuit = with_commutation_aware_dependencies(parse_circuit(source), source)

    assert circuit.predecessors == (frozenset(), frozenset({0}))


def test_commutation_aware_dependencies_preserve_gate_ids_across_independent_operations() -> None:
    """Map dependency nodes to stable IDs in source order."""
    source = _qasm("rx(0.5) q[1];", "rx(0.25) q[0];", "rz(0.125) q[1];")

    circuit = with_commutation_aware_dependencies(parse_circuit(source), source)

    assert circuit.predecessors == (frozenset(), frozenset(), frozenset({0}))


@pytest.mark.parametrize("version", ["3", "3.0"])
def test_commutation_aware_dependencies_support_qasm3_text(version: str) -> None:
    """Load OpenQASM 3 text before constructing the dependency view."""
    source = f"""OPENQASM {version};
    include "stdgates.inc";
    qubit[2] q;
    rx(0.5) q[0];
    rx(0.25) q[0];
    """

    circuit = with_commutation_aware_dependencies(parse_circuit(source), source)

    assert circuit.predecessors == (frozenset(), frozenset())
