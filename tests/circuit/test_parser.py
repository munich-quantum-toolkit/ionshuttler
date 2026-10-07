# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for parsing supported circuits into the internal circuit model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from mqt.ionshuttler.circuit import parse_circuit
from mqt.ionshuttler.core.gates import Rx, Rxx, Ry, Ryy, Rz, Rzz

if TYPE_CHECKING:
    from pathlib import Path

QASM2_ALL_GATES = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rx(pi/2) q[0];
ry(-pi/4) q[1];
rz(pi/8) q[2];
rxx(pi/3) q[0], q[1];
ryy(pi/5) q[1], q[2];
rzz(pi/7) q[2], q[0];
"""


def test_qasm2_supports_native_gate_set_and_arithmetic() -> None:
    """Read every supported gate and safely evaluate arithmetic parameters."""
    circuit = parse_circuit(QASM2_ALL_GATES)

    assert circuit.num_ions == 3
    assert circuit.gates == (
        Rx(ion=0, theta=math.pi / 2),
        Ry(ion=1, theta=-(math.pi / 4)),
        Rz(ion=2, theta=math.pi / 8),
        Rxx(ion_a=0, ion_b=1, theta=math.pi / 3),
        Ryy(ion_a=1, ion_b=2, theta=math.pi / 5),
        Rzz(ion_a=2, ion_b=0, theta=math.pi / 7),
    )


@pytest.mark.parametrize("version", ["3", "3.0"])
def test_limited_qasm3_ignores_metadata_and_trailing_measurements(version: str) -> None:
    """Accept QASM 3 declarations while ignoring barriers and final measurements."""
    circuit = parse_circuit(
        f"""
        OPENQASM {version};
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        barrier q;
        ry(3.14) q[1];
        rz(pi/2 + pi/4) q[0];
        measure q[0];
        measure q[1];
        """
    )

    assert circuit.num_ions == 2
    assert circuit.gates == (
        Ry(ion=1, theta=float("3.14")),
        Rz(ion=0, theta=3 * math.pi / 4),
    )


def test_dependencies_follow_each_ions_previous_gate() -> None:
    """Connect each gate to the latest gate acting on the same ions."""
    circuit = parse_circuit(QASM2_ALL_GATES)

    assert circuit.predecessors == (
        frozenset(),
        frozenset(),
        frozenset(),
        frozenset({0, 1}),
        frozenset({2, 3}),
        frozenset({3, 4}),
    )


def test_barriers_order_later_gates_after_earlier_gates_on_their_qubits() -> None:
    """Make each gate after a barrier wait for every earlier gate on the barrier's qubits."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    rx(pi) q[0];
    ry(pi) q[1];
    rx(pi) q[2];
    barrier q[0], q[1];
    rz(pi) q[0];
    rz(pi) q[1];
    rz(pi) q[2];
    """
    circuit = QuantumCircuit(3)
    circuit.rx(math.pi, 0)
    circuit.ry(math.pi, 1)
    circuit.rx(math.pi, 2)
    circuit.barrier(0, 1)
    circuit.rz(math.pi, 0)
    circuit.rz(math.pi, 1)
    circuit.rz(math.pi, 2)

    expected_predecessors = (
        frozenset(),
        frozenset(),
        frozenset(),
        frozenset({0, 1}),
        frozenset({0, 1}),
        frozenset({2}),
    )
    assert parse_circuit(qasm).predecessors == expected_predecessors
    assert parse_circuit(circuit).predecessors == expected_predecessors


@pytest.mark.parametrize("barrier", ["barrier q;", "barrier;", "barrier q[0], q[1];"])
def test_register_barrier_orders_gates_on_other_qubits(barrier: str) -> None:
    """Order a gate after a whole-register barrier behind gates on other qubits."""
    qasm = f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrx(pi) q[1];\n{barrier}\nrz(pi) q[0];\n'

    assert parse_circuit(qasm).predecessors == (frozenset(), frozenset({0}))


def test_consecutive_barriers_apply_in_circuit_order() -> None:
    """Keep consecutive barriers distinct instead of merging their qubits."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    rx(pi) q[0];
    rx(pi) q[2];
    barrier q[0], q[1];
    barrier q[1], q[2];
    rz(pi) q[0];
    rz(pi) q[1];
    """
    circuit = QuantumCircuit(3)
    circuit.rx(math.pi, 0)
    circuit.rx(math.pi, 2)
    circuit.barrier(0, 1)
    circuit.barrier(1, 2)
    circuit.rz(math.pi, 0)
    circuit.rz(math.pi, 1)

    expected_predecessors = (frozenset(), frozenset(), frozenset({0}), frozenset({0, 1}))
    assert parse_circuit(qasm).predecessors == expected_predecessors
    assert parse_circuit(circuit).predecessors == expected_predecessors


def test_path_inputs_are_read_as_utf8_qasm_files(tmp_path: Path) -> None:
    """Accept explicit paths without guessing whether an input string is a path."""
    qasm_path = tmp_path / "circuit.qasm"
    qasm_path.write_text(f"// π parameter\n{QASM2_ALL_GATES}", encoding="utf-8")

    assert parse_circuit(qasm_path) == parse_circuit(QASM2_ALL_GATES)


def test_qiskit_and_qasm_inputs_parse_identically() -> None:
    """Parse equivalent Qiskit and QASM circuits identically."""
    circuit = QuantumCircuit(3)
    circuit.rx(math.pi / 2, 0)
    circuit.ry(-(math.pi / 4), 1)
    circuit.rz(math.pi / 8, 2)
    circuit.rxx(math.pi / 3, 0, 1)
    circuit.ryy(math.pi / 5, 1, 2)
    circuit.rzz(math.pi / 7, 2, 0)

    assert parse_circuit(circuit) == parse_circuit(QASM2_ALL_GATES)


def test_qiskit_ignores_barriers_and_trailing_measurements() -> None:
    """Ignore barriers and final measurements when collecting gates."""
    circuit = QuantumCircuit(2, 2)
    circuit.rx(math.pi / 2, 0)
    circuit.barrier()
    circuit.measure([0, 1], [0, 1])

    parsed = parse_circuit(circuit)

    assert parsed.num_ions == 2
    assert parsed.gates == (Rx(ion=0, theta=math.pi / 2),)


@pytest.mark.parametrize(
    ("qasm", "message"),
    [
        ('include "qelib1.inc";\nqreg q[1];', "Missing OPENQASM header"),
        ('OPENQASM 2.0;\ninclude "qelib1.inc";', "Missing quantum register"),
        ("OPENQASM 2.0;\nqreg q[0];", "num_ions must be positive"),
        ("OPENQASM 2.0;\nqreg q[1];\nx q[0];", "unavailable gate 'x'"),
        ("OPENQASM 2.0;\nqreg q[1];\nrx(foo) q[0];", "Unsupported parameter"),
        ("OPENQASM 2.0;\nqreg q[1];\nrx(pi*) q[0];", r"Invalid parameter expression: pi\*"),
        ("OPENQASM 2.0;\nqreg q[1];\nrx(pi/0) q[0];", "Invalid parameter expression: pi/0"),
        ("OPENQASM 2.0;\nqreg q[1];\nrx(1" + "0" * 400 + ") q[0];", "Invalid parameter expression"),
        ("OPENQASM 2.0;\nqreg q[1];\nrx(pi) q[1];", "outside the quantum register"),
        ("OPENQASM 2.0;\nqreg q[1];\nrxx(pi) q[0], q[0];", "two distinct qubits"),
        ("OPENQASM 2.0;\nqreg q[1];\nbarrier r[0];\nrx(pi) q[0];", "Unsupported QASM syntax"),
        ("OPENQASM 2.0;\nqreg q[1];\nbarrier q[1];\nrx(pi) q[0];", "barrier references a qubit outside"),
        (
            "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nrx(pi) q[0];",
            "measurements must be trailing",
        ),
    ],
)
def test_qasm_rejects_unsupported_or_malformed_input(qasm: str, message: str) -> None:
    """Reject circuit constructs that the scheduler cannot faithfully compile."""
    with pytest.raises(ValueError, match=message):
        parse_circuit(qasm)


def test_qiskit_rejects_unsupported_unbound_and_nontrailing_operations() -> None:
    """Fail before scheduling when a Qiskit circuit cannot be interpreted safely."""
    unsupported = QuantumCircuit(1)
    unsupported.x(0)
    with pytest.raises(ValueError, match="unavailable gate 'x'"):
        parse_circuit(unsupported)

    parameterized = QuantumCircuit(1)
    parameterized.rx(Parameter("theta"), 0)
    with pytest.raises(ValueError, match="unbound parameter"):
        parse_circuit(parameterized)

    measured = QuantumCircuit(1, 1)
    measured.measure(0, 0)
    measured.rx(math.pi, 0)
    with pytest.raises(ValueError, match="measurements must be trailing"):
        parse_circuit(measured)


def test_parse_circuit_rejects_unknown_input_types() -> None:
    """Require callers to use one of the documented circuit input forms."""
    with pytest.raises(TypeError, match="QuantumCircuit"):
        parse_circuit(cast("str", object()))
