# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for preparing circuit inputs for Linear scheduling."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from mqt.ionshuttler.linear.actions import Rx, Rxx, Ry, Ryy, Rz, Rzz
from mqt.ionshuttler.linear.config import GateTiming
from mqt.ionshuttler.linear.parser import (
    compute_gate_dependencies_from_qasm,
    parse_circuit,
    parse_qasm,
    parse_qasm_file,
    parse_qasm_file_with_dependencies,
    parse_qasm_to_gate_sequence,
    parse_qasm_to_gate_sequence_with_dependencies,
    parse_quantum_circuit,
)

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


def test_parse_qasm2_supports_native_gate_set_and_arithmetic() -> None:
    """Read every supported gate and safely evaluate arithmetic parameters."""
    num_qubits, gates = parse_qasm_to_gate_sequence(QASM2_ALL_GATES)

    assert num_qubits == 3
    assert gates == [
        Rx(ion=0, theta=math.pi / 2),
        Ry(ion=1, theta=-(math.pi / 4)),
        Rz(ion=2, theta=math.pi / 8),
        Rxx(ion_a=0, ion_b=1, theta=math.pi / 3),
        Ryy(ion_a=1, ion_b=2, theta=math.pi / 5),
        Rzz(ion_a=2, ion_b=0, theta=math.pi / 7),
    ]


@pytest.mark.parametrize("version", ["3", "3.0"])
def test_parse_limited_qasm3_ignores_metadata_and_trailing_measurements(version: str) -> None:
    """Accept QASM 3 declarations while ignoring barriers and final measurements."""
    num_qubits, gates = parse_qasm_to_gate_sequence(
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

    assert num_qubits == 2
    assert gates == [
        Ry(ion=1, theta=float("3.14")),
        Rz(ion=0, theta=3 * math.pi / 4),
    ]


def test_gate_timing_controls_durations_and_virtual_implementations() -> None:
    """Use the hardware's configured duration and implementation for each rotation."""
    timing = GateTiming(
        rx=0,
        ry=0,
        rz=1,
        rxx=4,
        ryy=5,
        rzz=6,
        virtual_single_qubit_gates=frozenset({"rx", "ry"}),
    )

    _, gates = parse_qasm_to_gate_sequence(QASM2_ALL_GATES, gate_timing=timing)

    assert gates == [
        Rx(ion=0, theta=math.pi / 2, duration=0, virtual=True),
        Ry(ion=1, theta=-(math.pi / 4), duration=0, virtual=True),
        Rz(ion=2, theta=math.pi / 8, duration=1, virtual=False),
        Rxx(ion_a=0, ion_b=1, theta=math.pi / 3, duration=4),
        Ryy(ion_a=1, ion_b=2, theta=math.pi / 5, duration=5),
        Rzz(ion_a=2, ion_b=0, theta=math.pi / 7, duration=6),
    ]


def test_parser_preserves_zero_duration_physical_gate_mode() -> None:
    """Keep a zero-duration rotation physical when configured that way."""
    timing = GateTiming(rz=0, virtual_single_qubit_gates=frozenset())

    with pytest.warns(UserWarning, match="physical single-qubit gate"):
        _, gates = parse_qasm_to_gate_sequence(QASM2_ALL_GATES, gate_timing=timing)

    assert isinstance(gates[2], Rz)
    assert gates[2].duration == 0
    assert not gates[2].virtual


def test_duration_overrides_keep_unspecified_gate_defaults() -> None:
    """Apply supplied durations and use defaults for gates that are omitted."""
    _, gates = parse_qasm_to_gate_sequence(
        QASM2_ALL_GATES,
        gate_durations={"RX": 2, "rxx": 4},
    )

    assert isinstance(gates[0], Rx)
    assert isinstance(gates[3], Rxx)
    assert isinstance(gates[4], Ryy)
    assert gates[0].duration == 2
    assert gates[2] == Rz(ion=2, theta=math.pi / 8)
    assert gates[3].duration == 4
    assert gates[4].duration == 2

    with pytest.raises(ValueError, match="virtual gate 'rz'"):
        parse_qasm_to_gate_sequence(QASM2_ALL_GATES, gate_durations={"rz": 1})


@pytest.mark.parametrize("gate_name", ["rxxz", "x"])
def test_duration_overrides_reject_unknown_gate_names(gate_name: str) -> None:
    """Reject misspelled and unsupported duration-override keys."""
    with pytest.raises(ValueError, match=rf"unsupported gate name '{gate_name}'"):
        parse_qasm_to_gate_sequence(QASM2_ALL_GATES, gate_durations={gate_name: 2})


def test_duration_overrides_allow_zero_duration_physical_gates() -> None:
    """Keep compact duration overrides consistent with the hardware timing model."""
    with pytest.warns(UserWarning, match="physical single-qubit gate"):
        _, gates = parse_qasm_to_gate_sequence(QASM2_ALL_GATES, gate_durations={"rx": 0})

    assert isinstance(gates[0], Rx)
    assert gates[0].duration == 0
    assert not gates[0].virtual


@pytest.mark.parametrize(
    ("duration", "error_type"),
    [(-1, ValueError), (True, ValueError), (1.5, TypeError)],
)
def test_duration_overrides_reject_invalid_physical_durations(
    duration: object,
    error_type: type[Exception],
) -> None:
    """Reject physical gate durations that cannot be scheduled."""
    durations = cast("dict[str, int]", {"rx": duration})
    with pytest.raises(error_type, match="duration"):
        parse_qasm_to_gate_sequence(QASM2_ALL_GATES, gate_durations=durations)


def test_timing_model_and_duration_overrides_are_mutually_exclusive() -> None:
    """Reject two competing descriptions of the same hardware timing."""
    with pytest.raises(ValueError, match="cannot both be supplied"):
        parse_qasm_to_gate_sequence(
            QASM2_ALL_GATES,
            gate_timing=GateTiming(),
            gate_durations={"rx": 2},
        )


def test_dependencies_follow_each_ions_previous_gate() -> None:
    """Connect each gate to the latest gate acting on the same ions."""
    predecessors, successors = compute_gate_dependencies_from_qasm(QASM2_ALL_GATES)

    assert predecessors == {
        0: frozenset(),
        1: frozenset(),
        2: frozenset(),
        3: frozenset({0, 1}),
        4: frozenset({2, 3}),
        5: frozenset({3, 4}),
    }
    assert successors == {
        0: frozenset({3}),
        1: frozenset({3}),
        2: frozenset({4}),
        3: frozenset({4, 5}),
        4: frozenset({5}),
        5: frozenset(),
    }


def test_barriers_break_direct_dependencies() -> None:
    """Keep barriers as breaks between otherwise adjacent direct dependencies."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    rx(pi) q[0];
    ry(pi) q[1];
    barrier q[0];
    rz(pi) q[0];
    rz(pi) q[1];
    """
    circuit = QuantumCircuit(2)
    circuit.rx(math.pi, 0)
    circuit.ry(math.pi, 1)
    circuit.barrier(0)
    circuit.rz(math.pi, 0)
    circuit.rz(math.pi, 1)

    expected_predecessors = {
        0: frozenset(),
        1: frozenset(),
        2: frozenset(),
        3: frozenset({1}),
    }
    expected_successors = {
        0: frozenset(),
        1: frozenset({3}),
        2: frozenset(),
        3: frozenset(),
    }
    assert compute_gate_dependencies_from_qasm(qasm) == (
        expected_predecessors,
        expected_successors,
    )
    assert parse_quantum_circuit(circuit)[2:] == (
        expected_predecessors,
        expected_successors,
    )


def test_parser_has_stable_shape_with_optional_dependencies() -> None:
    """Always return four items even when dependency computation is disabled."""
    enabled = parse_qasm(QASM2_ALL_GATES)
    disabled = parse_qasm(QASM2_ALL_GATES, use_dependencies=False)

    assert len(enabled) == 4
    assert enabled[2] is not None
    assert enabled[3] is not None
    assert disabled[:2] == enabled[:2]
    assert disabled[2:] == (None, None)


def test_qasm_dependency_convenience_function_keeps_gate_alignment() -> None:
    """Return actions and dependencies in the same circuit order."""
    parsed = parse_qasm_to_gate_sequence_with_dependencies(QASM2_ALL_GATES)

    assert parsed[:2] == parse_qasm_to_gate_sequence(QASM2_ALL_GATES)
    assert parsed[2:] == compute_gate_dependencies_from_qasm(QASM2_ALL_GATES)


def test_qasm_file_helpers_read_utf8_and_keep_the_stable_shape(tmp_path: Path) -> None:
    """Accept explicit paths without guessing whether an input string is a path."""
    qasm_path = tmp_path / "circuit.qasm"
    qasm_path.write_text(f"// π parameter\n{QASM2_ALL_GATES}", encoding="utf-8")

    without_dependencies = parse_qasm_file(qasm_path, use_dependencies=False)
    with_dependencies = parse_qasm_file_with_dependencies(qasm_path)

    assert without_dependencies[:2] == with_dependencies[:2]
    assert without_dependencies[2:] == (None, None)
    assert with_dependencies[2] is not None
    assert with_dependencies[3] is not None


def test_qiskit_and_qasm_inputs_lower_identically() -> None:
    """Prepare equivalent Qiskit and QASM circuits identically."""
    circuit = QuantumCircuit(3)
    circuit.rx(math.pi / 2, 0)
    circuit.ry(-(math.pi / 4), 1)
    circuit.rz(math.pi / 8, 2)
    circuit.rxx(math.pi / 3, 0, 1)
    circuit.ryy(math.pi / 5, 1, 2)
    circuit.rzz(math.pi / 7, 2, 0)

    assert parse_quantum_circuit(circuit) == parse_qasm(QASM2_ALL_GATES)
    assert parse_circuit(circuit) == parse_circuit(QASM2_ALL_GATES)


def test_qiskit_ignores_barriers_and_trailing_measurements() -> None:
    """Ignore barriers and final measurements when collecting gates."""
    circuit = QuantumCircuit(2, 2)
    circuit.rx(math.pi / 2, 0)
    circuit.barrier()
    circuit.measure([0, 1], [0, 1])

    parsed = parse_quantum_circuit(circuit)

    assert parsed[0] == 2
    assert parsed[1] == [Rx(ion=0, theta=math.pi / 2)]


@pytest.mark.parametrize(
    ("qasm", "message"),
    [
        ('include "qelib1.inc";\nqreg q[1];', "Missing OPENQASM header"),
        ('OPENQASM 2.0;\ninclude "qelib1.inc";', "Missing quantum register"),
        ("OPENQASM 2.0;\nqreg q[1];\nx q[0];", "unavailable gate 'x'"),
        ("OPENQASM 2.0;\nqreg q[1];\nrx(foo) q[0];", "Unsupported parameter"),
        ("OPENQASM 2.0;\nqreg q[1];\nrx(pi) q[1];", "outside the quantum register"),
        ("OPENQASM 2.0;\nqreg q[1];\nrxx(pi) q[0], q[0];", "two distinct qubits"),
        (
            "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nrx(pi) q[0];",
            "Unsupported QASM syntax",
        ),
    ],
)
def test_qasm_rejects_unsupported_or_malformed_input(qasm: str, message: str) -> None:
    """Reject circuit constructs that the scheduler cannot faithfully compile."""
    with pytest.raises(ValueError, match=message):
        parse_qasm(qasm)


def test_qiskit_rejects_unsupported_unbound_and_nontrailing_operations() -> None:
    """Fail before scheduling when a Qiskit circuit cannot be interpreted safely."""
    unsupported = QuantumCircuit(1)
    unsupported.x(0)
    with pytest.raises(ValueError, match="unavailable gate 'x'"):
        parse_quantum_circuit(unsupported)

    parameterized = QuantumCircuit(1)
    parameterized.rx(Parameter("theta"), 0)
    with pytest.raises(ValueError, match="unbound parameter"):
        parse_quantum_circuit(parameterized)

    measured = QuantumCircuit(1, 1)
    measured.measure(0, 0)
    measured.rx(math.pi, 0)
    with pytest.raises(ValueError, match="measurements must be trailing"):
        parse_quantum_circuit(measured)


def test_parse_circuit_rejects_unknown_input_types() -> None:
    """Require callers to use one of the documented circuit input forms."""
    with pytest.raises(TypeError, match="QuantumCircuit"):
        parse_circuit(cast("str", object()))
