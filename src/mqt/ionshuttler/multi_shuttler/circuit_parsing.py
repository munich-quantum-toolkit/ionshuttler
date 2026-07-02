from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.qasm2 import dumps
from qiskit.qasm3 import loads as load_qasm3

from .circuit_types import GateInfo, ParsedCircuit

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_HEADER_PREFIXES = ("OPENQASM", "include", "qreg", "creg", "gate", "barrier", "measure")
_QUBIT_PATTERN = re.compile(r"q\[(\d+)\]")
_NO_QUBITS_EXTRACTED_MSG = (
    "Failed to extract qubits from QASM gate lines. Use normalize_registers=True for non-canonical register names."
)


def is_qasm_file(file_path: Path) -> bool:
    """Return whether the file appears to contain an OpenQASM program."""

    with file_path.open(encoding="utf-8") as file:
        for _ in range(20):
            line = file.readline()
            if not line:
                break
            if "OPENQASM" in line:
                return True
    return False


def extract_qubits_from_gate(gate_line: str) -> list[int]:
    """Extract canonicalized qubit indices from a gate operation line."""

    return [int(match) for match in _QUBIT_PATTERN.findall(gate_line)]


def normalize_qasm_registers(qasm_str: str, *, qreg_name: str = "q", creg_name: str = "c") -> str:
    """Rewrite all quantum registers into a single canonical register order."""

    circuit = _load_quantum_circuit(qasm_str)
    clbit_map: dict[object, object] = {}

    qreg_new = QuantumRegister(circuit.num_qubits, qreg_name)
    if circuit.num_clbits:
        creg_new = ClassicalRegister(circuit.num_clbits, creg_name)
        normalized_circuit = QuantumCircuit(qreg_new, creg_new)
        clbit_map = {old: creg_new[index] for index, old in enumerate(circuit.clbits)}
    else:
        normalized_circuit = QuantumCircuit(qreg_new)

    qubit_map = {old: qreg_new[index] for index, old in enumerate(circuit.qubits)}
    normalized_circuit.compose(
        circuit,
        qubits=[qubit_map[qubit] for qubit in circuit.qubits],
        clbits=[clbit_map[clbit] for clbit in circuit.clbits] if circuit.clbits else None,
        inplace=True,
    )
    return dumps(normalized_circuit)


def parse_qasm_circuit(filename: Path, *, normalize_registers: bool = True) -> ParsedCircuit:
    """Parse a QASM file into stable gate ids and per-gate metadata."""

    qasm_str = filename.read_text(encoding="utf-8")
    if normalize_registers:
        qasm_str = normalize_qasm_registers(qasm_str)

    sequence: list[int] = []
    gate_info: dict[int, GateInfo] = {}
    for line in _iter_gate_lines(qasm_str):
        qubits = extract_qubits_from_gate(line)
        if not qubits:
            raise ValueError(_NO_QUBITS_EXTRACTED_MSG)

        gate_id = len(sequence)
        sequence.append(gate_id)
        gate_info[gate_id] = GateInfo(qubits=tuple(qubits), qasm=line)

    return ParsedCircuit(sequence=sequence, gate_info=gate_info)


def _iter_gate_lines(qasm_str: str) -> Iterator[str]:
    for raw_line in qasm_str.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("//", "#")):
            continue
        if line.startswith(_HEADER_PREFIXES):
            continue
        yield line


def _load_quantum_circuit(qasm_str: str) -> QuantumCircuit:
    try:
        return QuantumCircuit.from_qasm_str(qasm_str)
    except Exception:
        return cast("QuantumCircuit", load_qasm3(qasm_str))
