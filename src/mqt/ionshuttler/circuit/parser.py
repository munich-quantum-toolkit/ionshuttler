# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Parse supported circuits into the internal circuit model."""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, SupportsFloat, SupportsIndex, cast

from qiskit import QuantumCircuit

from mqt.ionshuttler.circuit.model import Circuit
from mqt.ionshuttler.core.gates import BUILTIN_GATE_TYPES, GateAction

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

GateRecord = tuple[type[GateAction], tuple[float, ...], tuple[int, ...]]
Barriers = dict[int, tuple[frozenset[int], ...]]
CircuitInput = QuantumCircuit | str | Path

_GATE_PATTERN = re.compile(r"^([A-Za-z_]\w*)(?:\((.*)\))?\s+(.+);$")
_QUBIT_OPERAND_PATTERN = re.compile(r"^q\[(\d+)\]$")
_QASM2_QREG_PATTERN = re.compile(r"^qreg\s+q\[(\d+)\];$")
_QASM3_QREG_PATTERN = re.compile(r"^qubit\[(\d+)\]\s+q;$")
_CLASSICAL_REGISTER_PATTERN = re.compile(r"^(?:creg\s+\w+\[\d+\]|bit\[\d+\]\s+\w+);$")
_BARRIER_PATTERN = re.compile(r"^barrier\b.*;$")
_MEASURE_PATTERN = re.compile(r"^measure\b.*;$")


def parse_circuit(
    circuit: CircuitInput,
    *,
    gate_types: Sequence[type[GateAction]] | None = None,
) -> Circuit:
    """Return the internal immutable form of a supported circuit.

    Strings are interpreted as QASM text, while :class:`pathlib.Path` inputs
    are read as UTF-8 files. The parsed circuit always retains the real
    dependency graph. A compiler can impose a serial scheduling policy without
    changing these circuit facts.

    Args:
        circuit: Circuit to parse.
        gate_types: Gate classes understood by the circuit frontend.

    Returns:
        The parsed circuit.

    Raises:
        TypeError: If ``circuit`` has an unsupported type.
    """
    registry = _gate_type_registry(gate_types)
    if isinstance(circuit, Path):
        circuit = circuit.read_text(encoding="utf-8")
    if isinstance(circuit, QuantumCircuit):
        num_qubits, records, barriers = _records_from_quantum_circuit(circuit, registry)
    elif isinstance(circuit, str):
        num_qubits, records, barriers = _records_from_qasm(circuit, registry)
    else:
        msg = "circuit must be a QuantumCircuit, QASM string, or pathlib.Path"
        raise TypeError(msg)
    return Circuit(
        num_ions=num_qubits,
        gates=tuple(_lower_gate(record, gate_id=gate_id) for gate_id, record in enumerate(records)),
        predecessors=_compute_predecessors(records, barriers),
    )


def _gate_type_registry(
    gate_types: Sequence[type[GateAction]] | None,
) -> dict[str, type[GateAction]]:
    """Index available gate classes by their circuit names.

    Args:
        gate_types: Gate classes to register, or ``None`` for the built-in gates.

    Returns:
        Gate classes keyed by lowercase circuit name.

    Raises:
        TypeError: If an entry is not a ``GateAction`` subclass.
        ValueError: If multiple classes declare the same circuit name.
    """
    registry: dict[str, type[GateAction]] = {}
    for gate_type in BUILTIN_GATE_TYPES if gate_types is None else gate_types:
        if not isinstance(gate_type, type) or not issubclass(gate_type, GateAction):
            msg = "gate_types must contain GateAction subclasses"
            raise TypeError(msg)
        name = gate_type.circuit_name
        if name is None:
            continue
        normalized_name = name.lower()
        if normalized_name in registry:
            msg = f"duplicate circuit gate name {normalized_name!r}"
            raise ValueError(msg)
        registry[normalized_name] = gate_type
    return registry


def _gate_record_from_qasm_line(
    line: str,
    gate_types: Mapping[str, type[GateAction]],
) -> GateRecord | None:
    """Read one supported gate instruction from a QASM line.

    Args:
        line: QASM statement to inspect.
        gate_types: Available gate classes keyed by circuit name.

    Returns:
        The parsed gate record, or ``None`` if the line is not a supported gate statement.

    Raises:
        ValueError: If the gate is unavailable or a parameter expression is unsupported.
    """
    match = _GATE_PATTERN.match(line)
    if match is None:
        return None
    gate_name, parameter_text, operand_text = match.groups()
    gate_type = gate_types.get(gate_name.lower())
    if gate_type is None:
        msg = f"circuit requires unavailable gate {gate_name!r}"
        raise ValueError(msg)
    parameters = (
        ()
        if parameter_text is None or not parameter_text.strip()
        else tuple(_safe_eval(item.strip()) for item in parameter_text.split(","))
    )
    ions: list[int] = []
    for operand in operand_text.split(","):
        operand_match = _QUBIT_OPERAND_PATTERN.fullmatch(operand.strip())
        if operand_match is None:
            return None
        ions.append(int(operand_match.group(1)))
    return gate_type, parameters, tuple(ions)


def _records_from_qasm(
    qasm: str,
    gate_types: Mapping[str, type[GateAction]],
) -> tuple[int, list[GateRecord], Barriers]:
    """Extract gate records and barriers from QASM text.

    Args:
        qasm: OpenQASM source text.
        gate_types: Available gate classes keyed by circuit name.

    Returns:
        The qubit count, gate records, and barriers.

    Raises:
        ValueError: If the input is malformed, unsupported, or references invalid qubits.
    """
    num_qubits: int | None = None
    records: list[GateRecord] = []
    # ``None`` marks a barrier on the whole register, whose size may be unknown here.
    barrier_targets: dict[int, list[frozenset[int] | None]] = {}
    header_seen = False
    measurement_seen = False

    for line in _significant_qasm_lines(qasm):
        if _CLASSICAL_REGISTER_PATTERN.match(line):
            continue
        if _BARRIER_PATTERN.match(line):
            barrier_targets.setdefault(len(records), []).append(_qasm_barrier_targets(line))
            continue
        if _MEASURE_PATTERN.match(line):
            measurement_seen = True
            continue
        if measurement_seen:
            msg = "measurements must be trailing"
            raise ValueError(msg)
        if line in {"OPENQASM 2.0;", "OPENQASM 3;", "OPENQASM 3.0;"}:
            header_seen = True
            continue
        if line in {'include "qelib1.inc";', 'include "stdgates.inc";'}:
            continue
        if match := _QASM2_QREG_PATTERN.match(line) or _QASM3_QREG_PATTERN.match(line):
            num_qubits = int(match.group(1))
            continue
        if record := _gate_record_from_qasm_line(line, gate_types):
            records.append(record)
            continue
        msg = f"Unsupported QASM syntax: {line}"
        raise ValueError(msg)

    if not header_seen:
        msg = "Missing OPENQASM header"
        raise ValueError(msg)
    if num_qubits is None:
        msg = "Missing quantum register declaration"
        raise ValueError(msg)
    _validate_gate_records(records, num_qubits)
    barriers = {
        gate_id: tuple(frozenset(range(num_qubits)) if targets is None else targets for targets in gate_barriers)
        for gate_id, gate_barriers in barrier_targets.items()
    }
    _validate_barriers(barriers, num_qubits)
    return num_qubits, records, barriers


def _records_from_quantum_circuit(
    circuit: QuantumCircuit,
    gate_types: Mapping[str, type[GateAction]],
) -> tuple[int, list[GateRecord], Barriers]:
    """Extract gate records and barriers from a Qiskit circuit.

    Args:
        circuit: Qiskit circuit to inspect.
        gate_types: Available gate classes keyed by circuit name.

    Returns:
        The qubit count, gate records, and barriers.

    Raises:
        ValueError: If an operation is unavailable, unsupported, or has invalid operands or parameters.
    """
    records: list[GateRecord] = []
    barrier_targets: dict[int, list[frozenset[int]]] = {}
    measurement_seen = False
    for instruction in circuit.data:
        operation = instruction.operation
        gate_name = operation.name.lower()
        if gate_name == "barrier":
            barrier_targets.setdefault(len(records), []).append(
                frozenset(circuit.find_bit(qubit).index for qubit in instruction.qubits)
            )
            continue
        if gate_name == "measure":
            measurement_seen = True
            continue
        if measurement_seen:
            msg = "measurements must be trailing"
            raise ValueError(msg)
        gate_type = gate_types.get(gate_name)
        if gate_type is None:
            msg = f"circuit requires unavailable gate {operation.name!r}"
            raise ValueError(msg)
        if instruction.clbits:
            msg = f"classically controlled operation {operation.name!r} is unsupported"
            raise ValueError(msg)
        ions = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        parameters = tuple(_numeric_parameter(parameter, operation.name) for parameter in operation.params)
        records.append((gate_type, parameters, ions))

    _validate_gate_records(records, circuit.num_qubits)
    return (
        circuit.num_qubits,
        records,
        {gate_id: tuple(gate_barriers) for gate_id, gate_barriers in barrier_targets.items()},
    )


def _lower_gate(record: GateRecord, *, gate_id: int) -> GateAction:
    """Construct a gate occurrence from a parsed circuit record.

    Args:
        record: Gate class, parameters, and ion operands to lower.
        gate_id: Stable circuit-order identity of the gate occurrence.

    Returns:
        The constructed gate action.
    """
    gate_type, parameters, ions = record
    return gate_type.from_instruction(ions, parameters, gate_id=gate_id)


def _compute_predecessors(
    records: list[GateRecord],
    barriers: Barriers,
) -> tuple[frozenset[int], ...]:
    """Return the gates that directly precede each gate.

    A gate depends on the latest earlier gate on each of its qubits. A barrier
    synchronizes its qubits: a later gate on any of these qubits depends on the
    latest earlier gate on every qubit of the barrier. Barriers before the
    same gate apply in circuit order.

    Args:
        records: Gate records in circuit order.
        barriers: Barrier qubit sets keyed by the identifier of the next gate.

    Returns:
        The direct predecessors of each gate in circuit order.
    """
    # The gates that a later gate on each qubit must wait for.
    latest_gates_by_ion: dict[int, frozenset[int]] = {}
    predecessors: list[frozenset[int]] = []
    for gate_id, (_, _, ions) in enumerate(records):
        for targets in barriers.get(gate_id, ()):
            synchronized = frozenset[int]().union(*(latest_gates_by_ion.get(ion, frozenset()) for ion in targets))
            for ion in targets:
                latest_gates_by_ion[ion] = synchronized
        predecessors.append(frozenset[int]().union(*(latest_gates_by_ion.get(ion, frozenset()) for ion in ions)))
        for ion in ions:
            latest_gates_by_ion[ion] = frozenset({gate_id})
    return tuple(predecessors)


def _qasm_barrier_targets(line: str) -> frozenset[int] | None:
    """Return the qubits of a QASM barrier.

    Returns:
        The barrier qubits, or ``None`` for a barrier on the whole register.

    Raises:
        ValueError: If an operand is not the register or one of its qubits.
    """
    operands = line.removeprefix("barrier").removesuffix(";").strip()
    if operands in {"", "q"}:
        return None
    targets: set[int] = set()
    for operand in operands.split(","):
        match = _QUBIT_OPERAND_PATTERN.fullmatch(operand.strip())
        if match is None:
            msg = f"Unsupported QASM syntax: {line}"
            raise ValueError(msg)
        targets.add(int(match.group(1)))
    return frozenset(targets)


def _safe_eval(expression: str) -> float:
    def evaluate(current: ast.AST) -> float:
        if (
            isinstance(current, ast.Constant)
            and isinstance(current.value, int | float)
            and not isinstance(current.value, bool)
        ):
            return float(current.value)
        if isinstance(current, ast.Name) and current.id == "pi":
            return math.pi
        if isinstance(current, ast.UnaryOp) and isinstance(current.op, ast.UAdd | ast.USub):
            value = evaluate(current.operand)
            return value if isinstance(current.op, ast.UAdd) else -value
        if isinstance(current, ast.BinOp) and isinstance(
            current.op,
            ast.Add | ast.Sub | ast.Mult | ast.Div,
        ):
            left = evaluate(current.left)
            right = evaluate(current.right)
            if isinstance(current.op, ast.Add):
                return left + right
            if isinstance(current.op, ast.Sub):
                return left - right
            if isinstance(current.op, ast.Mult):
                return left * right
            return left / right
        msg = f"Unsupported parameter expression: {expression}"
        raise ValueError(msg)

    try:
        return evaluate(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ZeroDivisionError, OverflowError) as error:
        msg = f"Invalid parameter expression: {expression}"
        raise ValueError(msg) from error


def _numeric_parameter(value: object, operation_name: str) -> float:
    parameters = getattr(value, "parameters", frozenset())
    if parameters:
        msg = f"operation {operation_name!r} contains an unbound parameter"
        raise ValueError(msg)
    try:
        return float(cast("str | SupportsFloat | SupportsIndex", value))
    except (TypeError, ValueError) as error:
        msg = f"operation {operation_name!r} parameter must be numeric"
        raise ValueError(msg) from error


def _validate_gate_records(records: list[GateRecord], num_qubits: int) -> None:
    """Validate qubit references in parsed gate records.

    Args:
        records: Parsed gate records to validate.
        num_qubits: Number of qubits declared by the circuit.

    Raises:
        ValueError: If a gate has invalid qubit operands.
    """
    for gate_type, _, ions in records:
        gate_name = gate_type.circuit_name
        if any(ion >= num_qubits for ion in ions):
            msg = f"gate {gate_name!r} references a qubit outside the quantum register"
            raise ValueError(msg)
        if len(ions) == 2 and ions[0] == ions[1]:
            msg = f"two-qubit gate {gate_name!r} requires two distinct qubits"
            raise ValueError(msg)


def _validate_barriers(barriers: Barriers, num_qubits: int) -> None:
    """Validate qubit references in parsed barriers.

    Args:
        barriers: Parsed barriers to validate.
        num_qubits: Number of qubits declared by the circuit.

    Raises:
        ValueError: If a barrier references a qubit outside the quantum register.
    """
    for gate_barriers in barriers.values():
        for targets in gate_barriers:
            if any(ion >= num_qubits for ion in targets):
                msg = "barrier references a qubit outside the quantum register"
                raise ValueError(msg)


def _significant_qasm_lines(qasm: str) -> list[str]:
    return [line for raw_line in qasm.splitlines() if (line := raw_line.split("//", maxsplit=1)[0].strip())]


__all__ = [
    "CircuitInput",
    "parse_circuit",
]
