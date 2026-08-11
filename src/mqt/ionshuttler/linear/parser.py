# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Read supported circuits and prepare their gates for scheduling."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import SupportsFloat, SupportsIndex, cast

from qiskit import QuantumCircuit

from mqt.ionshuttler.linear.actions import BUILTIN_ACTION_TYPES, GateAction
from mqt.ionshuttler.linear.config import GATE_NAMES, GateTiming

DependencyMap = dict[int, frozenset[int]]
ParsedCircuit = tuple[int, list[GateAction], DependencyMap | None, DependencyMap | None]
GateDurationMap = Mapping[str, int]
GateRecord = tuple[type[GateAction], tuple[float, ...], tuple[int, ...]]
DependencyBreaks = dict[int, frozenset[int]]
CircuitInput = QuantumCircuit | str | Path

_DEFAULT_GATE_TIMING = GateTiming()
_BUILTIN_GATE_TYPES = tuple(
    action_type
    for action_type in BUILTIN_ACTION_TYPES
    if issubclass(action_type, GateAction) and action_type.circuit_name is not None
)
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
    use_dependencies: bool = True,
    gate_timing: GateTiming | None = None,
    gate_durations: GateDurationMap | None = None,
    gate_types: Sequence[type[GateAction]] | None = None,
) -> ParsedCircuit:
    """Prepare a Qiskit circuit or QASM circuit for scheduling.

    Strings are interpreted as QASM text, while :class:`pathlib.Path` inputs
    are read as UTF-8 files. When dependency analysis is disabled, the final
    two return values are ``None``.

    Args:
        circuit: Circuit to prepare.
        use_dependencies: Whether to identify which gates must precede others.
        gate_timing: Hardware durations and virtual gate implementations.
        gate_durations: Optional duration overrides. This compact form cannot be
            combined with ``gate_timing``.
        gate_types: Gate classes understood by the circuit frontend.

    Returns:
        The qubit count, gates ready to schedule, and optional maps of which
        gates directly precede and follow one another.

    Raises:
        TypeError: If ``circuit`` has an unsupported type.
    """
    timing = _resolve_gate_timing(gate_timing, gate_durations)
    registry = _gate_type_registry(gate_types)
    if isinstance(circuit, QuantumCircuit):
        num_qubits, records, dependency_breaks = _records_from_quantum_circuit(circuit, registry)
    elif isinstance(circuit, Path):
        num_qubits, records, dependency_breaks = _records_from_qasm(
            circuit.read_text(encoding="utf-8"),
            registry,
        )
    elif isinstance(circuit, str):
        num_qubits, records, dependency_breaks = _records_from_qasm(circuit, registry)
    else:
        msg = "circuit must be a QuantumCircuit, QASM string, or pathlib.Path"
        raise TypeError(msg)
    return _build_parsed_circuit(
        num_qubits,
        records,
        dependency_breaks,
        timing,
        use_dependencies=use_dependencies,
    )


def parse_quantum_circuit(
    circuit: QuantumCircuit,
    *,
    use_dependencies: bool = True,
    gate_timing: GateTiming | None = None,
    gate_types: Sequence[type[GateAction]] | None = None,
) -> ParsedCircuit:
    """Prepare a Qiskit circuit for scheduling.

    Returns:
        The qubit count, gates ready to schedule, and maps of which gates
        directly precede and follow one another.
    """
    return parse_circuit(
        circuit,
        use_dependencies=use_dependencies,
        gate_timing=gate_timing,
        gate_types=gate_types,
    )


def parse_qasm_to_gate_sequence(
    qasm: str,
    *,
    gate_timing: GateTiming | None = None,
    gate_durations: GateDurationMap | None = None,
    gate_types: Sequence[type[GateAction]] | None = None,
) -> tuple[int, list[GateAction]]:
    """Read the supported gates from QASM text.

    Returns:
        The qubit count and ordered gate actions.
    """
    num_qubits, records, _ = _records_from_qasm(qasm, _gate_type_registry(gate_types))
    timing = _resolve_gate_timing(gate_timing, gate_durations)
    return num_qubits, [_lower_gate(record, timing) for record in records]


def compute_gate_dependencies_from_qasm(
    qasm: str,
    *,
    gate_types: Sequence[type[GateAction]] | None = None,
) -> tuple[DependencyMap, DependencyMap]:
    """Return the gates that directly precede and follow each QASM gate."""
    _, records, dependency_breaks = _records_from_qasm(qasm, _gate_type_registry(gate_types))
    return _compute_dependencies(records, dependency_breaks)


def parse_qasm_to_gate_sequence_with_dependencies(
    qasm: str,
    *,
    gate_timing: GateTiming | None = None,
    gate_durations: GateDurationMap | None = None,
    gate_types: Sequence[type[GateAction]] | None = None,
) -> tuple[int, list[GateAction], DependencyMap, DependencyMap]:
    """Read QASM gates together with their direct dependencies.

    Returns:
        The qubit count, gates in circuit order, and maps of which gates
        directly precede and follow one another.
    """
    num_qubits, records, dependency_breaks = _records_from_qasm(qasm, _gate_type_registry(gate_types))
    timing = _resolve_gate_timing(gate_timing, gate_durations)
    predecessors, successors = _compute_dependencies(records, dependency_breaks)
    return num_qubits, [_lower_gate(record, timing) for record in records], predecessors, successors


def parse_qasm(
    qasm: str,
    *,
    use_dependencies: bool = True,
    gate_timing: GateTiming | None = None,
    gate_durations: GateDurationMap | None = None,
    gate_types: Sequence[type[GateAction]] | None = None,
) -> ParsedCircuit:
    """Prepare supported QASM text for scheduling.

    Returns:
        The qubit count, gates ready to schedule, and optional maps of which
        gates directly precede and follow one another.
    """
    return parse_circuit(
        qasm,
        use_dependencies=use_dependencies,
        gate_timing=gate_timing,
        gate_durations=gate_durations,
        gate_types=gate_types,
    )


def parse_qasm_file(
    path: Path,
    *,
    use_dependencies: bool = True,
    gate_timing: GateTiming | None = None,
    gate_durations: GateDurationMap | None = None,
    gate_types: Sequence[type[GateAction]] | None = None,
) -> ParsedCircuit:
    """Prepare a UTF-8 QASM file for scheduling.

    Returns:
        The qubit count, gates ready to schedule, and optional maps of which
        gates directly precede and follow one another.
    """
    return parse_circuit(
        path,
        use_dependencies=use_dependencies,
        gate_timing=gate_timing,
        gate_durations=gate_durations,
        gate_types=gate_types,
    )


def parse_qasm_file_with_dependencies(
    path: Path,
    *,
    gate_timing: GateTiming | None = None,
    gate_durations: GateDurationMap | None = None,
    gate_types: Sequence[type[GateAction]] | None = None,
) -> tuple[int, list[GateAction], DependencyMap, DependencyMap]:
    """Read a UTF-8 QASM file together with its direct dependencies.

    Returns:
        The qubit count, gates in circuit order, and maps of which gates
        directly precede and follow one another.
    """
    parsed = parse_qasm_file(
        path,
        use_dependencies=True,
        gate_timing=gate_timing,
        gate_durations=gate_durations,
        gate_types=gate_types,
    )
    num_qubits, gates, predecessors, successors = parsed
    return (
        num_qubits,
        gates,
        cast("DependencyMap", predecessors),
        cast("DependencyMap", successors),
    )


def _gate_type_registry(
    gate_types: Sequence[type[GateAction]] | None,
) -> dict[str, type[GateAction]]:
    registry: dict[str, type[GateAction]] = {}
    for gate_type in _BUILTIN_GATE_TYPES if gate_types is None else gate_types:
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
) -> tuple[int, list[GateRecord], DependencyBreaks]:
    num_qubits: int | None = None
    records: list[GateRecord] = []
    dependency_break_targets: dict[int, set[int] | None] = {}
    header_seen = False
    lines = _significant_qasm_lines(qasm)

    for index, line in enumerate(lines):
        if _CLASSICAL_REGISTER_PATTERN.match(line):
            continue
        if _BARRIER_PATTERN.match(line):
            targets = _qasm_barrier_targets(line)
            gate_id = len(records)
            previous_targets = dependency_break_targets.get(gate_id, set())
            dependency_break_targets[gate_id] = (
                None if targets is None or previous_targets is None else previous_targets.union(targets)
            )
            continue
        if _MEASURE_PATTERN.match(line):
            if all(_is_trailing_nonunitary_line(trailing_line) for trailing_line in lines[index + 1 :]):
                continue
            msg = f"Unsupported QASM syntax: {line}"
            raise ValueError(msg)
        if line in {"OPENQASM 2.0;", "OPENQASM 3;", "OPENQASM 3.0;"}:
            header_seen = True
            continue
        if line in {'include "qelib1.inc";', 'include "stdgates.inc";'}:
            continue
        if match := _QASM2_QREG_PATTERN.match(line):
            num_qubits = int(match.group(1))
            continue
        if match := _QASM3_QREG_PATTERN.match(line):
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
    dependency_breaks = {
        gate_id: frozenset(range(num_qubits) if targets is None else targets)
        for gate_id, targets in dependency_break_targets.items()
    }
    return num_qubits, records, dependency_breaks


def _records_from_quantum_circuit(
    circuit: QuantumCircuit,
    gate_types: Mapping[str, type[GateAction]],
) -> tuple[int, list[GateRecord], DependencyBreaks]:
    records: list[GateRecord] = []
    dependency_breaks: dict[int, set[int]] = {}
    measurement_seen = False
    for instruction in circuit.data:
        operation = instruction.operation
        gate_name = operation.name.lower()
        if gate_name == "barrier":
            gate_id = len(records)
            dependency_breaks.setdefault(gate_id, set()).update(
                circuit.find_bit(qubit).index for qubit in instruction.qubits
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
        {gate_id: frozenset(ions) for gate_id, ions in dependency_breaks.items()},
    )


def _build_parsed_circuit(
    num_qubits: int,
    records: list[GateRecord],
    dependency_breaks: DependencyBreaks,
    timing: GateTiming,
    *,
    use_dependencies: bool,
) -> ParsedCircuit:
    gates = [_lower_gate(record, timing) for record in records]
    if not use_dependencies:
        return num_qubits, gates, None, None
    predecessors, successors = _compute_dependencies(records, dependency_breaks)
    return num_qubits, gates, predecessors, successors


def _lower_gate(record: GateRecord, timing: GateTiming) -> GateAction:
    gate_type, parameters, ions = record
    return gate_type.from_instruction(ions, parameters, timing)


def _compute_dependencies(
    records: list[GateRecord],
    dependency_breaks: DependencyBreaks,
) -> tuple[DependencyMap, DependencyMap]:
    last_gate_by_ion: dict[int, int] = {}
    predecessors: DependencyMap = {}
    mutable_successors: dict[int, set[int]] = {gate_id: set() for gate_id in range(len(records))}
    for gate_id, (_, _, ions) in enumerate(records):
        for ion in dependency_breaks.get(gate_id, frozenset()):
            last_gate_by_ion.pop(ion, None)
        gate_predecessors = frozenset(last_gate_by_ion[ion] for ion in ions if ion in last_gate_by_ion)
        predecessors[gate_id] = gate_predecessors
        for predecessor in gate_predecessors:
            mutable_successors[predecessor].add(gate_id)
        for ion in ions:
            last_gate_by_ion[ion] = gate_id
    successors = {gate_id: frozenset(gate_successors) for gate_id, gate_successors in mutable_successors.items()}
    return predecessors, successors


def _qasm_barrier_targets(line: str) -> set[int] | None:
    operands = line.removeprefix("barrier").removesuffix(";").strip()
    if operands == "q":
        return None
    targets: set[int] = set()
    for operand in operands.split(","):
        match = re.fullmatch(r"q\[(\d+)\]", operand.strip())
        if match is None:
            return None
        targets.add(int(match.group(1)))
    return targets


def _resolve_gate_timing(
    gate_timing: GateTiming | None,
    gate_durations: GateDurationMap | None,
) -> GateTiming:
    if gate_timing is not None and gate_durations is not None:
        msg = "gate_timing and gate_durations cannot both be supplied"
        raise ValueError(msg)
    if gate_timing is not None:
        return gate_timing
    if gate_durations is None:
        return _DEFAULT_GATE_TIMING
    normalized: dict[str, int] = {}
    for gate_name, duration in gate_durations.items():
        normalized_name = gate_name.lower()
        if normalized_name not in GATE_NAMES:
            msg = f"unsupported gate name {gate_name!r}"
            raise ValueError(msg)
        normalized[normalized_name] = duration
    durations = _DEFAULT_GATE_TIMING.gate_durations
    durations.update(normalized)
    return GateTiming(
        rx=durations["rx"],
        ry=durations["ry"],
        rz=durations["rz"],
        rxx=durations["rxx"],
        ryy=durations["ryy"],
        rzz=durations["rzz"],
    )


def _safe_eval(expression: str) -> float:
    node = ast.parse(expression, mode="eval")

    def evaluate(current: ast.AST) -> float:
        if isinstance(current, ast.Expression):
            return evaluate(current.body)
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

    return evaluate(node)


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
    if num_qubits < 1:
        msg = "quantum register must contain at least one qubit"
        raise ValueError(msg)
    for gate_type, _, ions in records:
        gate_name = gate_type.circuit_name or gate_type.__name__
        if any(ion < 0 or ion >= num_qubits for ion in ions):
            msg = f"gate {gate_name!r} references a qubit outside the quantum register"
            raise ValueError(msg)
        if len(ions) == 2 and ions[0] == ions[1]:
            msg = f"two-qubit gate {gate_name!r} requires two distinct qubits"
            raise ValueError(msg)


def _significant_qasm_lines(qasm: str) -> list[str]:
    return [line for raw_line in qasm.splitlines() if (line := raw_line.split("//", maxsplit=1)[0].strip())]


def _is_trailing_nonunitary_line(line: str) -> bool:
    return bool(_CLASSICAL_REGISTER_PATTERN.match(line) or _BARRIER_PATTERN.match(line) or _MEASURE_PATTERN.match(line))


__all__ = [
    "CircuitInput",
    "DependencyMap",
    "GateDurationMap",
    "ParsedCircuit",
    "compute_gate_dependencies_from_qasm",
    "parse_circuit",
    "parse_qasm",
    "parse_qasm_file",
    "parse_qasm_file_with_dependencies",
    "parse_qasm_to_gate_sequence",
    "parse_qasm_to_gate_sequence_with_dependencies",
    "parse_quantum_circuit",
]
