# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Read dynamic programs from MQT Core's QCO dialect."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.core.gates import GateAction, Measure, Reset, Rx, Rxx, Ry, Ryy, Rz, Rzz, TwoQubitGate

from .program import Branch, Condition, Loop, Program

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .program import Step

_MAX_CONDITION_OUTCOMES = 16

_GATE_TYPES: dict[str, type[GateAction]] = {
    gate_type.circuit_name: gate_type for gate_type in (Rx, Ry, Rz, Rxx, Ryy, Rzz) if gate_type.circuit_name
}
_OPERATION = re.compile(r"^(?:(?P<results>%[^=]+?)\s*=\s*)?(?P<name>[a-z_]+\.[a-z_]+)(?P<rest>.*)$")
_VALUE = re.compile(r"%[\w\-]+(?:#\d+)?")
_GATE_OPERANDS = re.compile(r"^(?:\((?P<parameters>[^)]*)\))?\s*(?P<operands>[^:]*):")
_REGISTER_TYPE = re.compile(r"!cbit\.reg<(\d+)>")
_INTEGER_TYPE = re.compile(r"\bi(\d+)\b")
_ARGUMENT_BINDING = re.compile(r"(%[\w\-]+)\s*=\s*(%[\w\-]+(?:#\d+)?)")
_FOR_BOUNDS = re.compile(r"^\s*(%[\w\-]+)\s*=\s*(%[\w\-]+)\s+to\s+(%[\w\-]+)\s+step\s+(%[\w\-]+)")
_BLOCK_ARGUMENT = re.compile(r"(%[\w\-]+)\s*:")


@dataclass(frozen=True)
class _Constant:
    value: int


@dataclass(frozen=True)
class _MeasuredBit:
    outcome: int


@dataclass(frozen=True)
class _Unset:
    pass


@dataclass(frozen=True)
class _ChangesPerIteration:
    """A bit whose value depends on the loop iteration that wrote it."""


@dataclass(frozen=True)
class _Operation:
    name: str
    operands: tuple[_Expression, ...]
    width: int


@dataclass(frozen=True)
class _Choice:
    condition: _Expression
    if_true: _Expression
    if_false: _Expression


_Expression: TypeAlias = _Constant | _MeasuredBit | _Unset | _ChangesPerIteration | _Operation | _Choice
_Record: TypeAlias = "tuple[type[GateAction], tuple[float, ...], tuple[int, ...], int | None]"


@dataclass
class _Scope:
    """Values visible at one point of the program text."""

    ions: dict[str, int]
    numbers: dict[str, float]
    expressions: dict[str, _Expression]
    registers: dict[str, list[_Expression]]
    loop_variables: set[str] = field(default_factory=set)

    def child(self) -> _Scope:
        return _Scope(
            dict(self.ions),
            dict(self.numbers),
            dict(self.expressions),
            {name: list(bits) for name, bits in self.registers.items()},
            set(self.loop_variables),
        )


@dataclass
class _DraftBranch:
    condition: Condition
    then_steps: list[_DraftStep]
    else_steps: list[_DraftStep]


@dataclass
class _DraftLoop:
    steps: list[_DraftStep]
    count: int | None
    entry: Condition | None
    repeat: Condition | None


_DraftStep: TypeAlias = "list[_Record] | _DraftBranch | _DraftLoop"


class _Reader:
    """Read the region structure of one QCO function body."""

    def __init__(self, lines: Sequence[str]) -> None:
        self.lines = lines
        self.next_outcome = 0
        self.max_ion = -1

    def error(self, index: int, message: str) -> ValueError:
        return ValueError(f"line {index + 1}: {message}: {self.lines[index].strip()}")

    def read_region(self, index: int, scope: _Scope) -> tuple[list[_DraftStep], int, list[str]]:
        """Read operations until the closing brace of the current region.

        Returns:
            The steps, the index of the closing line, and the yielded values.

        Raises:
            ValueError: If the region contains unsupported syntax or has no closing brace.
        """
        steps: list[_DraftStep] = []
        records: list[_Record] = []
        yielded: list[str] = []
        while index < len(self.lines):
            line = self.lines[index].strip()
            if not line:
                index += 1
                continue
            if line.startswith("}"):
                if records:
                    steps.append(records)
                return steps, index, yielded
            if line.startswith(("return", "func.return")):
                index += 1
                continue
            match = _OPERATION.match(line)
            if match is None:
                raise self.error(index, "unsupported syntax")
            results = _VALUE.findall(match.group("results") or "")
            name = match.group("name")
            rest = match.group("rest")
            if name == "qco.if":
                if records:
                    steps.append(records)
                    records = []
                branch, index = self.read_branch(index, results, rest, scope)
                steps.append(branch)
                continue
            if name == "scf.if":
                index = self.read_classical_branch(index, results, rest, scope)
                continue
            if name in {"scf.for", "scf.while"}:
                if records:
                    steps.append(records)
                    records = []
                read = self.read_counted_loop if name == "scf.for" else self.read_conditional_loop
                loop, index = read(index, results, rest, scope)
                steps.append(loop)
                continue
            if name in {"qco.yield", "scf.yield", "scf.condition"}:
                # A loop condition yields its decision first, then the values it forwards.
                yielded = _VALUE.findall(rest.split(":", 1)[0])
            else:
                record = self.read_operation(index, name, results, rest, scope)
                if record is not None:
                    records.append(record)
            index += 1
        msg = "missing closing brace at the end of the program"
        raise ValueError(msg)

    def read_operation(self, index: int, name: str, results: list[str], rest: str, scope: _Scope) -> _Record | None:
        """Read one operation that does not open a region.

        Returns:
            A gate record, or ``None`` for an operation without a schedule entry.
        """
        dialect, operation = name.split(".", 1)
        if dialect == "qco":
            return self.read_quantum(index, operation, results, rest, scope)
        if dialect == "cbit":
            self.read_register(index, operation, results, rest, scope)
            return None
        if dialect == "arith":
            self.read_arithmetic(index, operation, results, rest, scope)
            return None
        raise self.error(index, f"unsupported operation {name!r}")

    def read_quantum(self, index: int, operation: str, results: list[str], rest: str, scope: _Scope) -> _Record | None:
        if operation == "static":
            ion = int(rest.split(":", 1)[0])
            self.max_ion = max(self.max_ion, ion)
            scope.ions[results[0]] = ion
            return None
        if operation in {"sink", "gphase"}:
            return None
        match = _GATE_OPERANDS.match(rest.strip())
        if match is None:
            raise self.error(index, "malformed quantum operation")
        operands = _VALUE.findall(match.group("operands"))
        try:
            ions = tuple(scope.ions[operand] for operand in operands)
        except KeyError as error:
            raise self.error(index, f"unknown qubit value {error.args[0]}") from error
        if operation == "measure":
            outcome = self.next_outcome
            self.next_outcome += 1
            scope.ions[results[0]] = ions[0]
            scope.expressions[results[1]] = _MeasuredBit(outcome)
            return (Measure, (), ions, outcome)
        if operation == "reset":
            scope.ions[results[0]] = ions[0]
            return (Reset, (), ions, None)
        gate_type = _GATE_TYPES.get(operation)
        if gate_type is None:
            raise self.error(index, f"operation 'qco.{operation}' is not a native gate; synthesize the program first")
        parameters = tuple(
            self.number(index, value, scope) for value in _VALUE.findall(match.group("parameters") or "")
        )
        for result, ion in zip(results, ions, strict=True):
            scope.ions[result] = ion
        return (gate_type, parameters, ions, None)

    def read_register(self, index: int, operation: str, results: list[str], rest: str, scope: _Scope) -> None:
        values = _VALUE.findall(rest.split(":", 1)[0])
        if operation == "alloc":
            width_match = _REGISTER_TYPE.search(rest)
            if width_match is None:
                raise self.error(index, "missing register width")
            initial: _Expression = _Constant(0) if "init<zero>" in rest else _Unset()
            scope.registers[results[0]] = [initial] * int(width_match.group(1))
        elif operation == "store":
            bit, register, position = values
            bits = self.register(index, register, scope)
            if position in scope.loop_variables:
                bits[:] = [_ChangesPerIteration()] * len(bits)
            else:
                bits[int(self.number(index, position, scope))] = self.value(index, bit, scope)
        elif operation == "load":
            register, position = values
            bits = self.register(index, register, scope)
            scope.expressions[results[0]] = (
                _ChangesPerIteration()
                if position in scope.loop_variables
                else bits[int(self.number(index, position, scope))]
            )
        elif operation == "write":
            value, register = values
            bits = self.register(index, register, scope)
            expression = self.value(index, value, scope)
            for position in range(len(bits)):
                bits[position] = _bit_of(expression, position)
        elif operation == "read":
            bits = self.register(index, values[0], scope)
            scope.expressions[results[0]] = _Operation("concat", tuple(bits), len(bits))
        else:
            raise self.error(index, f"unsupported operation 'cbit.{operation}'")

    def read_arithmetic(self, index: int, operation: str, results: list[str], rest: str, scope: _Scope) -> None:
        if operation == "constant":
            text, _, value_type = rest.partition(":")
            text = text.strip()
            if text in {"true", "false"}:
                scope.expressions[results[0]] = _Constant(int(text == "true"))
                return
            if value_type.strip() in {"f64", "f32"}:
                scope.numbers[results[0]] = float(text)
                return
            number = int(text)
            scope.numbers[results[0]] = number
            width = _integer_width(value_type)
            if width is not None:
                scope.expressions[results[0]] = _Constant(number % (1 << width))
            return
        width = _integer_width(rest.rsplit(":", 1)[-1]) or 1
        if operation == "cmpi":
            predicate, _, operand_text = rest.strip().partition(",")
            if predicate.strip() not in {"eq", "ne"}:
                raise self.error(index, f"unsupported comparison {predicate.strip()!r}")
            left, right = (self.value(index, value, scope) for value in _VALUE.findall(operand_text.split(":", 1)[0]))
            scope.expressions[results[0]] = _Operation(predicate.strip(), (left, right), width)
            return
        if operation in {"andi", "ori", "xori"}:
            left, right = (self.value(index, value, scope) for value in _VALUE.findall(rest.split(":", 1)[0]))
            scope.expressions[results[0]] = _Operation(operation, (left, right), width)
            return
        raise self.error(index, f"unsupported operation 'arith.{operation}'")

    def read_branch(self, index: int, results: list[str], rest: str, scope: _Scope) -> tuple[_DraftBranch, int]:
        """Read one ``qco.if`` with both regions.

        Returns:
            The branch and the index after its closing line.
        """
        expression = self.value(index, _VALUE.findall(rest)[0], scope)
        condition = self.condition(index, expression)
        then_scope = self.region_scope(index, rest, scope)
        then_steps, close, then_yield = self.read_region(index + 1, then_scope)
        closing = self.lines[close].strip()
        if not closing.startswith("} else"):
            raise self.error(close, "expected the else region of qco.if")
        else_scope = self.region_scope(close, closing, scope)
        else_steps, end, else_yield = self.read_region(close + 1, else_scope)
        then_ions = [then_scope.ions[value] for value in then_yield]
        else_ions = [else_scope.ions[value] for value in else_yield]
        if then_ions != else_ions:
            raise self.error(index, "both branches must yield the same ions in the same order")
        for name, ion in zip(_result_names(results, len(then_ions)), then_ions, strict=True):
            scope.ions[name] = ion
        _merge_registers(scope, expression, then_scope, else_scope)
        return _DraftBranch(condition, then_steps, else_steps), end + 1

    def read_classical_branch(self, index: int, results: list[str], rest: str, scope: _Scope) -> int:
        """Read one ``scf.if`` that only computes classical values.

        Returns:
            The index after its closing line.
        """
        expression = self.value(index, _VALUE.findall(rest)[0], scope)
        then_scope = scope.child()
        then_steps, close, then_yield = self.read_region(index + 1, then_scope)
        else_scope = scope.child()
        else_steps: list[_DraftStep] = []
        else_yield: list[str] = []
        end = close
        if self.lines[close].strip().startswith("} else"):
            else_steps, end, else_yield = self.read_region(close + 1, else_scope)
        if then_steps or else_steps:
            raise self.error(index, "a classical scf.if must not contain quantum operations")
        names = _result_names(results, len(then_yield))
        for name, then_value, else_value in zip(names, then_yield, else_yield, strict=True):
            scope.expressions[name] = _Choice(
                expression, self.value(index, then_value, then_scope), self.value(index, else_value, else_scope)
            )
        _merge_registers(scope, expression, then_scope, else_scope)
        return end + 1

    def read_counted_loop(self, index: int, results: list[str], rest: str, scope: _Scope) -> tuple[_DraftLoop, int]:
        """Read one ``scf.for`` with constant bounds.

        Returns:
            The loop and the index after its closing line.
        """
        bounds = _FOR_BOUNDS.match(rest)
        if bounds is None:
            raise self.error(index, "malformed scf.for")
        variable, lower, upper, step = bounds.groups()
        count = len(range(*(int(self.number(index, name, scope)) for name in (lower, upper, step))))
        if count < 1:
            raise self.error(index, "a loop must run at least once")
        written = self.written_bits(index, scope)
        arguments = rest.split("iter_args", 1)[1] if "iter_args" in rest else ""
        body_scope = self.region_scope(index, f"args{arguments}" if arguments else "", scope)
        body_scope.loop_variables.add(variable)
        _forget_bits(body_scope, written)
        received = [body_scope.ions[name] for name, _value in _ARGUMENT_BINDING.findall(arguments.split("->", 1)[0])]
        steps, close, yielded = self.read_region(index + 1, body_scope)
        self.require_same_ions(index, received, [body_scope.ions[value] for value in yielded])
        for name, ion in zip(_result_names(results, len(received)), received, strict=True):
            scope.ions[name] = ion
        _forget_bits(scope, written)
        return _DraftLoop(steps, count, None, None), close + 1

    def read_conditional_loop(self, index: int, results: list[str], rest: str, scope: _Scope) -> tuple[_DraftLoop, int]:
        """Read one ``scf.while`` whose before region only computes its condition.

        Returns:
            The loop and the index after its closing line.
        """
        header = rest.split(":", 1)[0]
        written = self.written_bits(index, scope)
        entry, close, received = self.read_loop_condition(index, header, scope)
        if not self.lines[close].strip().startswith("} do"):
            raise self.error(close, "expected the do region of scf.while")
        body_scope = scope.child()
        _forget_bits(body_scope, written)
        label = self.lines[close + 1].strip()
        names = _BLOCK_ARGUMENT.findall(label) if label.startswith("^") else []
        for name, ion in zip(names, received, strict=True):
            body_scope.ions[name] = ion
        steps, end, yielded = self.read_region(close + 2 if label.startswith("^") else close + 1, body_scope)
        self.require_same_ions(index, received, [body_scope.ions[value] for value in yielded])
        repeat, _close, _received = self.read_loop_condition(index, header, body_scope)
        for name, ion in zip(_result_names(results, len(received)), received, strict=True):
            scope.ions[name] = ion
        _forget_bits(scope, written)
        return _DraftLoop(steps, None, entry, repeat), end + 1

    def read_loop_condition(self, index: int, header: str, scope: _Scope) -> tuple[Condition, int, list[int]]:
        """Read the before region of ``scf.while`` with the register values of ``scope``.

        Returns:
            The condition, the index of the closing line, and the ions of the forwarded qubits.
        """
        region = self.region_scope(index, f"args{header}", scope)
        steps, close, yielded = self.read_region(index + 1, region)
        if steps or not yielded:
            raise self.error(index, "the condition region of scf.while must only compute its condition")
        decision, *forwarded = yielded
        ions = [region.ions[value] for value in forwarded]
        return self.condition(index, self.value(index, decision, region)), close, ions

    def written_bits(self, index: int, scope: _Scope) -> dict[str, set[int] | None]:
        """Return the register bits that the region opened at ``index`` writes; ``None`` means every bit."""
        writes: dict[str, set[int] | None] = {}
        depth = 0
        for line in self.lines[index:]:
            stripped = line.strip()
            depth += stripped.count("{") - stripped.count("}")
            match = _OPERATION.match(stripped)
            if match is not None and match.group("name") in {"cbit.store", "cbit.write"}:
                values = _VALUE.findall(match.group("rest").split(":", 1)[0])
                register = values[1]
                position = scope.numbers.get(values[2]) if match.group("name") == "cbit.store" else None
                known = writes.get(register, set())
                if position is None or known is None:
                    writes[register] = None
                else:
                    known.add(int(position))
                    writes[register] = known
            if depth <= 0:
                return writes
        return writes

    def require_same_ions(self, index: int, received: list[int], yielded: list[int]) -> None:
        if received != yielded:
            raise self.error(index, "a loop body must yield the ions it receives in the same order")

    def region_scope(self, index: int, header: str, scope: _Scope) -> _Scope:
        region = scope.child()
        arguments = header.split("args", 1)[1] if "args" in header else ""
        for name, value in _ARGUMENT_BINDING.findall(arguments.split("->", 1)[0]):
            try:
                region.ions[name] = scope.ions[value]
            except KeyError as error:
                raise self.error(index, f"unknown qubit value {value}") from error
        return region

    def condition(self, index: int, expression: _Expression) -> Condition:
        outcomes = sorted(_outcomes_in(expression))
        if len(outcomes) > _MAX_CONDITION_OUTCOMES:
            raise self.error(index, f"a condition may read at most {_MAX_CONDITION_OUTCOMES} measurement outcomes")
        true_for: set[tuple[int, ...]] = set()
        for values in product((0, 1), repeat=len(outcomes)):
            try:
                if _evaluate(expression, dict(zip(outcomes, values, strict=True))):
                    true_for.add(values)
            except ValueError as error:
                raise self.error(index, str(error)) from error
        return Condition(tuple(outcomes), frozenset(true_for))

    def register(self, index: int, name: str, scope: _Scope) -> list[_Expression]:
        try:
            return scope.registers[name]
        except KeyError as error:
            raise self.error(index, f"unknown classical register {name}") from error

    def value(self, index: int, name: str, scope: _Scope) -> _Expression:
        try:
            return scope.expressions[name]
        except KeyError as error:
            raise self.error(index, f"unknown classical value {name}") from error

    def number(self, index: int, name: str, scope: _Scope) -> float:
        try:
            return scope.numbers[name]
        except KeyError as error:
            raise self.error(index, f"value {name} is not a constant number") from error


def parse_program(qco: str) -> Program:
    """Read a dynamic program from QCO text after native-gate synthesis.

    The text must come from MQT Core's ``QCOProgram.ir`` after
    ``synthesize_for_target`` with an all-to-all target. Each ``qco.static``
    qubit becomes the ion with the same number. Supported operations are the
    native rotations ``rx``, ``ry``, ``rz``, ``rxx``, ``ryy``, and ``rzz``,
    ``measure``, ``reset``, measurement-conditioned ``qco.if``, ``scf.for``
    with constant bounds, ``scf.while`` whose condition region only computes
    its condition, classical registers, and integer ``and``, ``or``, ``xor``,
    and equality comparisons. Every measurement gets an outcome number in
    program order. A condition may not read a register bit that a loop wrote
    in another iteration.

    Args:
        qco: Textual QCO module with one entry function.

    Returns:
        The program.

    Raises:
        ValueError: If the text contains an unsupported operation or structure.
    """
    lines = qco.splitlines()
    start = next((index for index, line in enumerate(lines) if line.strip().startswith("func.func")), None)
    if start is None:
        msg = "QCO text contains no function"
        raise ValueError(msg)
    reader = _Reader(lines)
    steps, close, _yielded = reader.read_region(start + 1, _Scope({}, {}, {}, {}))
    if any(line.strip().startswith("func.func") for line in lines[close + 1 :]):
        msg = "QCO text must contain exactly one function"
        raise ValueError(msg)
    if reader.max_ion < 0:
        msg = "QCO program uses no qubits"
        raise ValueError(msg)
    num_ions = reader.max_ion + 1
    return Program(num_ions=num_ions, steps=_build_steps(steps, num_ions))


def program_from_qasm(
    source: str | Path,
    *,
    gates: Iterable[type[GateAction]] = (Rx, Ry, Rz, Rxx),
) -> Program:
    """Read an OpenQASM 3 program with MQT Core and synthesize native gates.

    MQT Core imports the program, converts it to QCO, and synthesizes the
    given native gates together with measurement and reset. Requires the
    ``dynamic`` extra.

    Args:
        source: OpenQASM text, or a path to an OpenQASM file.
        gates: Native gate types that synthesis may use.

    Returns:
        The program.

    Raises:
        ValueError: If a gate type has no circuit name or the qubit count is unknown.
    """
    from mqt.core import mlir  # ruff: ignore[import-outside-top-level] - optional dependency

    if isinstance(source, Path):
        source = source.read_text(encoding="utf-8")
    program = mlir.QCProgram.from_openqasm_str(source).to_qco()
    num_qubits = program.inspect().num_qubits
    if num_qubits is None:
        msg = "MQT Core reports no fixed qubit count for the program"
        raise ValueError(msg)
    capabilities: list[mlir.CompilerTarget.OperationCapability] = []
    for gate_type in gates:
        name = gate_type.circuit_name
        if name is None:
            msg = f"gate type {gate_type.__name__} has no circuit name"
            raise ValueError(msg)
        arity = 2 if issubclass(gate_type, TwoQubitGate) else 1
        capabilities.append(mlir.CompilerTarget.OperationCapability(name, arity, 1))
    capabilities.extend((
        mlir.CompilerTarget.OperationCapability("measure", 1, 0),
        mlir.CompilerTarget.OperationCapability("reset", 1, 0),
    ))
    target = mlir.CompilerTarget(
        num_qubits,
        connectivity=mlir.CompilerTarget.Connectivity.all_to_all(),
        native_operations=mlir.CompilerTarget.NativeOperations(capabilities),
    )
    payload = mlir.PayloadSpecification(
        mlir.PayloadFormat("qco", "1"),
        [
            mlir.ProgramCapability(mlir.ProgramCapability.FORWARD_BRANCHING),
            mlir.ProgramCapability(mlir.ProgramCapability.COUNTED_ITERATION),
            mlir.ProgramCapability(mlir.ProgramCapability.CONDITIONAL_LOOP),
        ],
    )
    program.synthesize_for_target(mlir.TargetEnvironment(target, payload))
    return parse_program(program.ir)


def _build_steps(steps: Sequence[_DraftStep], num_ions: int) -> tuple[Step, ...]:
    built: list[Step] = []
    for step in steps:
        if isinstance(step, _DraftBranch):
            built.append(
                Branch(step.condition, _build_steps(step.then_steps, num_ions), _build_steps(step.else_steps, num_ions))
            )
        elif isinstance(step, _DraftLoop):
            built.append(Loop(_build_steps(step.steps, num_ions), step.count, step.entry, step.repeat))
        else:
            built.append(_build_circuit(step, num_ions))
    return tuple(built)


def _build_circuit(records: Sequence[_Record], num_ions: int) -> Circuit:
    gates: list[GateAction] = []
    predecessors: list[frozenset[int]] = []
    latest_by_ion: dict[int, int] = {}
    for gate_id, (gate_type, parameters, ions, outcome) in enumerate(records):
        if gate_type is Measure:
            gates.append(Measure(ion=ions[0], outcome=outcome, gate_id=gate_id))
        else:
            gates.append(gate_type.from_instruction(ions, parameters, gate_id=gate_id))
        predecessors.append(frozenset(latest_by_ion[ion] for ion in ions if ion in latest_by_ion))
        for ion in ions:
            latest_by_ion[ion] = gate_id
    return Circuit(num_ions=num_ions, gates=tuple(gates), predecessors=tuple(predecessors))


def _forget_bits(scope: _Scope, written: dict[str, set[int] | None]) -> None:
    """Mark register bits that a loop writes as unknown outside their iteration."""
    for register, positions in written.items():
        bits = scope.registers.get(register)
        if bits is None:
            continue
        for position in range(len(bits)) if positions is None else positions:
            bits[position] = _ChangesPerIteration()


def _merge_registers(scope: _Scope, condition: _Expression, then_scope: _Scope, else_scope: _Scope) -> None:
    """Make each register bit after a branch depend on the side that ran."""
    for register, bits in scope.registers.items():
        for position in range(len(bits)):
            then_bit = then_scope.registers[register][position]
            else_bit = else_scope.registers[register][position]
            bits[position] = then_bit if then_bit == else_bit else _Choice(condition, then_bit, else_bit)


def _result_names(results: list[str], count: int) -> list[str]:
    if len(results) == 1 and count != 1:
        base = results[0].split(":", 1)[0]
        return [f"{base}#{position}" for position in range(count)]
    return results


def _integer_width(value_type: str) -> int | None:
    if value_type.strip() == "index":
        return None
    match = _INTEGER_TYPE.search(value_type)
    return None if match is None else int(match.group(1))


def _bit_of(expression: _Expression, position: int) -> _Expression:
    if isinstance(expression, _Constant):
        return _Constant((expression.value >> position) & 1)
    return _Operation("bit", (expression, _Constant(position)), 1)


def _outcomes_in(expression: _Expression) -> set[int]:
    if isinstance(expression, _MeasuredBit):
        return {expression.outcome}
    if isinstance(expression, _Operation):
        return set().union(*(_outcomes_in(operand) for operand in expression.operands))
    if isinstance(expression, _Choice):
        return _outcomes_in(expression.condition) | _outcomes_in(expression.if_true) | _outcomes_in(expression.if_false)
    return set()


def _evaluate(expression: _Expression, values: dict[int, int]) -> int:
    if isinstance(expression, _Constant):
        return expression.value
    if isinstance(expression, _MeasuredBit):
        return values[expression.outcome]
    if isinstance(expression, _Unset):
        msg = "a condition reads a classical bit that is never set"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Program errors use ValueError.
    if isinstance(expression, _ChangesPerIteration):
        msg = "a condition reads a classical bit that a loop writes in another iteration"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Program errors use ValueError.
    if isinstance(expression, _Choice):
        chosen = expression.if_true if _evaluate(expression.condition, values) else expression.if_false
        return _evaluate(chosen, values)
    operands = [_evaluate(operand, values) for operand in expression.operands]
    mask = (1 << expression.width) - 1
    if expression.name == "concat":
        return sum(bit << position for position, bit in enumerate(operands))
    if expression.name == "bit":
        return (operands[0] >> operands[1]) & 1
    if expression.name == "eq":
        return int(operands[0] == operands[1])
    if expression.name == "ne":
        return int(operands[0] != operands[1])
    if expression.name == "andi":
        return operands[0] & operands[1] & mask
    if expression.name == "ori":
        return (operands[0] | operands[1]) & mask
    return (operands[0] ^ operands[1]) & mask


__all__ = ["parse_program", "program_from_qasm"]
