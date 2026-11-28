# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Programs whose operations depend on measurement outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.core.gates import GATE_TYPES, GateAction, Measure
from mqt.ionshuttler.core.schedule import Schedule

from .._json_utils import require_int, require_int_list, require_list, require_mapping, require_str

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

PROGRAM_SCHEMA = "mqt.ionshuttler.program"
PROGRAM_VERSION = 2

# One measured value: the outcome number, followed by the iteration of each
# enclosing loop for an outcome measured inside loops.
OutcomeKey: TypeAlias = "int | tuple[int, ...]"

_OPERATIONS = frozenset({"measured", "always", "never", "not", "all", "any", "odd"})
_SYMBOLS = {"all": " & ", "any": " | ", "odd": " ^ "}
# A condition with more parts, or with a longer expression for one part, is shown as a summary.
_LONGEST_SHOWN_CONDITION = 40
_LONGEST_SHOWN_EXPRESSION = 400

# One part of a condition: its operation, its outcome number (-1 if none),
# and the positions of the parts it combines.
_Part: TypeAlias = "tuple[str, int, tuple[int, ...]]"


class Condition:
    """Select the side of a conditional branch or the iterations of a conditional loop from measurement outcomes.

    A condition is a logic expression over measurement outcomes, each read as
    0 or 1. Build it from :meth:`measured`, :meth:`always`, and :meth:`never`
    with the operators ``&`` (both hold), ``|`` (at least one holds), ``^``
    (exactly one holds), and ``~`` (does not hold). The operators simplify
    constant parts. Conditions may share parts, so a condition that reads many
    outcomes, such as the comparison of two syndromes, stays small. Two
    conditions are equal when they have the same structure.
    """

    __slots__ = ("_hash", "_operands", "_operation", "_outcome", "_outcomes", "_parts")

    def __init__(self, operation: str, operands: Iterable[Condition] = (), outcome: int | None = None) -> None:
        """Create one part of a condition.

        Prefer :meth:`measured`, :meth:`always`, :meth:`never`, and the
        operators, which also simplify constant parts.

        Args:
            operation: ``"measured"``, ``"always"``, ``"never"``, ``"not"``,
                ``"all"``, ``"any"``, or ``"odd"`` (an odd number of operands hold).
            operands: The conditions that ``"not"``, ``"all"``, ``"any"``, and ``"odd"`` combine.
            outcome: The measurement outcome number that ``"measured"`` reads.

        Raises:
            TypeError: If an operand is not a condition.
            ValueError: If the operation, operands, and outcome do not fit together.
        """
        operands = tuple(operands)
        if any(not isinstance(operand, Condition) for operand in operands):
            msg = "condition operands must be conditions"
            raise TypeError(msg)
        if operation not in _OPERATIONS:
            msg = f"unknown condition operation: {operation!r}"
            raise ValueError(msg)
        if operation == "measured":
            if isinstance(outcome, bool) or not isinstance(outcome, int) or outcome < 0:
                msg = "condition outcomes must be non-negative integers"
                raise ValueError(msg)
        elif outcome is not None:
            msg = f"a condition part {operation!r} reads no outcome"
            raise ValueError(msg)
        expected = {"measured": 0, "always": 0, "never": 0, "not": 1}.get(operation)
        if (expected is not None and len(operands) != expected) or (expected is None and not operands):
            msg = f"wrong number of operands for a condition part {operation!r}"
            raise ValueError(msg)
        self._operation = operation
        self._operands: tuple[Condition, ...] = operands
        self._outcome = outcome
        # Computed on first use; a condition does not change after construction.
        self._parts: tuple[_Part, ...] | None = None
        self._outcomes: tuple[int, ...] | None = None
        self._hash: int | None = None

    @classmethod
    def measured(cls, outcome: int) -> Condition:
        """Return the condition that measurement ``outcome`` gave 1."""
        return cls("measured", outcome=outcome)

    @classmethod
    def always(cls) -> Condition:
        """Return the condition that always holds."""
        return cls("always")

    @classmethod
    def never(cls) -> Condition:
        """Return the condition that never holds."""
        return cls("never")

    @property
    def operation(self) -> str:
        """The operation of this part, as passed to the constructor."""
        return self._operation

    @property
    def operands(self) -> tuple[Condition, ...]:
        """The conditions that this part combines."""
        return self._operands

    @property
    def outcome(self) -> int | None:
        """The outcome number that a ``"measured"`` part reads, else ``None``."""
        return self._outcome

    @property
    def outcomes(self) -> tuple[int, ...]:
        """All measurement outcome numbers that the condition reads, in ascending order."""
        if self._outcomes is None:
            outcomes = sorted({outcome for operation, outcome, _ in self._part_list() if operation == "measured"})
            self._outcomes = tuple(outcomes)
        return self._outcomes

    def holds(self, values: Mapping[int, int]) -> bool:
        """Return whether the condition holds for the given measurement outcomes.

        Raises:
            KeyError: If ``values`` lacks an outcome that the condition reads.
        """  # ruff: ignore[docstring-extraneous-exception] - The outcome lookup raises KeyError.
        results: list[bool] = []
        for operation, outcome, operands in self._part_list():
            if operation == "measured":
                result = bool(values[outcome])
            elif operation in {"always", "never"}:
                result = operation == "always"
            elif operation == "not":
                result = not results[operands[0]]
            elif operation == "all":
                result = all(results[position] for position in operands)
            elif operation == "any":
                result = any(results[position] for position in operands)
            else:
                result = sum(results[position] for position in operands) % 2 == 1
            results.append(result)
        return results[-1]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible condition data.

        ``parts`` lists each distinct part once, after the parts it combines;
        the last part is the whole condition.
        """
        parts: list[dict[str, object]] = []
        for operation, outcome, operands in self._part_list():
            part: dict[str, object] = {"operation": operation}
            if operation == "measured":
                part["outcome"] = outcome
            elif operands:
                part["operands"] = list(operands)
            parts.append(part)
        return {"parts": parts}

    @classmethod
    def from_dict(cls, data: object) -> Condition:
        """Restore a condition.

        Returns:
            The restored condition.

        Raises:
            ValueError: If the serialized condition is malformed.
        """
        mapping = require_mapping(data, "condition")
        built: list[Condition] = []
        for raw_part in require_list(mapping, "parts"):
            part = require_mapping(raw_part, "condition part")
            operation = require_str(part, "operation")
            positions = require_int_list(part, "operands") if "operands" in part else []
            if any(position < 0 or position >= len(built) for position in positions):
                msg = "condition parts may only combine earlier parts"
                raise ValueError(msg)
            outcome = require_int(part, "outcome") if operation == "measured" else None
            built.append(cls(operation, (built[position] for position in positions), outcome))
        if not built:
            msg = "a condition needs at least one part"
            raise ValueError(msg)
        return built[-1]

    def __and__(self, other: object) -> Condition:
        """Return the condition that both conditions hold."""
        return _combine("all", self, other)

    def __or__(self, other: object) -> Condition:
        """Return the condition that at least one condition holds."""
        return _combine("any", self, other)

    def __xor__(self, other: object) -> Condition:
        """Return the condition that exactly one condition holds."""
        return _combine("odd", self, other)

    def __invert__(self) -> Condition:
        """Return the condition that this condition does not hold."""
        if self._operation in {"always", "never"}:
            return Condition("never" if self._operation == "always" else "always")
        if self._operation == "not":
            return self._operands[0]
        return Condition("not", (self,))

    def __eq__(self, other: object) -> bool:
        """Return whether both conditions have the same structure."""
        if not isinstance(other, Condition):
            return NotImplemented
        return self is other or self._part_list() == other._part_list()

    def __hash__(self) -> int:
        """Return a hash of the condition structure."""
        if self._hash is None:
            self._hash = hash(self._part_list())
        return self._hash

    def __repr__(self) -> str:
        """Return the condition as an expression, or a summary of a long condition."""
        parts = self._part_list()
        summary = f"Condition(<{len(parts)} parts reading {len(self.outcomes)} outcomes>)"
        if len(parts) > _LONGEST_SHOWN_CONDITION:
            return summary
        texts: list[str] = []
        for operation, outcome, operands in parts:
            if operation == "measured":
                texts.append(f"measured({outcome})")
            elif operation in {"always", "never"}:
                texts.append(f"{operation}()")
            elif operation == "not":
                texts.append(f"~{texts[operands[0]]}")
            else:
                texts.append("(" + _SYMBOLS[operation].join(texts[position] for position in operands) + ")")
            # Shared parts appear once per use in the expression, so a few parts can give a very long text.
            if len(texts[-1]) > _LONGEST_SHOWN_EXPRESSION:
                return summary
        return f"Condition({texts[-1]})"

    def _part_list(self) -> tuple[_Part, ...]:
        """Return each distinct part once, after the parts it combines, with this condition last."""
        if self._parts is None:
            position_of_node: dict[int, int] = {}
            position_of_part: dict[_Part, int] = {}
            parts: list[_Part] = []
            stack: list[tuple[Condition, bool]] = [(self, False)]
            # Walk without recursion, so deep conditions do not reach Python's recursion limit.
            while stack:
                node, expanded = stack.pop()
                if id(node) in position_of_node:
                    continue
                if not expanded:
                    stack.append((node, True))
                    stack.extend((operand, False) for operand in reversed(node.operands))
                    continue
                part = (
                    node.operation,
                    -1 if node.outcome is None else node.outcome,
                    tuple(position_of_node[id(operand)] for operand in node.operands),
                )
                if part not in position_of_part:
                    position_of_part[part] = len(parts)
                    parts.append(part)
                position_of_node[id(node)] = position_of_part[part]
            self._parts = tuple(parts)
        return self._parts


def _combine(operation: str, left: Condition, right: object) -> Condition:
    """Combine two conditions with ``"all"``, ``"any"``, or ``"odd"`` and simplify constant parts.

    Returns:
        The combined condition, or ``NotImplemented`` if ``right`` is not a condition.
    """
    if not isinstance(right, Condition):
        return NotImplemented
    for constant, other in ((left, right), (right, left)):
        if constant.operation in {"always", "never"}:
            if operation == "odd":
                return ~other if constant.operation == "always" else other
            absorbing = "never" if operation == "all" else "always"
            return constant if constant.operation == absorbing else other
    operands = [
        *(left.operands if left.operation == operation else (left,)),
        *(right.operands if right.operation == operation else (right,)),
    ]
    return Condition(operation, operands)


@dataclass(frozen=True)
class Branch:
    """Run one of two step sequences depending on measurement outcomes."""

    condition: Condition
    then_steps: tuple[Step, ...]
    else_steps: tuple[Step, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the step sequences."""
        object.__setattr__(self, "then_steps", _normalize_steps(self.then_steps))
        object.__setattr__(self, "else_steps", _normalize_steps(self.else_steps))


@dataclass(frozen=True)
class Loop:
    """Repeat steps a fixed number of times (``for``) or while a condition holds (``while``).

    With ``count``, the loop has a fixed number of iterations: the steps run
    exactly ``count`` times. Without it, the loop is a conditional loop:
    ``entry`` decides whether the first iteration runs, and ``repeat``, which
    reads the outcomes of the iteration that just finished, decides each
    further iteration. Outcomes measured inside the loop belong to their
    iteration; conditions after the loop cannot read them.
    """

    steps: tuple[Step, ...]
    count: int | None = None
    entry: Condition | None = None
    repeat: Condition | None = None

    def __post_init__(self) -> None:
        """Normalize the steps and validate how the loop repeats."""
        object.__setattr__(self, "steps", _normalize_steps(self.steps))
        check_repetition(self.count, self.entry, self.repeat)


Step: TypeAlias = Circuit | Branch | Loop


@runtime_checkable
class _BranchShape(Protocol):
    """A program or schedule branch."""

    @property
    def condition(self) -> Condition: ...

    @property
    def then_steps(self) -> tuple[object, ...]: ...

    @property
    def else_steps(self) -> tuple[object, ...]: ...


@runtime_checkable
class _LoopShape(Protocol):
    """A program or schedule loop."""

    @property
    def steps(self) -> tuple[object, ...]: ...

    @property
    def count(self) -> int | None: ...

    @property
    def entry(self) -> Condition | None: ...

    @property
    def repeat(self) -> Condition | None: ...


@dataclass(frozen=True)
class Program:
    """Store a program of straight-line circuits, measurement-dependent branches, and loops.

    Every circuit of the program acts on the same ``num_ions`` ions. Each
    measurement that a condition reads carries a program-wide ``outcome``
    number.
    """

    num_ions: int
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        """Validate ion counts and measurement outcomes.

        Raises:
            TypeError: If the ion count has the wrong type.
            ValueError: If circuits disagree on the ion count, an outcome is
                measured twice, or a condition reads an outcome that is not
                measured before it on every path.
        """
        if isinstance(self.num_ions, bool) or not isinstance(self.num_ions, int):
            msg = "num_ions must be an integer"
            raise TypeError(msg)
        steps = _normalize_steps(self.steps)
        object.__setattr__(self, "steps", steps)
        if any(circuit.num_ions != self.num_ions for circuit in self.circuits()):
            msg = "every program circuit must act on num_ions ions"
            raise ValueError(msg)
        measured = [gate.outcome for gate in self.gates() if isinstance(gate, Measure) and gate.outcome is not None]
        if len(set(measured)) != len(measured):
            msg = "each measurement outcome must be produced by exactly one measurement"
            raise ValueError(msg)
        _check_outcomes_measured_before_use(steps, frozenset())

    @property
    def outcomes(self) -> tuple[int, ...]:
        """All measurement outcome numbers that conditions read, in ascending order."""
        return tuple(sorted({outcome for condition in self.conditions() for outcome in condition.outcomes}))

    def circuits(self) -> Iterator[Circuit]:
        """Yield every straight-line circuit of the program, on all paths."""
        yield from _circuits(self.steps)

    def branches(self) -> Iterator[Branch]:
        """Yield every branch of the program, including nested branches."""
        yield from (step for step in _all_steps(self.steps) if isinstance(step, Branch))

    def loops(self) -> Iterator[Loop]:
        """Yield every loop of the program, including nested loops."""
        yield from (step for step in _all_steps(self.steps) if isinstance(step, Loop))

    def conditions(self) -> Iterator[Condition]:
        """Yield every branch and loop condition of the program."""
        for step in _all_steps(self.steps):
            if isinstance(step, Branch):
                yield step.condition
            elif isinstance(step, Loop):
                yield from (condition for condition in (step.entry, step.repeat) if condition is not None)

    def gates(self) -> Iterator[GateAction]:
        """Yield every gate, measurement, and reset of the program, on all paths."""
        for circuit in self.circuits():
            yield from circuit.gates

    def path(self, values: Mapping[OutcomeKey, int]) -> tuple[Circuit, ...]:
        """Return the circuits that run for the given measurement outcomes, with loops repeated.

        Raises:
            ValueError: If a conditional loop repeats forever for these outcomes.
        """  # ruff: ignore[docstring-extraneous-exception] - The path walk raises ValueError.
        circuits: list[Circuit] = []

        def collect(step: object, _iterations: tuple[int, ...]) -> None:
            assert isinstance(step, Circuit)
            circuits.append(step)

        walk_path(self.steps, values, outcome_depths(self.steps), on_step=collect)
        return tuple(circuits)

    def paths(self, *, max_iterations: int | None = None) -> Iterator[dict[OutcomeKey, int]]:
        """Yield one assignment of the deciding outcomes for each distinct path.

        Each assignment contains exactly the outcomes that the conditions on
        its path read. A conditional loop needs ``max_iterations``; paths with
        more iterations are left out.
        """
        yield from enumerate_paths(self.steps, max_iterations=max_iterations)

    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON-compatible program."""
        return {
            "schema": PROGRAM_SCHEMA,
            "version": PROGRAM_VERSION,
            "num_ions": self.num_ions,
            "steps": [_step_to_dict(step) for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: object) -> Program:
        """Restore a program from its versioned representation.

        Returns:
            The restored program.

        Raises:
            ValueError: If the schema, version, or program data is invalid.
        """
        mapping = require_mapping(data, "serialized program")
        if mapping.get("schema") != PROGRAM_SCHEMA or mapping.get("version") != PROGRAM_VERSION:
            msg = "unsupported program schema or version"
            raise ValueError(msg)
        return cls(
            num_ions=require_int(mapping, "num_ions"),
            steps=tuple(_step_from_dict(item) for item in require_list(mapping, "steps")),
        )


def check_repetition(count: int | None, entry: Condition | None, repeat: Condition | None) -> None:
    """Validate that a loop has either a fixed count or both conditions.

    Raises:
        TypeError: If the count or a condition has the wrong type.
        ValueError: If both or neither ways to repeat are given, or the count is not positive.
    """
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int):
            msg = "loop count must be an integer"
            raise TypeError(msg)
        if count < 1:
            msg = "loop count must be at least 1"
            raise ValueError(msg)
        if entry is not None or repeat is not None:
            msg = "a loop with a count must not have conditions"
            raise ValueError(msg)
        return
    if not isinstance(entry, Condition) or not isinstance(repeat, Condition):
        msg = "a loop without a count needs entry and repeat conditions"
        raise TypeError(msg)


def outcome_depths(steps: Sequence[object]) -> dict[int, int]:
    """Return the number of loops around the measurement of each outcome.

    Steps are program steps, or schedule steps with the same branch and loop
    fields and :class:`~mqt.ionshuttler.core.schedule.Schedule` in place of
    circuits.
    """
    depths: dict[int, int] = {}

    def walk(items: Sequence[object], depth: int) -> None:
        for step in items:
            if isinstance(step, _BranchShape):
                walk(step.then_steps, depth)
                walk(step.else_steps, depth)
            elif isinstance(step, _LoopShape):
                walk(step.steps, depth + 1)
            elif isinstance(step, Circuit | Schedule):
                operations = step.gates if isinstance(step, Circuit) else step.path
                for operation in operations:
                    if isinstance(operation, Measure) and operation.outcome is not None:
                        depths[operation.outcome] = depth

    walk(steps, 0)
    return depths


def outcome_key(outcome: int, depth: int, iterations: tuple[int, ...]) -> OutcomeKey:
    """Return the key of an outcome measured inside ``depth`` loops, seen at ``iterations``."""
    return outcome if depth == 0 else (outcome, *iterations[:depth])


def walk_path(
    steps: Sequence[object],
    values: Mapping[OutcomeKey, int],
    depths: Mapping[int, int],
    *,
    on_step: Callable[[object, tuple[int, ...]], None],
    on_decision: Callable[[tuple[OutcomeKey, ...]], None] | None = None,
    iterations: tuple[int, ...] = (),
) -> None:
    """Follow the path that ``values`` selects through program or schedule steps.

    ``on_step`` receives each straight-line step with the iterations of its
    enclosing loops. ``on_decision`` receives the outcome keys of each
    condition before it is evaluated.

    Raises:
        ValueError: If a conditional loop repeats forever, because its repeat
            condition holds and reads no outcome of the loop body.
    """

    def decide(condition: Condition, condition_iterations: tuple[int, ...]) -> bool:
        keys = _keys(condition, depths, condition_iterations)
        if on_decision is not None:
            on_decision(keys)
        return _holds(condition, keys, values)

    for step in steps:
        if isinstance(step, _BranchShape):
            chosen = step.then_steps if decide(step.condition, iterations) else step.else_steps
            walk_path(chosen, values, depths, on_step=on_step, on_decision=on_decision, iterations=iterations)
        elif isinstance(step, _LoopShape):
            if step.count is not None:
                for iteration in range(step.count):
                    current = (*iterations, iteration)
                    walk_path(step.steps, values, depths, on_step=on_step, on_decision=on_decision, iterations=current)
                continue
            assert step.entry is not None
            assert step.repeat is not None
            # Outcomes measured outside the loop keep their values, so such a repeat condition never changes.
            reads_body = any(depths.get(outcome, 0) > len(iterations) for outcome in step.repeat.outcomes)
            iteration = 0
            running = decide(step.entry, iterations)
            while running:
                current = (*iterations, iteration)
                walk_path(step.steps, values, depths, on_step=on_step, on_decision=on_decision, iterations=current)
                running = decide(step.repeat, current)
                if running and not reads_body:
                    msg = "a conditional loop repeats forever: its repeat condition reads no outcome of its body"
                    raise ValueError(msg)
                iteration += 1
        else:
            on_step(step, iterations)


def enumerate_paths(steps: Sequence[object], *, max_iterations: int | None = None) -> Iterator[dict[OutcomeKey, int]]:
    """Yield one outcome assignment for each distinct sequence of condition results.

    Steps follow the shapes described in :func:`outcome_depths`. Each
    assignment contains exactly the outcomes that the conditions on its path
    read. A conditional loop needs ``max_iterations``; paths with more
    iterations are left out.

    Raises:
        ValueError: If a conditional loop is reached and ``max_iterations`` is ``None``.
    """  # ruff: ignore[docstring-extraneous-exception] - The path walk raises ValueError.
    depths = outcome_depths(steps)
    seen: set[tuple[bool, ...]] = set()
    work = tuple((step, ()) for step in steps)
    for values, decisions in _walk_paths(work, {}, (), depths, max_iterations):
        if decisions not in seen:
            seen.add(decisions)
            yield values


@dataclass(frozen=True)
class _LoopCheck:
    """The condition check before iteration ``iteration`` of a conditional loop."""

    loop: _LoopShape
    iteration: int


def _walk_paths(
    work: tuple[tuple[object, tuple[int, ...]], ...],
    values: dict[OutcomeKey, int],
    decisions: tuple[bool, ...],
    depths: Mapping[int, int],
    max_iterations: int | None,
) -> Iterator[tuple[dict[OutcomeKey, int], tuple[bool, ...]]]:
    for index, (step, iterations) in enumerate(work):
        rest = work[index + 1 :]
        condition: Condition | None
        if isinstance(step, _LoopCheck):
            condition = step.loop.entry if step.iteration == 0 else step.loop.repeat
            condition_iterations = iterations if step.iteration == 0 else (*iterations, step.iteration - 1)
        elif isinstance(step, _BranchShape):
            condition = step.condition
            condition_iterations = iterations
        elif isinstance(step, _LoopShape):
            if step.count is not None:
                body = tuple((item, (*iterations, iteration)) for iteration in range(step.count) for item in step.steps)
                yield from _walk_paths((*body, *rest), values, decisions, depths, max_iterations)
                return
            if max_iterations is None:
                msg = "enumerating the paths of a conditional loop needs max_iterations"
                raise ValueError(msg)
            yield from _walk_paths(
                ((_LoopCheck(step, 0), iterations), *rest), values, decisions, depths, max_iterations
            )
            return
        else:
            continue
        assert condition is not None
        keys = _keys(condition, depths, condition_iterations)
        unknown = [key for key in dict.fromkeys(keys) if key not in values]
        if unknown:
            for combination in product((0, 1), repeat=len(unknown)):
                assigned = {**values, **dict(zip(unknown, combination, strict=True))}
                yield from _walk_paths(work[index:], assigned, decisions, depths, max_iterations)
            return
        holds = _holds(condition, keys, values)
        if isinstance(step, _LoopCheck):
            following = rest
            if holds:
                assert max_iterations is not None
                if step.iteration >= max_iterations:
                    return
                body = tuple((item, (*iterations, step.iteration)) for item in step.loop.steps)
                following = (*body, (_LoopCheck(step.loop, step.iteration + 1), iterations), *rest)
        else:
            assert isinstance(step, _BranchShape)
            chosen = step.then_steps if holds else step.else_steps
            following = (*((item, iterations) for item in chosen), *rest)
        yield from _walk_paths(following, values, (*decisions, holds), depths, max_iterations)
        return
    yield dict(values), decisions


def _keys(condition: Condition, depths: Mapping[int, int], iterations: tuple[int, ...]) -> tuple[OutcomeKey, ...]:
    return tuple(outcome_key(outcome, depths.get(outcome, 0), iterations) for outcome in condition.outcomes)


def _holds(condition: Condition, keys: Sequence[OutcomeKey], values: Mapping[OutcomeKey, int]) -> bool:
    """Evaluate ``condition`` with the values of ``keys``, which follow the order of its outcomes.

    Returns:
        Whether the condition holds.
    """
    return condition.holds({outcome: values[key] for outcome, key in zip(condition.outcomes, keys, strict=True)})


def _normalize_steps(steps: Iterable[object]) -> tuple[Step, ...]:
    normalized: list[Step] = []
    for step in steps:
        if not isinstance(step, Circuit | Branch | Loop):
            msg = "program steps must be Circuit, Branch, or Loop values"
            raise TypeError(msg)
        normalized.append(step)
    return tuple(normalized)


def _all_steps(steps: tuple[Step, ...]) -> Iterator[Step]:
    for step in steps:
        yield step
        if isinstance(step, Branch):
            yield from _all_steps(step.then_steps)
            yield from _all_steps(step.else_steps)
        elif isinstance(step, Loop):
            yield from _all_steps(step.steps)


def _circuits(steps: tuple[Step, ...]) -> Iterator[Circuit]:
    yield from (step for step in _all_steps(steps) if isinstance(step, Circuit))


def _measured_outcomes(circuit: Circuit) -> frozenset[int]:
    return frozenset(gate.outcome for gate in circuit.gates if isinstance(gate, Measure) and gate.outcome is not None)


def _require_measured(condition: Condition, measured: frozenset[int], reader: str) -> None:
    missing = sorted(set(condition.outcomes) - measured)
    if missing:
        msg = f"{reader} reads outcomes {missing} before they are measured on every path"
        raise ValueError(msg)


def _check_outcomes_measured_before_use(steps: tuple[Step, ...], measured: frozenset[int]) -> frozenset[int]:
    """Return the outcomes measured on every path through ``steps``.

    Outcomes measured inside a loop stay inside it, so they are not returned.
    """
    for step in steps:
        if isinstance(step, Circuit):
            measured |= _measured_outcomes(step)
        elif isinstance(step, Branch):
            _require_measured(step.condition, measured, "a branch")
            then_measured = _check_outcomes_measured_before_use(step.then_steps, measured)
            else_measured = _check_outcomes_measured_before_use(step.else_steps, measured)
            measured = then_measured & else_measured
        else:
            if step.entry is not None:
                _require_measured(step.entry, measured, "a loop entry condition")
            body_measured = _check_outcomes_measured_before_use(step.steps, measured)
            if step.repeat is not None:
                _require_measured(step.repeat, body_measured, "a loop repeat condition")
    return measured


def _step_to_dict(step: Step) -> dict[str, object]:
    if isinstance(step, Circuit):
        return {
            "type": "circuit",
            "num_ions": step.num_ions,
            "gates": [gate.to_dict() for gate in step.gates],
            "predecessors": [sorted(values) for values in step.predecessors],
            "barriers": [[sorted(targets) for targets in group] for group in step.barriers],
        }
    if isinstance(step, Branch):
        return {
            "type": "branch",
            "condition": step.condition.to_dict(),
            "then_steps": [_step_to_dict(item) for item in step.then_steps],
            "else_steps": [_step_to_dict(item) for item in step.else_steps],
        }
    return {
        "type": "loop",
        "count": step.count,
        "entry": None if step.entry is None else step.entry.to_dict(),
        "repeat": None if step.repeat is None else step.repeat.to_dict(),
        "steps": [_step_to_dict(item) for item in step.steps],
    }


def _step_from_dict(data: object) -> Step:
    mapping = require_mapping(data, "program step")
    kind = require_str(mapping, "type")
    if kind == "circuit":
        gates: list[GateAction] = []
        for raw_gate in require_list(mapping, "gates"):
            gate_data = require_mapping(raw_gate, "gate")
            gate_type = GATE_TYPES.get(require_str(gate_data, "type"))
            if gate_type is None or not issubclass(gate_type, GateAction):
                msg = f"unknown program gate type: {gate_data.get('type')}"
                raise ValueError(msg)
            gate = gate_type.from_dict(gate_data)
            if not isinstance(gate, GateAction):
                msg = f"program step contains a non-gate action: {gate_data.get('type')}"
                raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
            gates.append(gate)
        predecessors = tuple(
            frozenset(_int_list(values, "predecessor set")) for values in require_list(mapping, "predecessors")
        )
        barriers: list[tuple[frozenset[int], ...]] = []
        for raw_group in require_list(mapping, "barriers"):
            if not isinstance(raw_group, list):
                msg = "barriers must contain lists of target sets"
                raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
            barriers.append(tuple(frozenset(_int_list(targets, "barrier target set")) for targets in raw_group))
        return Circuit(
            num_ions=require_int(mapping, "num_ions"),
            gates=tuple(gates),
            predecessors=predecessors,
            barriers=tuple(barriers),
        )
    if kind == "branch":
        return Branch(
            condition=Condition.from_dict(mapping.get("condition")),
            then_steps=tuple(_step_from_dict(item) for item in require_list(mapping, "then_steps")),
            else_steps=tuple(_step_from_dict(item) for item in require_list(mapping, "else_steps")),
        )
    if kind == "loop":
        entry = mapping.get("entry")
        repeat = mapping.get("repeat")
        count = mapping.get("count")
        if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
            msg = "loop count must be an integer or null"
            raise ValueError(msg)
        if count is None and (entry is None or repeat is None):
            msg = "a loop without a count needs entry and repeat conditions"
            raise ValueError(msg)
        return Loop(
            steps=tuple(_step_from_dict(item) for item in require_list(mapping, "steps")),
            count=count,
            entry=None if entry is None else Condition.from_dict(entry),
            repeat=None if repeat is None else Condition.from_dict(repeat),
        )
    msg = f"unknown program step type: {kind}"
    raise ValueError(msg)


def _int_list(values: object, label: str) -> list[int]:
    if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        msg = f"{label} must be a list of integers"
        raise ValueError(msg)
    return values


__all__ = ["Branch", "Condition", "Loop", "OutcomeKey", "Program", "Step"]
