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
PROGRAM_VERSION = 1

OutcomeKey: TypeAlias = "int | tuple[int, ...]"
"""Identify one measured value: the outcome number, followed by the iteration
of each enclosing loop for an outcome measured inside loops."""


@dataclass(frozen=True)
class Condition:
    """Decide a branch from measurement outcomes.

    ``outcomes`` lists the measurement outcomes that the decision reads, in
    ascending order. ``true_for`` lists every combination of these outcomes,
    as 0/1 values in the order of ``outcomes``, for which the condition holds.
    """

    outcomes: tuple[int, ...]
    true_for: frozenset[tuple[int, ...]]

    def __post_init__(self) -> None:
        """Validate the outcome numbers and combinations.

        Raises:
            ValueError: If an outcome number or combination is malformed.
        """
        outcomes = tuple(self.outcomes)
        if any(isinstance(outcome, bool) or not isinstance(outcome, int) or outcome < 0 for outcome in outcomes):
            msg = "condition outcomes must be non-negative integers"
            raise ValueError(msg)
        if list(outcomes) != sorted(set(outcomes)):
            msg = "condition outcomes must be unique and ascending"
            raise ValueError(msg)
        true_for = frozenset(tuple(values) for values in self.true_for)
        for values in true_for:
            if len(values) != len(outcomes) or any(value not in {0, 1} for value in values):
                msg = "each true_for entry must give one 0/1 value per condition outcome"
                raise ValueError(msg)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "true_for", true_for)

    def holds(self, values: Mapping[int, int]) -> bool:
        """Return whether the condition holds for the given measurement outcomes."""
        return tuple(values[outcome] for outcome in self.outcomes) in self.true_for

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible condition data."""
        return {"outcomes": list(self.outcomes), "true_for": [list(values) for values in sorted(self.true_for)]}

    @classmethod
    def from_dict(cls, data: object) -> Condition:
        """Restore a condition.

        Returns:
            The restored condition.

        Raises:
            ValueError: If the serialized condition is malformed.
        """
        mapping = require_mapping(data, "condition")
        true_for: list[tuple[int, ...]] = []
        for values in require_list(mapping, "true_for"):
            if not isinstance(values, list):
                msg = "true_for must contain lists of 0/1 values"
                raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
            true_for.append(tuple(values))
        return cls(tuple(require_int_list(mapping, "outcomes")), frozenset(true_for))


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
    """Repeat steps a fixed number of times or while a condition holds.

    With ``count``, the steps run exactly ``count`` times. Without it,
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
        """Return the circuits that run for the given measurement outcomes, with loops repeated."""
        circuits: list[Circuit] = []

        def collect(step: object, _iterations: tuple[int, ...]) -> None:
            assert isinstance(step, Circuit)
            circuits.append(step)

        walk_path(self.steps, values, outcome_depths(self.steps), on_step=collect)
        return tuple(circuits)

    def paths(self, *, max_iterations: int | None = None) -> Iterator[dict[OutcomeKey, int]]:
        """Yield one assignment of the deciding outcomes for each distinct path.

        Each assignment contains exactly the outcomes that the conditions on
        its path read. A loop without a fixed count needs ``max_iterations``;
        paths with more iterations are left out.
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
    """

    def decide(condition: Condition, condition_iterations: tuple[int, ...]) -> bool:
        keys = _keys(condition, depths, condition_iterations)
        if on_decision is not None:
            on_decision(keys)
        return tuple(values[key] for key in keys) in condition.true_for

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
            iteration = 0
            running = decide(step.entry, iterations)
            while running:
                current = (*iterations, iteration)
                walk_path(step.steps, values, depths, on_step=on_step, on_decision=on_decision, iterations=current)
                running = decide(step.repeat, current)
                iteration += 1
        else:
            on_step(step, iterations)


def enumerate_paths(steps: Sequence[object], *, max_iterations: int | None = None) -> Iterator[dict[OutcomeKey, int]]:
    """Yield one outcome assignment for each distinct sequence of decisions.

    Steps follow the shapes described in :func:`outcome_depths`. Each
    assignment contains exactly the outcomes that the conditions on its path
    read. A loop without a fixed count needs ``max_iterations``; paths with
    more iterations are left out.

    Raises:
        ValueError: If a loop without a fixed count is reached and ``max_iterations`` is ``None``.
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
    """The decision before iteration ``iteration`` of a loop without a fixed count."""

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
                msg = "enumerating the paths of a loop without a fixed count needs max_iterations"
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
        holds = tuple(values[key] for key in keys) in condition.true_for
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
