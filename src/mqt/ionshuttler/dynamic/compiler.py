# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compile dynamic programs into branching schedules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from mqt.ionshuttler.circuit import Circuit, with_commutation_aware_dependencies
from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.core.result import CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule

from .._json_utils import require_mapping, require_str
from .program import Branch, Loop, Program
from .schedule import BranchingSchedule, ScheduledBranch, ScheduledLoop

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.grid.compiler import GridCompiler

    from .program import OutcomeKey, Step
    from .schedule import ScheduleStep

PROGRAM_RESULT_SCHEMA = "mqt.ionshuttler.program_result"
PROGRAM_RESULT_VERSION = 1

Placement = dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class PathSummary:
    """Describe the schedule of one measurement-dependent path."""

    outcomes: dict[OutcomeKey, int]
    duration: int
    gates: int
    transport_operations: int


@dataclass(frozen=True)
class ProgramResult:
    """Contain a compiled program and its branching schedule.

    Equality ignores ``wall_clock_s``. ``message`` names the failing part of
    the program when ``status`` is not ``SUCCESS``.
    """

    status: CompilationStatus
    program: Program
    schedule: BranchingSchedule
    architecture: Any
    wall_clock_s: float = field(default=0.0, compare=False)
    message: str = ""

    def path_summaries(self, *, max_iterations: int | None = None) -> tuple[PathSummary, ...]:
        """Return the duration and operation counts of every path.

        A loop without a fixed count needs ``max_iterations``; paths with more
        iterations are left out.
        """
        summaries: list[PathSummary] = []
        for values in self.schedule.paths(max_iterations=max_iterations):
            flat = self.schedule.schedule_for(values)
            gates = sum(isinstance(action, GateAction) for action in flat.path)
            summaries.append(PathSummary(values, flat.duration, gates, len(flat.path) - gates))
        return tuple(summaries)

    def validate(self) -> None:
        """Check every schedule against the program and every connection between schedules.

        Each precompiled schedule must contain exactly the gates of its
        circuit, respect their dependencies, and replay on the architecture.
        Each schedule must start from every machine state in which the
        schedules before it can end, and a loop body must end in the state in
        which it starts. Every schedule starts idle, so every path through
        branches and loop iterations is then valid without listing the paths.

        Raises:
            ValueError: If a schedule or a connection is invalid.
        """  # ruff: ignore[docstring-extraneous-exception] - Validation helpers raise ValueError.
        _check_steps(self.program.steps, self.schedule.steps, "program")
        self._check_connections(self.schedule.steps, [self.schedule.initial_state], "program")

    def _check_connections(self, steps: tuple[ScheduleStep, ...], states: list[Any], where: str) -> list[Any]:
        """Replay each schedule once and check that it fits every state before it.

        Returns:
            The distinct machine states in which ``steps`` can end.

        Raises:
            ValueError: If a schedule or a loop body does not connect.
        """
        for position, step in enumerate(steps):
            label = f"{where}, step {position}"
            if isinstance(step, Schedule):
                if not all(_same_idle_state(step.initial_state, state) for state in states):
                    msg = f"{label}: the schedule does not start from the machine state reached before it"
                    raise ValueError(msg)
                states = [self.architecture.replay_schedule(step)]
            elif isinstance(step, ScheduledBranch):
                ends = self._check_connections(step.then_steps, states, f"{label}, then")
                ends += self._check_connections(step.else_steps, states, f"{label}, else")
                states = _distinct_states(ends)
            else:
                ends = self._check_connections(step.steps, states, f"{label}, loop")
                if not all(_same_idle_state(end, state) for end in ends for state in states):
                    msg = f"{label}: the loop body does not end in the machine state in which it starts"
                    raise ValueError(msg)
        return states

    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON-compatible result."""
        return {
            "schema": PROGRAM_RESULT_SCHEMA,
            "version": PROGRAM_RESULT_VERSION,
            "status": self.status.value,
            "message": self.message,
            "wall_clock_s": self.wall_clock_s,
            "architecture": self.architecture.to_dict(),
            "program": self.program.to_dict(),
            "schedule": self.schedule.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        decode_architecture: Callable[[object], Any],
        decode_action: Callable[[Mapping[str, object]], Action],
        decode_state: Callable[[object], Any],
    ) -> ProgramResult:
        """Restore a result with the architecture's decoders.

        Returns:
            The restored result.

        Raises:
            ValueError: If the schema, version, or result data is invalid.
        """
        mapping = require_mapping(data, "serialized program result")
        if mapping.get("schema") != PROGRAM_RESULT_SCHEMA or mapping.get("version") != PROGRAM_RESULT_VERSION:
            msg = "unsupported program result schema or version"
            raise ValueError(msg)
        wall_clock_s = mapping.get("wall_clock_s", 0.0)
        if isinstance(wall_clock_s, bool) or not isinstance(wall_clock_s, int | float):
            msg = "wall_clock_s must be a number"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
        return cls(
            status=CompilationStatus(require_str(mapping, "status")),
            program=Program.from_dict(mapping.get("program")),
            schedule=BranchingSchedule.from_dict(
                mapping.get("schedule"), decode_action=decode_action, decode_state=decode_state
            ),
            architecture=decode_architecture(mapping.get("architecture")),
            wall_clock_s=float(wall_clock_s),
            message=require_str(mapping, "message"),
        )

    def save(self, filename: str | Path) -> Path:
        """Write this result to a UTF-8 JSON file.

        Returns:
            The path written.
        """
        output_path = Path(filename)
        if output_path.suffix != ".json":
            output_path = Path(f"{output_path}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return output_path


class _CompilationFailedError(Exception):
    """Stop compilation at the first part that cannot be compiled."""


def compile_program(
    program: Program,
    compiler: GridCompiler,
    *,
    initial_placement: Mapping[str, Sequence[int]] | None = None,
    feedback_delay: int = 0,
) -> ProgramResult:
    """Compile every path of a dynamic program with a Grid compiler.

    Each straight-line circuit is compiled once. Both sides of a branch start
    from the placement before the branch. When later steps follow the branch,
    both sides return to that placement, so that the program continues from
    the same placement on every path. A branch that ends a step sequence ends
    where that sequence must end. A loop body is compiled once and returns to
    the placement in which the loop starts, so that every iteration runs the
    same schedules. Time between a measurement and a decision that reads it is
    at least ``feedback_delay``.

    Args:
        program: Program to compile.
        compiler: Compiler for the straight-line circuits.
        initial_placement: Optional segment-to-ion placement before the program.
        feedback_delay: Time from the end of a measurement until a branch that
            reads its outcome can start.

    Returns:
        The compiled program. Its status is ``FAILED`` with a message when a
        circuit or a return to the placement before a branch fails.
    """
    started = perf_counter()
    placement = (
        None if initial_placement is None else _placement_of(compiler.architecture.initial_state(initial_placement))
    )
    builder = _ProgramBuilder(program, compiler, placement)
    status = CompilationStatus.SUCCESS
    message = ""
    try:
        steps = builder.compile_steps(program.steps, None, "program", record=True)
    except _CompilationFailedError as error:
        status = CompilationStatus.FAILED
        message = str(error)
        steps = tuple(builder.compiled)
    initial_state = builder.initial_state
    if initial_state is None:
        initial_state = builder.compile_circuit(Circuit(program.num_ions, (), ())).schedule.initial_state
    schedule = BranchingSchedule(steps, initial_state, feedback_delay)
    return ProgramResult(status, program, schedule, compiler.architecture, perf_counter() - started, message)


class _ProgramBuilder:
    """Compile program steps while tracking the placement between them."""

    def __init__(self, program: Program, compiler: GridCompiler, placement: Placement | None) -> None:
        self.program = program
        self.compiler = compiler
        self.placement = placement
        self.initial_state: Any = None
        self.compiled: list[ScheduleStep] = []

    def compile_circuit(self, circuit: Circuit, target: Placement | None = None) -> Any:  # ruff: ignore[any-type] - compiler result
        result = self.compiler.compile(circuit, initial_placement=self.placement, target_placement=target)
        if self.initial_state is None:
            self.initial_state = result.schedule.initial_state
        self.placement = _placement_of(result.final_state)
        return result

    def compile_steps(
        self, steps: tuple[Step, ...], target: Placement | None, where: str, *, record: bool = False
    ) -> tuple[ScheduleStep, ...]:
        """Compile steps that end in ``target`` when it is given.

        The last circuit carries ``target`` as its target placement; without
        any steps, a circuit without gates moves the ions there. ``record``
        keeps each finished step, so that a failed compilation can still
        return the steps before the failure.

        Returns:
            The compiled steps.

        Raises:
            _CompilationFailedError: If a circuit or the return to ``target`` fails.
        """
        compiled = self.compiled if record else []
        for position, step in enumerate(steps):
            last = position == len(steps) - 1
            if isinstance(step, Circuit):
                result = self.compile_circuit(step, target if last else None)
                compiled.append(result.schedule)
                if result.status is not CompilationStatus.SUCCESS:
                    gates_done = result.diagnostics is not None and result.diagnostics.gates == len(step.gates)
                    reason = (
                        "returning to the placement before the branch failed"
                        if gates_done
                        else f"circuit compilation ended with {result.status.value}"
                    )
                    msg = f"{where}, step {position}: {reason}"
                    raise _CompilationFailedError(msg)
                continue
            if self.placement is None:
                self.compile_circuit(Circuit(self.program.num_ions, (), ()))
            if isinstance(step, Loop):
                compiled.append(self.compile_loop(step, f"{where}, step {position}"))
                continue
            # Paths merge only when later steps follow, so a final branch ends where the steps must end.
            join = target if last else self.placement
            compiled.append(self.compile_branch(step, join, f"{where}, step {position}"))
        if target is not None and self.placement != target:
            result = self.compile_circuit(Circuit(self.program.num_ions, (), ()), target)
            compiled.append(result.schedule)
            if result.status is not CompilationStatus.SUCCESS:
                msg = f"{where}: returning to the placement before the branch failed"
                raise _CompilationFailedError(msg)
        return tuple(compiled)

    def compile_branch(self, branch: Branch, join: Placement | None, where: str) -> ScheduledBranch:
        """Compile both sides of a branch so that each ends in ``join`` when it is given.

        Returns:
            The compiled branch.
        """
        before = self.placement
        then_steps = self.compile_steps(branch.then_steps, join, f"{where}, then")
        self.placement = before
        else_steps = self.compile_steps(branch.else_steps, join, f"{where}, else")
        self.placement = join
        return ScheduledBranch(branch.condition, then_steps, else_steps)

    def compile_loop(self, loop: Loop, where: str) -> ScheduledLoop:
        """Compile a loop body once so that it ends in the placement in which it starts.

        Returns:
            The compiled loop.
        """
        start = self.placement
        steps = self.compile_steps(loop.steps, start, f"{where}, loop")
        self.placement = start
        return ScheduledLoop(steps, loop.count, loop.entry, loop.repeat)


def _placement_of(state: Any) -> Placement:  # ruff: ignore[any-type] - architecture machine state
    return dict(state.occupancy)


def _same_idle_state(first: Any, second: Any) -> bool:  # ruff: ignore[any-type] - architecture machine states
    time = max(first.time, second.time)
    return first.at_time(time) == second.at_time(time)


def _distinct_states(states: list[Any]) -> list[Any]:
    distinct: list[Any] = []
    for state in states:
        if not any(_same_idle_state(state, known) for known in distinct):
            distinct.append(state)
    return distinct


def _check_steps(program_steps: tuple[Step, ...], schedule_steps: tuple[ScheduleStep, ...], where: str) -> None:
    extra = schedule_steps[len(program_steps) :]
    if len(extra) > 1 or len(schedule_steps) < len(program_steps):
        msg = f"{where}: the schedule does not match the program structure"
        raise ValueError(msg)
    for step in extra:
        if not isinstance(step, Schedule):
            msg = f"{where}: the schedule does not match the program structure"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Invalid artifacts use ValueError.
        if any(isinstance(action, GateAction) for action in step.path):
            msg = f"{where}: the return to the placement before the branch contains gates"
            raise ValueError(msg)
    if len(schedule_steps) < len(program_steps):
        msg = f"{where}: the schedule does not match the program structure"
        raise ValueError(msg)
    for position, (step, scheduled) in enumerate(zip(program_steps, schedule_steps, strict=False)):
        label = f"{where}, step {position}"
        if isinstance(step, Circuit) and isinstance(scheduled, Schedule):
            _check_circuit(step, scheduled, label)
        elif isinstance(step, Branch) and isinstance(scheduled, ScheduledBranch):
            if step.condition != scheduled.condition:
                msg = f"{label}: the schedule uses a different branch condition"
                raise ValueError(msg)
            _check_steps(step.then_steps, scheduled.then_steps, f"{label}, then")
            _check_steps(step.else_steps, scheduled.else_steps, f"{label}, else")
        elif isinstance(step, Loop) and isinstance(scheduled, ScheduledLoop):
            if (step.count, step.entry, step.repeat) != (scheduled.count, scheduled.entry, scheduled.repeat):
                msg = f"{label}: the schedule repeats the loop differently"
                raise ValueError(msg)
            _check_steps(step.steps, scheduled.steps, f"{label}, loop")
        else:
            msg = f"{label}: the schedule does not match the program structure"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Invalid artifacts use ValueError.


def _check_circuit(circuit: Circuit, schedule: Schedule[Any, Any], where: str) -> None:
    scheduled = {
        item.action.gate_id: item
        for item in schedule.scheduled_actions
        if isinstance(item.action, GateAction) and item.action.gate_id is not None
    }
    gate_count = sum(isinstance(action, GateAction) for action in schedule.path)
    if gate_count != len(circuit.gates) or set(scheduled) != set(circuit.gate_ids):
        msg = f"{where}: the schedule must contain each circuit gate exactly once"
        raise ValueError(msg)
    # Commuting gates may change order, so check only the dependencies that remain.
    predecessors = with_commutation_aware_dependencies(circuit).predecessors
    for gate_id, gate in enumerate(circuit.gates):
        item = scheduled[gate_id]
        if item.action != gate:
            msg = f"{where}: scheduled gate {gate_id} differs from the circuit"
            raise ValueError(msg)
        if any(item.start_time < scheduled[predecessor].end_time for predecessor in predecessors[gate_id]):
            msg = f"{where}: gate {gate_id} starts before one of its predecessors ends"
            raise ValueError(msg)


__all__ = ["PathSummary", "ProgramResult", "compile_program"]
