# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compile dynamic programs into branching schedules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, overload

from mqt.ionshuttler.circuit import Circuit, with_commutation_aware_dependencies
from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.core.result import CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule
from mqt.ionshuttler.grid.compiler import GridCompiler
from mqt.ionshuttler.grid.config import GridDependencyMode

from .._json_utils import require_mapping, require_str
from .program import Branch, Loop, Program
from .schedule import BranchingSchedule, ScheduledBranch, ScheduledLoop

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.core.result import CompilationResult
    from mqt.ionshuttler.linear.compiler import LinearCompiler

    from .program import OutcomeKey, Step
    from .schedule import ScheduleStep

PROGRAM_RESULT_SCHEMA = "mqt.ionshuttler.program_result"
PROGRAM_RESULT_VERSION = 1

# Grid placements map segments to ions; Linear placements list the site of each ion.
_PlacementT = TypeVar("_PlacementT")
_PlacementT_contra = TypeVar("_PlacementT_contra", contravariant=True)


class _CircuitCompiler(Protocol[_PlacementT_contra]):
    """Compile one straight-line circuit between placements of one architecture.

    ``GridCompiler`` and ``LinearCompiler`` satisfy this protocol. The machine
    state of each result reports its placement in the form that ``compile``
    accepts.
    """

    @property
    def architecture(self) -> object:
        """The architecture that the compiler targets."""
        ...

    def compile(
        self,
        circuit: Circuit,
        *,
        initial_placement: _PlacementT_contra | None = None,
        target_placement: _PlacementT_contra | None = None,
    ) -> CompilationResult[Any, Any, Any, Any]:
        """Compile a circuit that starts in ``initial_placement`` and ends in ``target_placement``."""
        ...


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


@overload
def compile_program(
    program: Program,
    compiler: GridCompiler,
    *,
    initial_placement: Mapping[str, Sequence[int]] | None = None,
    feedback_delay: int = 0,
    predicted_outcome: int | None = None,
) -> ProgramResult: ...


@overload
def compile_program(
    program: Program,
    compiler: LinearCompiler,
    *,
    initial_placement: Sequence[int] | None = None,
    feedback_delay: int = 0,
    predicted_outcome: int | None = None,
) -> ProgramResult: ...


def compile_program(
    program: Program,
    compiler: _CircuitCompiler[_PlacementT],
    *,
    initial_placement: _PlacementT | None = None,
    feedback_delay: int = 0,
    predicted_outcome: int | None = None,
) -> ProgramResult:
    """Compile every path of a dynamic program with a Grid or Linear compiler.

    Each straight-line circuit is compiled once. Both sides of a branch start
    from the placement before the branch. When later steps follow the branch,
    both sides return to that placement, so that the program continues from
    the same placement on every path. A branch that ends a step sequence ends
    where that sequence must end. A loop body is compiled once and returns to
    the placement in which the loop starts, so that every iteration runs the
    same schedules. Time between a measurement and a decision that reads it is
    at least ``feedback_delay``.

    With ``predicted_outcome``, the compiler assumes that every measurement
    that a condition reads returns this value. It compiles the circuits of
    this predicted path up to the next loop as one circuit, so that transport
    for a later circuit can start while an earlier circuit still runs. Gates
    of a circuit still start only after every gate of the circuits before it
    ends. The compiler then cuts the result into one schedule per circuit.
    The other side of each branch on the predicted path starts from the
    placement that the predicted path reaches at the branch and, when later
    steps follow, ends in the placement where the predicted path continues.

    Args:
        program: Program to compile.
        compiler: ``GridCompiler`` or ``LinearCompiler`` for the straight-line
            circuits.
        initial_placement: Optional placement before the program, in the form
            that ``compiler.compile`` accepts.
        feedback_delay: Time from the end of a measurement until a branch that
            reads its outcome can start.
        predicted_outcome: Optional outcome, 0 or 1, that every measurement
            on the predicted path returns.

    Returns:
        The compiled program. Its status is ``FAILED`` with a message when a
        circuit or a return to the placement before a branch fails.

    Raises:
        ValueError: If ``predicted_outcome`` is neither 0 nor 1.
    """
    if predicted_outcome is not None and (isinstance(predicted_outcome, bool) or predicted_outcome not in {0, 1}):
        msg = "predicted_outcome must be 0 or 1"
        raise ValueError(msg)
    started = perf_counter()
    # A circuit without gates checks the placement and brings it into the form of the machine state.
    placement = (
        None
        if initial_placement is None
        else compiler.compile(
            Circuit(program.num_ions, (), ()), initial_placement=initial_placement
        ).final_state.placement
    )
    builder = _ProgramBuilder(program, compiler, placement, predicted_outcome)
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


class _ProgramBuilder(Generic[_PlacementT]):
    """Compile program steps while tracking the placement between them."""

    def __init__(
        self,
        program: Program,
        compiler: _CircuitCompiler[_PlacementT],
        placement: _PlacementT | None,
        predicted_outcome: int | None,
    ) -> None:
        self.program = program
        self.compiler = compiler
        self.placement = placement
        self.predicted_outcome = predicted_outcome
        self.initial_state: Any = None
        self.compiled: list[ScheduleStep] = []

    def compile_circuit(
        self, circuit: Circuit, target: _PlacementT | None = None
    ) -> CompilationResult[Any, Any, Any, Any]:
        if circuit.gates and not isinstance(self.compiler, GridCompiler):
            # Grid derives dependencies according to its configuration; Linear keeps the ones it receives.
            circuit = with_commutation_aware_dependencies(circuit)
        result = self.compiler.compile(circuit, initial_placement=self.placement, target_placement=target)
        if self.initial_state is None:
            self.initial_state = result.schedule.initial_state
        self.placement = result.final_state.placement
        return result

    def compile_steps(
        self, steps: tuple[Step, ...], target: _PlacementT | None, where: str, *, record: bool = False
    ) -> tuple[ScheduleStep, ...]:
        """Compile steps that end in ``target`` when it is given.

        The last circuit carries ``target`` as its target placement; without
        any steps, a circuit without gates moves the ions there. ``record``
        keeps each finished step, so that a failed compilation can still
        return the steps before the failure. With a predicted outcome, each
        run of steps without loops is compiled along its predicted path.

        Returns:
            The compiled steps.

        Raises:
            _CompilationFailedError: If a circuit or the return to ``target`` fails.
        """
        compiled = self.compiled if record else []
        if self.predicted_outcome is None:
            for position, step in enumerate(steps):
                last = position == len(steps) - 1
                if isinstance(step, Circuit):
                    result = self.compile_circuit(step, target if last else None)
                    compiled.append(result.schedule)
                    _raise_if_failed(result, step, f"{where}, step {position}")
                else:
                    compiled.append(self.compile_control_step(step, target, f"{where}, step {position}", last=last))
        else:
            position = 0
            while position < len(steps):
                step = steps[position]
                if not isinstance(step, Circuit) and _contains_loop(step):
                    last = position == len(steps) - 1
                    compiled.append(self.compile_control_step(step, target, f"{where}, step {position}", last=last))
                    position += 1
                    continue
                end = position
                while end < len(steps) and not _contains_loop(steps[end]):
                    end += 1
                final = end == len(steps)
                compiled.extend(
                    self.compile_predicted_path(
                        steps[position:end],
                        target if final else None,
                        f"{where}, step {position}",
                        free_end=final and target is None,
                    )
                )
                position = end
        if target is not None and self.placement != target:
            result = self.compile_circuit(Circuit(self.program.num_ions, (), ()), target)
            compiled.append(result.schedule)
            if result.status is not CompilationStatus.SUCCESS:
                msg = f"{where}: returning to the placement before the branch failed"
                raise _CompilationFailedError(msg)
        return tuple(compiled)

    def compile_control_step(
        self, step: Branch | Loop, target: _PlacementT | None, where: str, *, last: bool
    ) -> ScheduledBranch | ScheduledLoop:
        """Compile a branch or loop step of a sequence that ends in ``target`` when it is given.

        Returns:
            The compiled step.
        """
        if self.placement is None:
            self.compile_circuit(Circuit(self.program.num_ions, (), ()))
        if isinstance(step, Loop):
            return self.compile_loop(step, where)
        # Paths merge only when later steps follow, so a final branch ends where the steps must end.
        join = target if last else self.placement
        return self.compile_branch(step, join, where)

    def compile_predicted_path(
        self, steps: tuple[Step, ...], target: _PlacementT | None, where: str, *, free_end: bool
    ) -> list[ScheduleStep]:
        """Compile loop-free steps along their predicted path.

        ``free_end`` means that no step follows and no placement is required
        after ``steps``.

        Returns:
            The compiled steps.
        """
        if self.placement is None:
            self.compile_circuit(Circuit(self.program.num_ions, (), ()))
        circuits = tuple(self.predicted_circuits(steps))
        cut = self.cut_predicted_path(circuits, target, where) if circuits else []
        return self.build_predicted(steps, iter(cut), where, free_end=free_end)

    def predicted_circuits(self, steps: tuple[Step, ...]) -> Iterator[Circuit]:
        """Yield the circuits of loop-free steps on their predicted path, in order."""
        for step in steps:
            if isinstance(step, Circuit):
                yield step
            else:
                assert isinstance(step, Branch)
                yield from self.predicted_circuits(self.predicted_side(step)[0])

    def predicted_side(self, branch: Branch) -> tuple[tuple[Step, ...], tuple[Step, ...], bool]:
        """Return the predicted side of a branch, the other side, and whether the predicted side is ``then``."""
        assert self.predicted_outcome is not None
        values = dict.fromkeys(branch.condition.outcomes, self.predicted_outcome)
        if branch.condition.holds(values):
            return branch.then_steps, branch.else_steps, True
        return branch.else_steps, branch.then_steps, False

    def cut_predicted_path(
        self, circuits: tuple[Circuit, ...], target: _PlacementT | None, where: str
    ) -> list[tuple[Schedule[Any, Any], _PlacementT]]:
        """Compile the predicted path as one circuit and cut it into one schedule per circuit.

        The schedule of a circuit contains its gates and the transport that
        starts after the gates of the circuits before it end and before its
        own gates end. The last schedule also contains the transport after
        the last gate. Each schedule starts idle once the schedule before it
        ends; this only delays transport, so each schedule stays valid.

        Returns:
            Each circuit's schedule with the placement after it.

        Raises:
            _CompilationFailedError: If the predicted path does not compile.
        """
        compiler, circuits_with_dependencies = _keeping_dependencies(self.compiler, circuits)
        path, origins = _predicted_path_circuit(self.program.num_ions, circuits_with_dependencies)
        result = compiler.compile(path, initial_placement=self.placement, target_placement=target)
        if self.initial_state is None:
            self.initial_state = result.schedule.initial_state
        if result.status is not CompilationStatus.SUCCESS:
            msg = f"{where}: compiling the predicted path ended with {result.status.value}"
            raise _CompilationFailedError(msg)
        items = result.schedule.scheduled_actions
        gate_ends = [result.schedule.start_time] * len(circuits)
        for item in items:
            if isinstance(item.action, GateAction):
                circuit_index = origins[item.action.gate_id][0]
                gate_ends[circuit_index] = max(gate_ends[circuit_index], item.end_time)
        for circuit_index in range(1, len(circuits)):
            gate_ends[circuit_index] = max(gate_ends[circuit_index], gate_ends[circuit_index - 1])
        architecture: Any = self.compiler.architecture
        cut: list[tuple[Schedule[Any, Any], _PlacementT]] = []
        initial_state = result.schedule.initial_state
        for circuit_index, circuit in enumerate(circuits):
            begin = result.schedule.start_time if circuit_index == 0 else gate_ends[circuit_index - 1]
            last = circuit_index == len(circuits) - 1
            members = [
                item
                for item in items
                if (
                    origins[item.action.gate_id][0] == circuit_index
                    if isinstance(item.action, GateAction)
                    else begin <= item.start_time and (last or item.start_time < gate_ends[circuit_index])
                )
            ]
            scheduled = tuple(
                replace(
                    item,
                    action_id=position,
                    action=(
                        circuit.gates[origins[item.action.gate_id][1]]
                        if isinstance(item.action, GateAction)
                        else item.action
                    ),
                    start_time=item.start_time - begin + initial_state.time,
                )
                for position, item in enumerate(members)
            )
            end_time = max((item.end_time for item in scheduled), default=initial_state.time)
            schedule: Schedule[Any, Any] = Schedule(scheduled, end_time, initial_state)
            placement = architecture.replay_schedule(schedule).placement
            cut.append((schedule, placement))
            initial_state = self.compiler.compile(
                Circuit(self.program.num_ions, (), ()), initial_placement=placement
            ).schedule.initial_state
        return cut

    def build_predicted(
        self,
        steps: tuple[Step, ...],
        cut: Iterator[tuple[Schedule[Any, Any], _PlacementT]],
        where: str,
        *,
        free_end: bool,
    ) -> list[ScheduleStep]:
        """Arrange the cut schedules of the predicted path and compile the other side of each branch.

        Returns:
            The compiled steps.
        """
        built: list[ScheduleStep] = []
        for position, step in enumerate(steps):
            label = f"{where}, step {position}" if position else where
            last = position == len(steps) - 1
            if isinstance(step, Circuit):
                schedule, self.placement = next(cut)
                built.append(schedule)
                continue
            assert isinstance(step, Branch)
            before = self.placement
            predicted, other, predicted_then = self.predicted_side(step)
            names = ("then", "else") if predicted_then else ("else", "then")
            predicted_steps = tuple(
                self.build_predicted(predicted, cut, f"{label}, {names[0]}", free_end=free_end and last)
            )
            after = self.placement
            # The other side ends where the predicted path continues; at the end of the program it ends anywhere.
            join = None if free_end and last else after
            self.placement = before
            other_steps = self.compile_steps(other, join, f"{label}, {names[1]}")
            self.placement = after
            then_steps, else_steps = (
                (predicted_steps, other_steps) if predicted_then else (other_steps, predicted_steps)
            )
            built.append(ScheduledBranch(step.condition, then_steps, else_steps))
        return built

    def compile_branch(self, branch: Branch, join: _PlacementT | None, where: str) -> ScheduledBranch:
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


def _raise_if_failed(result: CompilationResult[Any, Any, Any, Any], circuit: Circuit, where: str) -> None:
    """Stop the program compilation when a circuit or its return to a placement fails.

    Raises:
        _CompilationFailedError: If the result is not successful.
    """
    if result.status is CompilationStatus.SUCCESS:
        return
    gates_done = sum(isinstance(action, GateAction) for action in result.path) == len(circuit.gates)
    reason = (
        "returning to the placement before the branch failed"
        if gates_done
        else f"circuit compilation ended with {result.status.value}"
    )
    msg = f"{where}: {reason}"
    raise _CompilationFailedError(msg)


def _contains_loop(step: Step) -> bool:
    """Return whether a step is a loop or contains one."""
    if isinstance(step, Loop):
        return True
    if isinstance(step, Branch):
        return any(_contains_loop(inner) for inner in (*step.then_steps, *step.else_steps))
    return False


def _keeping_dependencies(
    compiler: _CircuitCompiler[Any], circuits: tuple[Circuit, ...]
) -> tuple[_CircuitCompiler[Any], tuple[Circuit, ...]]:
    """Return a compiler that keeps the dependencies of a joined circuit, and the circuits with these dependencies.

    Each circuit gets the dependencies that ``compile_circuit`` gives it alone:
    commutation-aware ones, unless a Grid compiler is configured for
    conservative ones. A Grid compiler that derives commutation-aware
    dependencies itself would derive them for the joined circuit and could
    move a commuting gate before an earlier circuit, so it receives them per
    circuit and keeps them.

    Returns:
        The compiler for the joined circuit and the circuits whose dependencies it keeps.
    """
    if isinstance(compiler, GridCompiler):
        if compiler.config.dependency_mode is not GridDependencyMode.COMMUTATION_AWARE:
            return compiler, circuits
        compiler = GridCompiler(
            compiler.architecture, replace(compiler.config, dependency_mode=GridDependencyMode.CONSERVATIVE)
        )
    return compiler, tuple(with_commutation_aware_dependencies(circuit) for circuit in circuits)


def _predicted_path_circuit(
    num_ions: int, circuits: tuple[Circuit, ...]
) -> tuple[Circuit, tuple[tuple[int, int], ...]]:
    """Join circuits into one circuit in which every gate follows all gates of the earlier circuits.

    Each gate without a predecessor in its own circuit depends on every gate
    of the closest earlier circuit with gates that no other gate of that
    circuit depends on.

    Returns:
        The joined circuit and, for each of its gates, the position of the
        source circuit and the gate ID there.
    """
    gates: list[GateAction] = []
    predecessors: list[frozenset[int]] = []
    barriers: list[tuple[frozenset[int], ...]] = []
    origins: list[tuple[int, int]] = []
    previous_last_gates: frozenset[int] = frozenset()
    pending: list[frozenset[int]] = []
    for circuit_index, circuit in enumerate(circuits):
        offset = len(gates)
        for gate_id, gate in enumerate(circuit.gates):
            own = frozenset(offset + predecessor for predecessor in circuit.predecessors[gate_id])
            predecessors.append(own or previous_last_gates)
            barriers.append((*pending, *circuit.barriers[gate_id]))
            pending = []
            gates.append(replace(gate, gate_id=offset + gate_id))
            origins.append((circuit_index, gate_id))
        pending.extend(circuit.barriers[-1])
        if circuit.gates:
            followed = {predecessor for gate_predecessors in circuit.predecessors for predecessor in gate_predecessors}
            previous_last_gates = frozenset(offset + gate_id for gate_id in circuit.gate_ids if gate_id not in followed)
    barriers.append(tuple(pending))
    return Circuit(num_ions, tuple(gates), tuple(predecessors), tuple(barriers)), tuple(origins)


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
