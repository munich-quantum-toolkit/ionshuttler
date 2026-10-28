# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for compiling dynamic programs on Grid architectures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.core.gates import GateAction, Measure, Rx, Rxx
from mqt.ionshuttler.core.schedule import Schedule
from mqt.ionshuttler.dynamic import (
    Branch,
    Condition,
    Loop,
    Program,
    ProgramResult,
    ScheduledBranch,
    ScheduledLoop,
    compile_program,
    parse_program,
)
from mqt.ionshuttler.grid import (
    CompilationStatus,
    GridArchitecture,
    GridCompiler,
    GridCompilerConfig,
    GridCompilerStrategy,
    GridMachineState,
    Junction,
    ProcessingZone,
    Segment,
    square_grid,
)
from mqt.ionshuttler.grid.actions import decode_grid_action

if TYPE_CHECKING:
    from mqt.ionshuttler.dynamic import BranchingSchedule

FIXTURES = Path(__file__).parent / "fixtures"
GRID = square_grid(3, processing_zone_segments=("h:1:0", "h:1:1"), segment_capacity=2)
BREADTH_FIRST = GridCompilerConfig(strategy=GridCompilerStrategy.BREADTH_FIRST)


def _program(name: str) -> Program:
    return parse_program((FIXTURES / f"{name}.qco").read_text(encoding="utf-8"))


def _two_zone_line() -> GridArchitecture:
    memory = Segment("memory", capacity=2)
    left = Segment("left", capacity=2, processing_zones=(ProcessingZone("left-pz"),))
    right = Segment("right", capacity=2, processing_zones=(ProcessingZone("right-pz"),))
    return GridArchitecture(
        (left, memory, right),
        (Junction("j0", (left.end, memory.start)), Junction("j1", (memory.end, right.start))),
    )


def _measure_then_branch() -> Program:
    prefix = Circuit(
        num_ions=2,
        gates=(Rxx(ion_a=0, ion_b=1, theta=0.5, gate_id=0), Measure(ion=0, outcome=0, gate_id=1)),
        predecessors=(frozenset(), frozenset({0})),
    )
    correction = Circuit(num_ions=2, gates=(Rx(ion=1, theta=0.5, gate_id=0),), predecessors=(frozenset(),))
    return Program(2, (prefix, Branch(Condition((0,), frozenset({(1,)})), (correction,))))


@pytest.mark.parametrize("name", ["teleportation", "nested_branches", "reset_and_reuse", "register_conditions"])
def test_every_path_of_a_fixture_program_replays(name: str) -> None:
    """Compile each fixture and replay every measurement-dependent path."""
    result = compile_program(_program(name), GridCompiler(GRID, BREADTH_FIRST), feedback_delay=2)

    assert result.status is CompilationStatus.SUCCESS
    result.validate()
    summaries = result.path_summaries()
    assert [summary.outcomes for summary in summaries] == list(result.program.paths())
    assert all(summary.gates > 0 for summary in summaries)


def test_flagged_steane_error_correction_compiles_on_every_path() -> None:
    """Compile the adaptive syndrome extraction of Chao and Reichardt on a Grid."""
    program = _program("steane_flagged_error_correction")

    result = compile_program(program, GridCompiler(GRID), feedback_delay=1)

    assert program.num_ions == 9
    first = program.steps[1]
    assert isinstance(first, Branch)
    assert first.condition == Condition((0, 1), frozenset({(0, 1), (1, 0), (1, 1)}))
    assert len(list(program.paths())) == 7
    assert result.status is CompilationStatus.SUCCESS
    result.validate()


def test_greedy_compiles_nested_branches_that_must_return() -> None:
    """Return to the placement before a branch when Greedy target routing alone would stall."""
    result = compile_program(_program("nested_branches"), GridCompiler(GRID), feedback_delay=1)

    assert result.status is CompilationStatus.SUCCESS
    result.validate()


def test_both_sides_of_a_branch_return_to_the_placement_before_it() -> None:
    """Let the program continue from one placement on every path."""
    result = compile_program(_program("teleportation"), GridCompiler(GRID, BREADTH_FIRST))
    branch = result.schedule.steps[1]

    assert isinstance(branch, ScheduledBranch)
    before = result.schedule.steps[0]
    assert isinstance(before, Schedule)
    placement = GRID.replay_schedule(before).occupancy
    for side in (branch.then_steps, branch.else_steps):
        state = before.initial_state
        for step in side:
            assert isinstance(step, Schedule)
            state = GRID.replay_schedule(step)
        if side:
            assert state.occupancy == placement


def test_a_branch_waits_for_the_feedback_delay_after_its_measurement() -> None:
    """Start the branch no earlier than the measurement end plus the delay."""
    program = _measure_then_branch()

    timeline = compile_program(program, GridCompiler(_two_zone_line()), feedback_delay=5).schedule.timeline_for({0: 1})

    measurement = next(
        item.end_time for item in timeline[0].schedule.scheduled_actions if isinstance(item.action, Measure)
    )
    assert timeline[1].start_time >= measurement + 5
    assert timeline[1].start_time > timeline[0].schedule.end_time


def test_a_path_schedule_contains_only_the_operations_of_its_branch() -> None:
    """Flatten the precompiled schedules of one path into one replayable schedule."""
    result = compile_program(_measure_then_branch(), GridCompiler(_two_zone_line()), feedback_delay=1)

    corrected = result.schedule.schedule_for({0: 1})
    uncorrected = result.schedule.schedule_for({0: 0})

    gates = [action for action in corrected.path if isinstance(action, GateAction)]
    assert any(isinstance(gate, Rx) for gate in gates)
    assert not any(isinstance(action, Rx) for action in uncorrected.path)
    assert [item.action_id for item in corrected.scheduled_actions] == list(range(len(corrected.scheduled_actions)))
    _two_zone_line().replay_schedule(corrected)


def test_result_round_trips_through_json(tmp_path: Path) -> None:
    """Restore a result with the Grid decoders."""
    result = compile_program(_program("register_conditions"), GridCompiler(GRID), feedback_delay=1)

    path = result.save(tmp_path / "result")
    restored = ProgramResult.from_dict(
        json.loads(path.read_text(encoding="utf-8")),
        decode_architecture=GridArchitecture.from_dict,
        decode_action=decode_grid_action,
        decode_state=GridMachineState.from_dict,
    )

    assert restored == result
    restored.validate()


def test_an_impossible_return_names_the_failing_branch_side() -> None:
    """Report the failing branch side instead of switching strategy silently."""
    first = Segment("first")
    second = Segment("second")
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (first, second, processor), (Junction("j0", (first.end, second.end, processor.start)),)
    )
    one_way = GridCompilerConfig(
        allowed_junction_crossings=frozenset({("first", "processor"), ("second", "processor")})
    )
    prefix = Circuit(num_ions=2, gates=(Measure(ion=0, outcome=0, gate_id=0),), predecessors=(frozenset(),))
    correction = Circuit(num_ions=2, gates=(Rx(ion=1, theta=0.5, gate_id=0),), predecessors=(frozenset(),))
    suffix = Circuit(num_ions=2, gates=(Rx(ion=0, theta=0.5, gate_id=0),), predecessors=(frozenset(),))
    program = Program(2, (prefix, Branch(Condition((0,), frozenset({(1,)})), (correction,)), suffix))

    result = compile_program(
        program, GridCompiler(architecture, one_way), initial_placement={"first": (0,), "second": (1,)}
    )

    assert result.status is CompilationStatus.FAILED
    assert result.message == "program, step 1, then, step 0: returning to the placement before the branch failed"
    assert len(result.schedule.steps) == 1


def test_a_final_branch_does_not_return_to_the_placement_before_it() -> None:
    """Skip the return when no later step follows the branch."""
    result = compile_program(_measure_then_branch(), GridCompiler(_two_zone_line()))
    branch = result.schedule.steps[1]

    assert isinstance(branch, ScheduledBranch)
    assert len(branch.then_steps) == 1
    assert branch.else_steps == ()
    result.validate()


def _replace_branch_side(schedule: BranchingSchedule, then_steps: tuple[Schedule, ...]) -> BranchingSchedule:
    branch = schedule.steps[1]
    assert isinstance(branch, ScheduledBranch)
    return replace(schedule, steps=(schedule.steps[0], replace(branch, then_steps=then_steps), *schedule.steps[2:]))


def test_validation_rejects_a_branch_that_does_not_return_to_the_placement() -> None:
    """Detect a continuation that would start from a different machine state."""
    suffix = Circuit(num_ions=2, gates=(Rx(ion=0, theta=0.1, gate_id=0),), predecessors=(frozenset(),))
    program = Program(2, (*_measure_then_branch().steps, suffix))
    architecture = _two_zone_line()
    result = compile_program(program, GridCompiler(architecture))
    branch = result.schedule.steps[1]
    program_branch = program.steps[1]
    assert isinstance(branch, ScheduledBranch)
    assert isinstance(program_branch, Branch)
    correction = program_branch.then_steps[0]
    compiled_correction = branch.then_steps[0]
    assert isinstance(correction, Circuit)
    assert isinstance(compiled_correction, Schedule)
    placement = dict(compiled_correction.initial_state.occupancy)
    elsewhere = {"memory": (0, 1)} if placement["memory"] != (0, 1) else {"right": (0, 1)}
    unreturned = (
        GridCompiler(architecture).compile(correction, initial_placement=placement, target_placement=elsewhere).schedule
    )
    broken = replace(result, schedule=_replace_branch_side(result.schedule, (unreturned,)))

    with pytest.raises(ValueError, match="does not start from the machine state reached before it"):
        broken.validate()


def test_validation_rejects_a_schedule_with_a_missing_gate() -> None:
    """Require every circuit gate in its precompiled schedule."""
    result = compile_program(_measure_then_branch(), GridCompiler(_two_zone_line()))
    branch = result.schedule.steps[1]
    assert isinstance(branch, ScheduledBranch)
    correction = branch.then_steps[0]
    assert isinstance(correction, Schedule)
    without_gate = Schedule(
        tuple(item for item in correction.scheduled_actions if not isinstance(item.action, Rx)),
        correction.end_time,
        correction.initial_state,
    )
    broken = replace(result, schedule=_replace_branch_side(result.schedule, (without_gate,)))

    with pytest.raises(ValueError, match="each circuit gate exactly once"):
        broken.validate()


def test_a_loop_body_is_compiled_once_and_returns_to_its_start() -> None:
    """Run the same body schedules in every iteration of a counted loop."""
    result = compile_program(_program("counted_loop"), GridCompiler(GRID), feedback_delay=1)

    (loop,) = result.schedule.steps
    assert isinstance(loop, ScheduledLoop)
    assert loop.count == 3
    result.validate()
    durations = [summary.duration for summary in result.path_summaries()]
    assert len(durations) == 8
    assert max(durations) > min(durations)


def test_repeat_until_success_paths_grow_by_one_body_per_iteration() -> None:
    """List paths of a loop without a count up to a requested number of iterations."""
    result = compile_program(_program("repeat_until_success"), GridCompiler(GRID), feedback_delay=1)

    result.validate()
    summaries = result.path_summaries(max_iterations=3)
    assert [len(summary.outcomes) for summary in summaries] == [1, 2, 3]
    assert [summary.duration for summary in summaries] == sorted(summary.duration for summary in summaries)
    with pytest.raises(ValueError, match="needs max_iterations"):
        result.path_summaries()


def test_validation_rejects_a_loop_body_that_ends_elsewhere() -> None:
    """Require a loop body to end in the machine state in which it starts."""
    circuit = Circuit(num_ions=2, gates=(Rx(ion=0, theta=0.5, gate_id=0),), predecessors=(frozenset(),))
    architecture = _two_zone_line()
    result = compile_program(Program(2, (Loop((circuit,), count=2),)), GridCompiler(architecture))
    (loop,) = result.schedule.steps
    assert isinstance(loop, ScheduledLoop)
    body = loop.steps[0]
    assert isinstance(body, Schedule)
    start = dict(body.initial_state.occupancy)
    elsewhere = {"memory": (0, 1)} if start["memory"] != (0, 1) else {"right": (0, 1)}
    moved = GridCompiler(architecture).compile(circuit, initial_placement=start, target_placement=elsewhere).schedule
    broken = replace(result, schedule=replace(result.schedule, steps=(replace(loop, steps=(moved,)),)))

    with pytest.raises(ValueError, match="loop body does not end in the machine state in which it starts"):
        broken.validate()


def test_loop_results_round_trip_through_json() -> None:
    """Restore loops in the program and in the branching schedule."""
    result = compile_program(_program("repeat_until_success"), GridCompiler(GRID))

    restored = ProgramResult.from_dict(
        json.loads(json.dumps(result.to_dict())),
        decode_architecture=GridArchitecture.from_dict,
        decode_action=decode_grid_action,
        decode_state=GridMachineState.from_dict,
    )

    assert restored == result
