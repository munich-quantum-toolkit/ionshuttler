# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Contract tests for minimal Grid compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from mqt.ionshuttler.core.gates import GateAction, Rx, Ry
from mqt.ionshuttler.grid import (
    CompilationStatus,
    Cycle,
    GateTiming,
    GridArchitecture,
    GridCompiler,
    GridCompilerConfig,
    GridCompilerStrategy,
    GridDependencyMode,
    GridDiagnostics,
    GridStopReason,
    Junction,
    JunctionMove,
    ProcessingZone,
    Segment,
    SegmentOccupancy,
    result_from_json,
    square_grid,
)

if TYPE_CHECKING:
    from pathlib import Path


def _qasm(num_ions: int, *instructions: str) -> str:
    return "\n".join(("OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{num_ions}];", *instructions))


@pytest.mark.parametrize("strategy", list(GridCompilerStrategy))
def test_compiler_routes_a_two_ion_gate_and_round_trips_result(strategy: GridCompilerStrategy) -> None:
    """Move a chain to a processing zone and retain the shared result contract."""
    memory = Segment("memory", capacity=2)
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (memory, processor),
        (Junction("j0", (memory.end, processor.start)),),
    )

    result = GridCompiler(architecture, GridCompilerConfig(strategy=strategy)).compile(
        _qasm(2, "rxx(0.5) q[0],q[1];"),
        initial_placement={"memory": (0, 1)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert isinstance(result.path[0], JunctionMove)
    assert isinstance(result.path[-1], GateAction)
    assert result.final_state.occupants("processor") == (0, 1)
    assert result.diagnostics is not None
    assert result.diagnostics.junction_moves == 1
    assert result.diagnostics.cycles == 0
    assert result.diagnostics.gates == 1
    result.validate()
    assert result_from_json(result.to_json()) == result


@pytest.mark.parametrize("strategy", list(GridCompilerStrategy))
def test_compiler_uses_a_simultaneous_cycle_when_every_segment_is_full(strategy: GridCompilerStrategy) -> None:
    """Rotate a full ring when no capacity-safe single move exists."""
    a = Segment("a")
    b = Segment("b", processing_zones=(ProcessingZone("pz"),))
    c = Segment("c")
    architecture = GridArchitecture(
        (a, b, c),
        (
            Junction("ab", (a.end, b.start)),
            Junction("bc", (b.end, c.start)),
            Junction("ca", (c.end, a.start)),
        ),
    )

    result = GridCompiler(architecture, GridCompilerConfig(strategy=strategy)).compile(
        _qasm(3, "rx(0.5) q[0];"),
        initial_placement={"a": (0,), "b": (1,), "c": (2,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert isinstance(result.path[0], Cycle)
    assert result.final_state.ion_segment(0) == "b"
    assert result.diagnostics is not None
    assert result.diagnostics.cycles == 1
    assert result.diagnostics.junction_moves == 3


def test_compiler_schedules_independent_ready_gates_together() -> None:
    """Use separate processing zones for independent gates in one layer."""
    left = Segment("left", processing_zones=(ProcessingZone("left-pz"),))
    right = Segment("right", processing_zones=(ProcessingZone("right-pz"),))
    architecture = GridArchitecture((left, right), ())

    result = GridCompiler(architecture).compile(
        _qasm(2, "rx(0.5) q[0];", "ry(0.25) q[1];"),
        initial_placement={"left": (0,), "right": (1,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert [item.start_time for item in result.schedule.scheduled_actions] == [0, 0]
    assert result.end_time == 1
    assert result.diagnostics is not None
    assert result.diagnostics.gates == 2


def test_greedy_compiler_clears_a_stall_for_the_selected_gate() -> None:
    """Try path clearing once after normal routing stalls."""
    source = Segment("source")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    free = Segment("free")
    architecture = GridArchitecture(
        (source, processor, free),
        (
            Junction("entry", (source.end, processor.start)),
            Junction("exit", (processor.end, free.start)),
        ),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(
            strategy=GridCompilerStrategy.GREEDY,
            max_routing_states=2,
            allowed_junction_crossings=frozenset({("source", "processor"), ("processor", "free")}),
        ),
    ).compile(
        _qasm(2, "rx(0.25) q[0];"),
        initial_placement={"source": (0,), "processor": (1,)},
        pre_partition=False,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.final_state.occupants("processor") == (0,)
    assert result.final_state.occupants("free") == (1,)


def test_greedy_compiler_suppresses_same_target_lookahead_when_selected_gate_is_blocked() -> None:
    """Do not route a later gate toward a zone whose selected gate has no candidate."""
    source = Segment("source")
    blocker = Segment("blocker")
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("pz"),))
    sink = Segment("sink")
    architecture = GridArchitecture(
        (source, blocker, processor, sink),
        (
            Junction("source-junction", (source.end, blocker.start)),
            Junction("processor-junction", (blocker.end, processor.start)),
            Junction("sink-junction", (processor.end, sink.start)),
        ),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(
            strategy=GridCompilerStrategy.GREEDY,
            max_routing_states=1,
            max_iterations=1,
            record_planning_trace=True,
        ),
    ).compile(
        _qasm(3, "rxx(0.5) q[0],q[2];", "rxx(0.25) q[0],q[1];"),
        initial_placement={"source": (0,), "blocker": (1,), "processor": (2,)},
        pre_partition=False,
    )

    assert result.diagnostics is not None
    assert result.diagnostics.planning_trace is not None
    first_decision = result.diagnostics.planning_trace[0]
    assert first_decision.selected_gates == (0,)
    assert tuple(request.ion for request in first_decision.prioritized_ions) == (0,)


def test_greedy_compiler_schedules_independent_moves_together() -> None:
    """Move ions toward independent lookahead targets in one layer."""
    left_memory = Segment("left-memory")
    left_processor = Segment("left-processor", processing_zones=(ProcessingZone("left-pz"),))
    right_memory = Segment("right-memory")
    right_processor = Segment("right-processor", processing_zones=(ProcessingZone("right-pz"),))
    architecture = GridArchitecture(
        (left_memory, left_processor, right_memory, right_processor),
        (
            Junction("left-junction", (left_memory.end, left_processor.start)),
            Junction("right-junction", (right_memory.end, right_processor.start)),
        ),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY),
    ).compile(
        _qasm(2, "rx(0.5) q[0];", "ry(0.25) q[1];"),
        initial_placement={"left-memory": (0,), "right-memory": (1,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    moves = [item for item in result.schedule.scheduled_actions if isinstance(item.action, JunctionMove)]
    gates = [item for item in result.schedule.scheduled_actions if isinstance(item.action, GateAction)]
    assert [item.start_time for item in moves] == [0, 0]
    assert [item.start_time for item in gates] == [1, 1]
    assert result.duration == 2
    assert result.diagnostics is not None
    assert result.diagnostics.junction_moves == 2


def test_greedy_compiler_prioritizes_only_ions_in_the_lookahead() -> None:
    """Do not route ions for gates beyond the configured lookahead."""
    first_source = Segment("first-source")
    second_source = Segment("second-source")
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (first_source, second_source, processor),
        (Junction("j0", (first_source.end, second_source.end, processor.start)),),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(
            greedy_lookahead_rounds=1,
            max_iterations=1,
            record_planning_trace=True,
        ),
    ).compile(
        _qasm(2, "rx(0.5) q[0];", "ry(0.25) q[1];"),
        initial_placement={"first-source": (0,), "second-source": (1,)},
        pre_partition=False,
    )

    assert result.diagnostics is not None
    planning_trace = result.diagnostics.planning_trace
    assert planning_trace is not None
    assert planning_trace[0].lookahead_gates == (0,)
    assert tuple(request.ion for request in planning_trace[0].prioritized_ions) == (0,)


def test_greedy_compiler_routes_multiple_ions_toward_one_processing_zone() -> None:
    """Advance multiple prioritized ions toward one target in the same layer."""
    left_source = Segment("left-source")
    left_corridor = Segment("left-corridor")
    right_source = Segment("right-source")
    right_corridor = Segment("right-corridor")
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (left_source, left_corridor, right_source, right_corridor, processor),
        (
            Junction("left-entry", (left_source.end, left_corridor.start)),
            Junction("left-processor", (left_corridor.end, processor.start)),
            Junction("right-entry", (right_source.end, right_corridor.start)),
            Junction("right-processor", (right_corridor.end, processor.end)),
        ),
    )
    compiler = GridCompiler(
        architecture,
        GridCompilerConfig(
            strategy=GridCompilerStrategy.GREEDY,
            allowed_junction_crossings=frozenset({
                ("left-source", "left-corridor"),
                ("left-corridor", "processor"),
                ("right-source", "right-corridor"),
                ("right-corridor", "processor"),
            }),
        ),
    )

    result = compiler.compile(
        _qasm(2, "rxx(0.5) q[0],q[1];"),
        initial_placement={"left-source": (0,), "right-source": (1,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    moves = [item for item in result.schedule.scheduled_actions if isinstance(item.action, JunctionMove)]
    assert [item.start_time for item in moves] == [0, 0, 1, 1]
    assert result.final_state.occupants("processor") == (0, 1)


def test_greedy_compiler_retains_a_parked_near_term_gate_ion() -> None:
    """Evict an unrelated ion before one needed by the next gate."""
    source = Segment("source")
    processor = Segment(
        "processor",
        capacity=2,
        occupancy=SegmentOccupancy.UNORDERED,
        processing_zones=(ProcessingZone("pz"),),
    )
    free = Segment("free")
    architecture = GridArchitecture(
        (source, processor, free),
        (
            Junction("entry", (source.end, processor.start)),
            Junction("exit", (processor.end, free.start)),
        ),
    )
    compiler = GridCompiler(
        architecture,
        GridCompilerConfig(
            strategy=GridCompilerStrategy.GREEDY,
            allowed_junction_crossings=frozenset({("source", "processor"), ("processor", "free")}),
        ),
    )

    result = compiler.compile(
        _qasm(3, "rx(0.5) q[0];", "rxx(0.25) q[0],q[1];"),
        initial_placement={"source": (0,), "processor": (1, 2)},
    )

    assert result.status is CompilationStatus.SUCCESS
    first_moves = tuple(
        item.action
        for item in result.schedule.scheduled_actions
        if item.start_time == 0 and isinstance(item.action, JunctionMove)
    )
    assert any(move.ions == (2,) and move.destination.segment_id == "free" for move in first_moves)
    assert all(move.ions != (1,) for move in first_moves)


def test_greedy_compiler_prefers_the_gate_with_lower_total_distance() -> None:
    """Prefer the tied gate that already has one operand at the processor."""
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("pz"),))
    source_a = Segment("source-a")
    source_b = Segment("source-b")
    source_c = Segment("source-c")
    architecture = GridArchitecture(
        (processor, source_a, source_b, source_c),
        (Junction("entry", (processor.start, source_a.end, source_b.end, source_c.end)),),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY, max_iterations=4),
    ).compile(
        _qasm(4, "rxx(0.5) q[2],q[3];", "rxx(0.25) q[0],q[1];"),
        initial_placement={"processor": (0,), "source-a": (1,), "source-b": (2,), "source-c": (3,)},
    )

    first_gate = next(item.action for item in result.schedule.scheduled_actions if isinstance(item.action, GateAction))
    assert first_gate.gate_id == 1


def test_greedy_compiler_routes_while_an_independent_gate_remains_active() -> None:
    """Use unrelated transport resources during a multi-timestep gate."""
    left_processor = Segment("left-processor", processing_zones=(ProcessingZone("left-pz"),))
    right_source = Segment("right-source")
    right_corridor = Segment("right-corridor")
    right_processor = Segment("right-processor", processing_zones=(ProcessingZone("right-pz"),))
    architecture = GridArchitecture(
        (left_processor, right_source, right_corridor, right_processor),
        (
            Junction("right-entry", (right_source.end, right_corridor.start)),
            Junction("right-processor", (right_corridor.end, right_processor.start)),
        ),
        gate_timing=GateTiming(rx=3),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY),
    ).compile(
        _qasm(2, "rx(0.5) q[0];", "rx(0.25) q[1];"),
        initial_placement={"left-processor": (0,), "right-source": (1,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    moves = [item for item in result.schedule.scheduled_actions if isinstance(item.action, JunctionMove)]
    gates = [item for item in result.schedule.scheduled_actions if isinstance(item.action, GateAction)]
    assert [item.start_time for item in moves] == [0, 1]
    assert [item.start_time for item in gates] == [0, 2]
    assert result.duration == 5


def test_greedy_compiler_routes_for_a_dependent_gate_while_its_predecessor_runs() -> None:
    """Prepare a dependent gate on resources not used by its running predecessor."""
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("pz"),))
    source = Segment("source")
    corridor = Segment("corridor")
    architecture = GridArchitecture(
        (processor, source, corridor),
        (
            Junction("source-junction", (source.end, corridor.start)),
            Junction("processor-junction", (corridor.end, processor.start)),
        ),
        gate_timing=GateTiming(rx=3),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY),
    ).compile(
        _qasm(2, "rx(0.5) q[0];", "rxx(0.25) q[0],q[1];"),
        initial_placement={"processor": (0,), "source": (1,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    moves = [item for item in result.schedule.scheduled_actions if isinstance(item.action, JunctionMove)]
    gates = [item for item in result.schedule.scheduled_actions if isinstance(item.action, GateAction)]
    assert [item.start_time for item in moves] == [0, 1]
    assert [item.start_time for item in gates] == [0, 3]


def test_greedy_compiler_allows_a_spectator_to_leave_during_a_gate() -> None:
    """Retain gate operands without retaining other ions on the PZ segment."""
    source = Segment("source")
    processor = Segment(
        "processor",
        capacity=2,
        occupancy=SegmentOccupancy.UNORDERED,
        processing_zones=(ProcessingZone("pz"),),
    )
    sink = Segment("sink")
    architecture = GridArchitecture(
        (source, processor, sink),
        (
            Junction("entry", (source.end, processor.start)),
            Junction("exit", (processor.end, sink.start)),
        ),
        gate_timing=GateTiming(rx=3),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY),
    ).compile(
        _qasm(3, "rx(0.5) q[0];", "rxx(0.25) q[0],q[2];"),
        initial_placement={"source": (2,), "processor": (0, 1)},
    )

    assert result.status is CompilationStatus.SUCCESS
    moves = [item for item in result.schedule.scheduled_actions if isinstance(item.action, JunctionMove)]
    assert moves[0].start_time == 0
    first_move = moves[0].action
    second_move = moves[1].action
    assert isinstance(first_move, JunctionMove)
    assert isinstance(second_move, JunctionMove)
    assert first_move.ions == (2,)
    assert second_move.ions == (1,)


def test_greedy_compiler_does_not_displace_a_higher_priority_ion_away_from_its_target() -> None:
    """Let a lower-priority ion wait instead of reversing a higher-priority route."""
    low_source = Segment("low-source")
    corridor = Segment("corridor")
    escape = Segment("escape")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    sink = Segment("sink")
    architecture = GridArchitecture(
        (low_source, corridor, escape, processor, sink),
        (
            Junction("low-entry", (low_source.end, corridor.start)),
            Junction("branch", (corridor.end, escape.start, processor.start)),
            Junction("sink", (processor.end, sink.start)),
        ),
        gate_timing=GateTiming(rx=3),
    )
    compiler = GridCompiler(
        architecture,
        GridCompilerConfig(
            strategy=GridCompilerStrategy.GREEDY,
            allowed_junction_crossings=frozenset({
                ("low-source", "corridor"),
                ("corridor", "escape"),
                ("corridor", "processor"),
                ("processor", "sink"),
            }),
            max_iterations=1,
        ),
    )

    result = compiler.compile(
        _qasm(3, "rx(0.5) q[2];", "rx(0.25) q[0];", "rx(0.125) q[1];"),
        initial_placement={"low-source": (1,), "corridor": (0,), "processor": (2,)},
    )

    assert [item.action for item in result.schedule.scheduled_actions if isinstance(item.action, JunctionMove)] == []


def test_greedy_compiler_finishes_a_started_gate_at_the_iteration_bound() -> None:
    """Include the full duration of a gate that starts on the last iteration."""
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture((processor,), (), gate_timing=GateTiming(rx=3))

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY, max_iterations=1),
    ).compile(
        _qasm(1, "rx(0.5) q[0];"),
        initial_placement={"processor": (0,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.duration == 3
    assert result.diagnostics is not None
    assert result.diagnostics.gates == 1


def test_greedy_compiler_schedules_an_independent_gate_and_move_together() -> None:
    """Execute a gate while another ion crosses an unrelated junction."""
    left_processor = Segment("left-processor", processing_zones=(ProcessingZone("left-pz"),))
    right_memory = Segment("right-memory")
    right_processor = Segment("right-processor", processing_zones=(ProcessingZone("right-pz"),))
    architecture = GridArchitecture(
        (left_processor, right_memory, right_processor),
        (Junction("right-junction", (right_memory.end, right_processor.start)),),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY),
    ).compile(
        _qasm(2, "rx(0.5) q[0];", "ry(0.25) q[1];"),
        initial_placement={"left-processor": (0,), "right-memory": (1,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    first_layer = tuple(item for item in result.schedule.scheduled_actions if item.start_time == 0)
    assert len(first_layer) == 2
    assert sum(isinstance(item.action, GateAction) for item in first_layer) == 1
    assert sum(isinstance(item.action, JunctionMove) for item in first_layer) == 1
    assert result.duration == 2


def test_greedy_compiler_does_not_move_an_unrelated_ion() -> None:
    """Fail a blocked route instead of moving an ion that cannot improve it."""
    source = Segment("source")
    isolated_target = Segment("isolated-target", processing_zones=(ProcessingZone("target-pz"),))
    unrelated_source = Segment("unrelated-source")
    unrelated_destination = Segment("unrelated-destination")
    architecture = GridArchitecture(
        (source, isolated_target, unrelated_source, unrelated_destination),
        (
            Junction("target-junction", (source.end, isolated_target.start)),
            Junction("unrelated-junction", (unrelated_source.end, unrelated_destination.start)),
        ),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY),
    ).compile(
        _qasm(3, "rx(0.5) q[0];"),
        initial_placement={"source": (0,), "isolated-target": (1,), "unrelated-source": (2,)},
    )

    assert result.status is CompilationStatus.FAILED
    assert result.path == []


def test_greedy_compiler_honors_allowed_junction_crossings() -> None:
    """Keep a routing policy from using the reverse direction of a junction."""
    source = Segment("source")
    target = Segment("target", processing_zones=(ProcessingZone("target-pz"),))
    architecture = GridArchitecture((source, target), (Junction("junction", (source.end, target.start)),))
    compiler = GridCompiler(
        architecture,
        GridCompilerConfig(
            strategy=GridCompilerStrategy.GREEDY,
            allowed_junction_crossings=frozenset({("target", "source")}),
        ),
    )

    with pytest.raises(ValueError, match="no processing-zone segment is reachable"):
        compiler.compile(
            _qasm(1, "rx(0.5) q[0];"),
            initial_placement={"source": (0,)},
        )


def test_greedy_compiler_returns_no_progress_if_a_later_gate_becomes_unreachable() -> None:
    """Stop cleanly if earlier routing leaves a later gate without a reachable zone."""
    source = Segment("source")
    rx_processor = Segment(
        "rx-processor",
        processing_zones=(ProcessingZone("rx-pz", supported_gate_types=(Rx,)),),
    )
    ry_processor = Segment(
        "ry-processor",
        processing_zones=(ProcessingZone("ry-pz", supported_gate_types=(Ry,)),),
    )
    architecture = GridArchitecture(
        (source, rx_processor, ry_processor),
        (Junction("junction", (source.end, rx_processor.start, ry_processor.start)),),
    )
    compiler = GridCompiler(
        architecture,
        GridCompilerConfig(
            allowed_junction_crossings=frozenset({("source", "rx-processor"), ("source", "ry-processor")}),
        ),
    )

    result = compiler.compile(
        _qasm(1, "rx(0.5) q[0];", "ry(0.25) q[0];"),
        initial_placement={"source": (0,)},
        pre_partition=False,
    )

    assert result.status is CompilationStatus.FAILED
    assert result.diagnostics is not None
    assert result.diagnostics.gates == 1
    assert result.diagnostics.stop_reason is GridStopReason.NO_PROGRESS
    result.validate()


def test_greedy_compiler_ignores_a_processing_zone_on_an_undersized_segment() -> None:
    """Route a multi-ion gate only to a segment that can contain all operands."""
    source = Segment("source", capacity=2)
    undersized = Segment("undersized", processing_zones=(ProcessingZone("undersized-pz"),))
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("processor-pz"),))
    architecture = GridArchitecture(
        (source, undersized, processor),
        (Junction("junction", (source.end, undersized.start, processor.start)),),
    )

    result = GridCompiler(architecture).compile(
        _qasm(2, "rxx(0.5) q[0],q[1];"),
        initial_placement={"source": (0, 1)},
        pre_partition=False,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.schedule.scheduled_actions[-1].processing_zone_id == "processor-pz"


def test_greedy_compiler_clears_a_blocked_route_from_the_free_end() -> None:
    """Shift downstream blockers before admitting an ion through a busy junction."""
    source = Segment("source")
    target = Segment("target", processing_zones=(ProcessingZone("target-pz"),))
    corridor = Segment("corridor")
    free = Segment("free")
    architecture = GridArchitecture(
        (source, target, corridor, free),
        (
            Junction("target-junction", (source.end, target.start, corridor.start)),
            Junction("corridor-junction", (corridor.end, free.start)),
        ),
    )
    compiler = GridCompiler(
        architecture,
        GridCompilerConfig(
            strategy=GridCompilerStrategy.GREEDY,
            allowed_junction_crossings=frozenset({("source", "target"), ("target", "corridor"), ("corridor", "free")}),
        ),
    )

    result = compiler.compile(
        _qasm(3, "rx(0.5) q[0];"),
        initial_placement={"source": (0,), "target": (1,), "corridor": (2,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    moves = [item for item in result.schedule.scheduled_actions if isinstance(item.action, JunctionMove)]
    assert [item.start_time for item in moves] == [0, 0, 1]
    assert result.duration == 3


def test_greedy_compiler_moves_a_blocking_ion_aside_at_the_exit_junction() -> None:
    """Let the ion in front step into a side segment so that the gate operand can leave."""
    source = Segment("source", capacity=2)
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    side = Segment("side")
    architecture = GridArchitecture(
        (source, processor, side),
        (Junction("exit", (source.end, processor.start, side.start)),),
    )

    result = GridCompiler(architecture, GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY)).compile(
        _qasm(2, "rx(0.5) q[0];"),
        initial_placement={"source": (0, 1)},
        pre_partition=False,
    )

    assert result.status is CompilationStatus.SUCCESS
    moves = [item.action for item in result.schedule.scheduled_actions if isinstance(item.action, JunctionMove)]
    assert [(move.ions, move.destination.segment_id) for move in moves] == [((1,), "side"), ((0,), "processor")]
    result.validate()


def test_greedy_compiler_completes_qft_on_the_four_zone_square_grid(qasm_file_qft6: Path) -> None:
    """Complete a benchmark in which ions in front of gate operands block their exit junctions."""
    architecture = square_grid(4, processing_zone_segments=("h:0:1", "v:1:0", "h:3:1", "v:1:3"), segment_capacity=2)

    result = GridCompiler(architecture).compile(qasm_file_qft6)

    assert result.status is CompilationStatus.SUCCESS
    result.validate()


def test_greedy_compiler_path_clearing_does_not_undo_a_higher_priority_move(qasm_file_qft6: Path) -> None:
    """Complete a benchmark in which clearing for a lower-priority gate would reverse a higher-priority chain move."""
    architecture = square_grid(5, processing_zone_segments=("h:1:1", "h:3:2"), segment_capacity=4)

    result = GridCompiler(architecture, GridCompilerConfig(max_iterations=1_000)).compile(qasm_file_qft6)

    assert result.status is CompilationStatus.SUCCESS
    result.validate()


def test_greedy_compiler_rotates_a_sparse_topology_cycle() -> None:
    """Move occupied cycle segments together without requiring a full cycle."""
    source = Segment("source")
    target = Segment("target", processing_zones=(ProcessingZone("target-pz"),))
    empty = Segment("empty")
    occupied = Segment("occupied")
    architecture = GridArchitecture(
        (source, target, empty, occupied),
        (
            Junction("source-target", (source.end, target.start)),
            Junction("target-empty", (target.end, empty.start)),
            Junction("empty-occupied", (empty.end, occupied.start)),
            Junction("occupied-source", (occupied.end, source.start)),
        ),
    )
    compiler = GridCompiler(
        architecture,
        GridCompilerConfig(
            strategy=GridCompilerStrategy.GREEDY,
            allowed_junction_crossings=frozenset({
                ("source", "target"),
                ("target", "empty"),
                ("empty", "occupied"),
                ("occupied", "source"),
            }),
        ),
    )

    result = compiler.compile(
        _qasm(3, "rx(0.5) q[0];"),
        initial_placement={"source": (0,), "target": (1,), "occupied": (2,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    moves = [item for item in result.schedule.scheduled_actions if isinstance(item.action, JunctionMove)]
    assert len(moves) == 3
    assert {item.start_time for item in moves} == {0}
    assert result.final_state.ion_segment(0) == "target"
    assert result.duration == 2


def test_greedy_compiler_routes_to_the_nearest_processing_zone() -> None:
    """Choose the nearest reachable processing zone that supports the gate."""
    source = Segment("source")
    near = Segment("near", processing_zones=(ProcessingZone("near-pz"),))
    far = Segment("far", processing_zones=(ProcessingZone("far-pz"),))
    architecture = GridArchitecture(
        (source, near, far),
        (
            Junction("near-junction", (source.end, near.start)),
            Junction("far-junction", (near.end, far.start)),
        ),
    )

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY),
    ).compile(
        _qasm(1, "rx(0.5) q[0];"),
        initial_placement={"source": (0,)},
        pre_partition=False,
    )

    assert result.status is CompilationStatus.SUCCESS
    assert isinstance(result.path[0], JunctionMove)
    assert result.schedule.scheduled_actions[-1].processing_zone_id == "near-pz"


@pytest.mark.parametrize("strategy", list(GridCompilerStrategy))
def test_explicit_gate_assignment_replaces_pre_partitioning(strategy: GridCompilerStrategy) -> None:
    """Use an explicit assignment instead of the enabled automatic partitioning."""
    source = Segment("source")
    near = Segment("near", processing_zones=(ProcessingZone("near-pz"),))
    far = Segment("far", processing_zones=(ProcessingZone("far-pz"),))
    architecture = GridArchitecture(
        (source, near, far),
        (
            Junction("near-junction", (source.end, near.start)),
            Junction("far-junction", (near.end, far.start)),
        ),
    )

    result = GridCompiler(architecture, GridCompilerConfig(strategy=strategy)).compile(
        _qasm(1, "rx(0.5) q[0];"),
        initial_placement={"source": (0,)},
        gate_to_processing_zone={0: "far-pz"},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.schedule.scheduled_actions[-1].processing_zone_id == "far-pz"


def test_compiler_can_compute_a_gate_zone_assignment() -> None:
    """Use the shared partitioner to assign physical gates before scheduling."""
    left = Segment("left", capacity=2, processing_zones=(ProcessingZone("left-pz"),))
    right = Segment("right", capacity=2, processing_zones=(ProcessingZone("right-pz"),))
    architecture = GridArchitecture(
        (left, right),
        (Junction("middle", (left.end, right.start)),),
    )
    compiler = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY),
    )

    result = compiler.compile(
        _qasm(2, "rx(0.5) q[0];", "ry(0.25) q[1];", "rxx(0.5) q[0],q[1];"),
        initial_placement={"left": (0,), "right": (1,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.diagnostics is not None
    assert result.diagnostics.gates == 3


def test_compiler_requires_a_boolean_pre_partition_switch() -> None:
    """Reject non-Boolean pre-partition switches at the public boundary."""
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))

    with pytest.raises(TypeError, match="pre_partition must be a boolean"):
        GridCompiler(GridArchitecture((processor,), ())).compile(
            _qasm(1, "rx(0.5) q[0];"),
            initial_placement={"processor": (0,)},
            pre_partition=cast("bool", 1),
        )


def test_greedy_compiler_skips_an_incompatible_processing_zone() -> None:
    """Route past the current segment when its processing zone rejects the gate."""
    incompatible = Segment(
        "incompatible",
        processing_zones=(ProcessingZone("incompatible-pz", supported_gate_types=()),),
    )
    processor = Segment("processor", processing_zones=(ProcessingZone("processor-pz"),))
    architecture = GridArchitecture(
        (incompatible, processor),
        (Junction("j0", (incompatible.end, processor.start)),),
    )

    result = GridCompiler(architecture, GridCompilerConfig(strategy=GridCompilerStrategy.GREEDY)).compile(
        _qasm(1, "rx(0.5) q[0];"),
        initial_placement={"incompatible": (0,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert isinstance(result.path[0], JunctionMove)
    assert result.schedule.scheduled_actions[-1].processing_zone_id == "processor-pz"


def test_compiler_preserves_gate_dependencies() -> None:
    """Start two gates on one ion in circuit order and on separate ticks."""
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture((processor,), ())

    result = GridCompiler(architecture).compile(
        _qasm(1, "rx(0.5) q[0];", "ry(0.25) q[0];"),
        initial_placement={"processor": (0,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert all(isinstance(action, GateAction) for action in result.path)
    assert [action.gate_id for action in result.path if isinstance(action, GateAction)] == [0, 1]
    assert [item.start_time for item in result.schedule.scheduled_actions] == [0, 1]


def test_compiler_executes_virtual_gates_without_a_processing_zone() -> None:
    """Keep virtual single-ion gates independent of physical zone placement."""
    architecture = GridArchitecture((Segment("memory"),), ())

    result = GridCompiler(architecture).compile(
        _qasm(1, "rz(0.5) q[0];"),
        initial_placement={"memory": (0,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.end_time == 0
    assert result.schedule.scheduled_actions[0].processing_zone_id is None


def test_compiler_returns_a_valid_failed_prefix_when_the_zone_is_unreachable() -> None:
    """Stop within the routing bound and return the replayable empty prefix."""
    memory = Segment("memory")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture((memory, processor), ())

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.BREADTH_FIRST, max_routing_states=2),
    ).compile(
        _qasm(1, "rx(0.5) q[0];"),
        initial_placement={"memory": (0,)},
    )

    assert result.status is CompilationStatus.FAILED
    assert result.path == []
    assert result.final_state == result.initial_state
    assert result.diagnostics == GridDiagnostics(1, 1, 0, 0, 0, GridStopReason.NO_PROGRESS)
    result.validate()


def test_grid_diagnostics_report_the_run_counters() -> None:
    """Name each counter of the scheduling run for people."""
    diagnostics = GridDiagnostics(4, 9, 2, 1, 3, GridStopReason.COMPLETED)

    assert diagnostics.statistics() == (
        ("Iterations", 4),
        ("Explored routing states", 9),
        ("Junction moves", 2),
        ("Cycles", 1),
        ("Completed gates", 3),
    )


def test_compiler_uses_deterministic_default_placement() -> None:
    """Fill segments in architecture order when no placement is supplied."""
    first = Segment("first")
    second = Segment("second", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (first, second),
        (Junction("j0", (first.end, second.start)),),
    )

    result = GridCompiler(architecture).compile(_qasm(1, "rx(0.5) q[0];"))

    assert result.status is CompilationStatus.SUCCESS
    assert result.initial_state.occupants("first") == (0,)


@pytest.mark.parametrize(
    "field",
    ["max_iterations", "max_routing_states", "greedy_lookahead_rounds"],
)
def test_compiler_config_requires_positive_integer_bounds(field: str) -> None:
    """Reject routing limits that cannot permit compiler progress."""
    constructors = {
        "max_iterations": lambda: GridCompilerConfig(max_iterations=0),
        "max_routing_states": lambda: GridCompilerConfig(max_routing_states=0),
        "greedy_lookahead_rounds": lambda: GridCompilerConfig(greedy_lookahead_rounds=0),
    }

    with pytest.raises(ValueError, match=rf"{field} must be an integer >= 1"):
        constructors[field]()


def test_compiler_config_requires_strategy_enum_values() -> None:
    """Reject raw strings so configuration choices remain explicit."""
    with pytest.raises(TypeError, match="strategy must be a GridCompilerStrategy"):
        GridCompilerConfig(strategy=cast("GridCompilerStrategy", "greedy"))


def test_compiler_config_requires_dependency_mode_enum_values() -> None:
    """Reject raw strings so dependency choices remain explicit."""
    with pytest.raises(TypeError, match="dependency_mode must be a GridDependencyMode"):
        GridCompilerConfig(dependency_mode=cast("GridDependencyMode", "commutation_aware"))


def test_compiler_config_uses_recommended_defaults() -> None:
    """Use the scalable routing and dependency modes unless configured otherwise."""
    config = GridCompilerConfig()

    assert config.strategy is GridCompilerStrategy.GREEDY
    assert config.dependency_mode is GridDependencyMode.COMMUTATION_AWARE


def test_compiler_config_rejects_malformed_junction_crossings() -> None:
    """Require each permitted junction crossing to name two segments."""
    crossings = frozenset({("source", "")})

    with pytest.raises(TypeError, match="allowed_junction_crossings must contain pairs of segment identifiers"):
        GridCompilerConfig(allowed_junction_crossings=crossings)


def test_compiler_rejects_an_incomplete_initial_placement() -> None:
    """Require one placement entry for every circuit ion."""
    architecture = GridArchitecture((Segment("memory", capacity=2),), ())

    with pytest.raises(ValueError, match="initial placement must contain every circuit ion exactly once"):
        GridCompiler(architecture).compile(_qasm(2, "rz(0.5) q[0];"), initial_placement={"memory": (0,)})


def test_compiler_rejects_a_gate_without_a_capable_processing_zone() -> None:
    """Report missing hardware capability before routing starts."""
    architecture = GridArchitecture((Segment("memory"),), ())

    with pytest.raises(ValueError, match="Grid architecture has no processing zone for Rx"):
        GridCompiler(architecture).compile(_qasm(1, "rx(0.5) q[0];"))


def test_compiler_rejects_a_gate_if_every_processing_zone_segment_is_too_small() -> None:
    """Require one processing-zone segment to contain every gate operand."""
    memory = Segment("memory", capacity=2)
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (memory, processor),
        (Junction("junction", (memory.end, processor.start)),),
    )

    with pytest.raises(ValueError, match="Grid architecture has no processing zone for Rxx"):
        GridCompiler(architecture).compile(
            _qasm(2, "rxx(0.5) q[0],q[1];"),
            initial_placement={"memory": (0, 1)},
        )


def test_compiler_rejects_an_explicit_gate_assignment_to_an_undersized_segment() -> None:
    """Reject an explicit target that cannot contain all gate operands."""
    memory = Segment("memory", capacity=2)
    undersized = Segment("undersized", processing_zones=(ProcessingZone("undersized-pz"),))
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("processor-pz"),))
    architecture = GridArchitecture(
        (memory, undersized, processor),
        (Junction("junction", (memory.end, undersized.start, processor.start)),),
    )

    with pytest.raises(ValueError, match="processing zone 'undersized-pz' cannot accommodate gate 0"):
        GridCompiler(architecture).compile(
            _qasm(2, "rxx(0.5) q[0],q[1];"),
            initial_placement={"memory": (0, 1)},
            gate_to_processing_zone={0: "undersized-pz"},
        )
