# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the optional Grid compiler planning trace and the stop reason."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from mqt.ionshuttler.core.gates import Rx, Rz
from mqt.ionshuttler.grid import (
    CompilationStatus,
    GridArchitecture,
    GridCompilationResult,
    GridCompiler,
    GridCompilerConfig,
    GridCompilerStrategy,
    GridDiagnostics,
    GridStopReason,
    Junction,
    JunctionMove,
    ProcessingZone,
    Segment,
    result_from_json,
)
from mqt.ionshuttler.grid.planning_trace import (
    IdleReason,
    MoveStatus,
    PlannedGate,
    PlanningSnapshot,
    PrioritizedIon,
)

_MEMORY = Segment("memory")
_PROCESSOR = Segment("processor", processing_zones=(ProcessingZone("pz"),))
_ARCHITECTURE = GridArchitecture((_MEMORY, _PROCESSOR), (Junction("j0", (_MEMORY.end, _PROCESSOR.start)),))
_ENTER = JunctionMove(_MEMORY.end, _PROCESSOR.start, (0,))
_RX = Rx(0, 0.5, gate_id=0)
_RZ = Rz(0, 0.5, gate_id=1)


def _qasm(num_ions: int, *instructions: str) -> str:
    return "\n".join(("OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{num_ions}];", *instructions))


def _compile(config: GridCompilerConfig) -> GridCompilationResult:
    return GridCompiler(_ARCHITECTURE, config).compile(
        _qasm(1, "rx(0.5) q[0];", "rz(0.5) q[0];"),
        initial_placement={"memory": (0,)},
    )


def _diagnostics(result: GridCompilationResult) -> GridDiagnostics:
    assert result.diagnostics is not None
    return result.diagnostics


@pytest.mark.parametrize("strategy", list(GridCompilerStrategy))
def test_planning_trace_is_off_by_default(strategy: GridCompilerStrategy) -> None:
    """Record no trace and write no trace data unless the caller asks for it."""
    result = GridCompiler(_ARCHITECTURE, GridCompilerConfig(strategy=strategy)).compile(
        _qasm(1, "rx(0.5) q[0];"),
        initial_placement={"memory": (0,)},
    )

    assert GridCompilerConfig().record_planning_trace is False
    assert result.diagnostics is not None
    assert result.diagnostics.planning_trace is None
    assert "planning_trace" not in result.diagnostics.to_dict()


@pytest.mark.parametrize("strategy", list(GridCompilerStrategy))
def test_planning_trace_does_not_change_the_schedule(strategy: GridCompilerStrategy) -> None:
    """Keep the schedule the record of the actions that happened."""
    plain = _compile(GridCompilerConfig(strategy=strategy))
    traced = _compile(GridCompilerConfig(strategy=strategy, record_planning_trace=True))

    assert traced.schedule == plain.schedule
    assert replace(_diagnostics(traced), planning_trace=None) == _diagnostics(plain)


def test_greedy_planning_trace_records_each_decision() -> None:
    """Record the gates, targets, ion requests, and moves of every greedy decision."""
    trace = _diagnostics(_compile(GridCompilerConfig(record_planning_trace=True))).planning_trace

    assert trace == (
        PlanningSnapshot(
            iteration=1,
            time=0,
            ready_gates=(0,),
            selected_gates=(0,),
            lookahead_gates=(0, 1),
            started_gates=(),
            gates=(PlannedGate(_RX, "processor"), PlannedGate(_RZ, None)),
            prioritized_ions=(PrioritizedIon(0, "processor", MoveStatus.SELECTED, (_ENTER,)),),
            retained_ions=(),
            moves=(_ENTER,),
        ),
        PlanningSnapshot(
            iteration=2,
            time=1,
            ready_gates=(0,),
            selected_gates=(0,),
            lookahead_gates=(0, 1),
            started_gates=(0,),
            gates=(PlannedGate(_RX, "processor"), PlannedGate(_RZ, None)),
            prioritized_ions=(PrioritizedIon(0, "processor", MoveStatus.IN_GATE),),
            retained_ions=(0,),
            moves=(),
        ),
        PlanningSnapshot(
            iteration=3,
            time=2,
            ready_gates=(1,),
            selected_gates=(1,),
            lookahead_gates=(1,),
            started_gates=(1,),
            gates=(PlannedGate(_RZ, None),),
            prioritized_ions=(),
            retained_ions=(0,),
            moves=(),
        ),
    )
    assert trace is not None
    assert [planned.gate_id for planned in trace[0].gates] == [0, 1]


def test_breadth_first_planning_trace_records_routes_and_started_gates() -> None:
    """Record the routed gate with its complete route, then the gates that start."""
    trace = _diagnostics(
        _compile(GridCompilerConfig(strategy=GridCompilerStrategy.BREADTH_FIRST, record_planning_trace=True))
    ).planning_trace

    assert trace is not None
    assert [(snapshot.iteration, snapshot.time) for snapshot in trace] == [(1, 0), (2, 1), (3, 2)]
    assert trace[0].selected_gates == (0,)
    assert trace[0].started_gates == ()
    assert trace[0].gates == (PlannedGate(_RX, "processor"),)
    assert trace[0].moves == (_ENTER,)
    assert [snapshot.started_gates for snapshot in trace[1:]] == [(0,), (1,)]
    assert all(snapshot.idle_reason is None for snapshot in trace)


@pytest.mark.parametrize("strategy", list(GridCompilerStrategy))
def test_stop_reason_reports_completed_compilations(strategy: GridCompilerStrategy) -> None:
    """Report that every gate completed."""
    result = _compile(GridCompilerConfig(strategy=strategy))

    assert result.status is CompilationStatus.SUCCESS
    assert _diagnostics(result).stop_reason is GridStopReason.COMPLETED


def test_stop_reason_reports_a_greedy_deadlock() -> None:
    """Stop when no action is legal and no busy resource becomes free."""
    source = Segment("source")
    target = Segment("target", processing_zones=(ProcessingZone("target-pz"),))
    architecture = GridArchitecture((source, target), (Junction("junction", (source.end, target.start)),))

    result = GridCompiler(architecture, GridCompilerConfig(record_planning_trace=True)).compile(
        _qasm(2, "rx(0.5) q[0];"),
        initial_placement={"source": (0,), "target": (1,)},
    )

    diagnostics = _diagnostics(result)
    assert result.status is CompilationStatus.FAILED
    assert diagnostics.stop_reason is GridStopReason.NO_PROGRESS
    assert diagnostics.planning_trace is not None
    last = diagnostics.planning_trace[-1]
    assert last.idle_reason is IdleReason.NO_LEGAL_ACTION
    assert last.moves == ()
    assert last.prioritized_ions == (PrioritizedIon(0, "target", MoveStatus.NO_CANDIDATE),)


def test_stop_reason_reports_a_failed_route_search() -> None:
    """Stop the breadth-first strategy when its bounded search finds no route."""
    memory = Segment("memory")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture((memory, processor), ())

    result = GridCompiler(
        architecture,
        GridCompilerConfig(strategy=GridCompilerStrategy.BREADTH_FIRST, record_planning_trace=True),
    ).compile(_qasm(1, "rx(0.5) q[0];"), initial_placement={"memory": (0,)})

    diagnostics = _diagnostics(result)
    assert diagnostics.stop_reason is GridStopReason.NO_PROGRESS
    assert diagnostics.planning_trace is not None
    assert [snapshot.idle_reason for snapshot in diagnostics.planning_trace] == [IdleReason.NO_ROUTE]
    assert diagnostics.planning_trace[0].gates == (PlannedGate(_RX, None),)


@pytest.mark.parametrize("strategy", list(GridCompilerStrategy))
def test_stop_reason_reports_the_iteration_limit(strategy: GridCompilerStrategy) -> None:
    """Stop after ``max_iterations`` decisions while gates remain."""
    result = _compile(GridCompilerConfig(strategy=strategy, max_iterations=1, record_planning_trace=True))

    diagnostics = _diagnostics(result)
    assert result.status is CompilationStatus.FAILED
    assert diagnostics.stop_reason is GridStopReason.ITERATION_LIMIT
    assert diagnostics.planning_trace is not None
    assert [snapshot.iteration for snapshot in diagnostics.planning_trace] == [1]


def test_planning_trace_round_trips_through_result_json() -> None:
    """Save and load the trace with the rest of the result."""
    result = GridCompiler(_ARCHITECTURE, GridCompilerConfig(record_planning_trace=True)).compile(
        _qasm(1, "rx(0.5) q[0];", "rz(0.5) q[0];"),
        initial_placement={"memory": (0,)},
    )

    restored = result_from_json(result.to_json())

    assert restored == result
    assert restored.diagnostics is not None
    assert result.diagnostics is not None
    assert restored.diagnostics.planning_trace == result.diagnostics.planning_trace
    assert restored.diagnostics.stop_reason is GridStopReason.COMPLETED


def test_grid_diagnostics_reject_an_unknown_stop_reason() -> None:
    """Name the field when saved diagnostics contain an unknown stop reason."""
    data = GridDiagnostics(1, 0, 0, 0, 0, GridStopReason.COMPLETED).to_dict()
    data["stop_reason"] = "finished"

    with pytest.raises(ValueError, match="stop_reason has unknown value 'finished'"):
        GridDiagnostics.from_dict(data)


def test_compiler_config_requires_a_boolean_trace_switch() -> None:
    """Reject trace switches that are not booleans."""
    with pytest.raises(TypeError, match="record_planning_trace must be a boolean"):
        GridCompilerConfig(record_planning_trace=cast("bool", 1))
