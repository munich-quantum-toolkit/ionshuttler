# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end tests for the Linear compiler facade."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest
from qiskit import QuantumCircuit

import mqt.ionshuttler.linear.compiler as compiler_module
import mqt.ionshuttler.linear.search as search_module
from mqt.ionshuttler.circuit import parse_circuit
from mqt.ionshuttler.linear import (
    DEFAULT_ACTION_TYPES,
    GateTiming,
    LinearArchitecture,
    LinearCompiler,
    result_from_json,
)
from mqt.ionshuttler.linear.actions import (
    Action,
    GateAction,
    PhysicalSwap,
    Rx,
    Rxx,
    Ry,
    Rz,
    Rzz,
    Shuttle,
)
from mqt.ionshuttler.linear.config import LinearCompilerConfig, SearchConfig
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.schedule import schedule_from_path
from mqt.ionshuttler.linear.state import create_initial_state

if TYPE_CHECKING:
    from pathlib import Path


def test_compiler_produces_a_compact_replayable_schedule() -> None:
    """Compile dependent gates without adding idle time after work completes."""
    architecture = LinearArchitecture(num_sites=9, processing_zones={"pz1": [2, 3], "pz2": [5, 6]})
    compiler = LinearCompiler(architecture)
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrx(0.1) q[0];\nry(0.2) q[1];\nrzz(0.3) q[0], q[1];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == [
        Shuttle(ion=0, src=3, dst=2),
        Shuttle(ion=1, src=4, dst=3),
        Rx(ion=0, theta=0.1),
        Ry(ion=1, theta=0.2),
        Rzz(ion_a=0, ion_b=1, theta=0.3),
    ]
    assert [item.start_time for item in result.schedule.scheduled_actions] == [0, 0, 1, 2, 3]
    assert result.architecture.supported_action_types == DEFAULT_ACTION_TYPES
    assert result.final_state is not None
    assert result.final_state.time == 5
    result.validate()


def test_pre_partition_is_a_strict_single_zone_no_op() -> None:
    """Produce identical deterministic output when one zone makes partitioning irrelevant."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2, 3]})
    compiler = LinearCompiler(architecture)
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrzz(0.3) q[0],q[1];\n'

    baseline = compiler.compile(qasm)
    partitioned = compiler.compile(qasm, pre_partition=True)

    assert partitioned == baseline


def test_pre_partitioned_multi_zone_schedule_is_complete_and_replayable() -> None:
    """Compile every gate and replay the resulting two-zone schedule in full."""
    architecture = LinearArchitecture(
        num_sites=9,
        processing_zones={"left": [1, 2], "right": [6, 7]},
    )
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rzz(0.1) q[0],q[1];
rx(0.2) q[2];
rzz(0.3) q[1],q[2];
"""
    compiler = LinearCompiler(architecture)

    result = compiler.compile(qasm, pre_partition=True)
    circuit = parse_circuit(
        qasm,
        gate_types=tuple(
            action_type for action_type in compiler.action_types or () if issubclass(action_type, GateAction)
        ),
    )
    gate_order = list(circuit.gate_ids)
    scheduled_gate_ids = {
        item.action.gate_id for item in result.schedule.scheduled_actions if isinstance(item.action, GateAction)
    }

    assert result.status is CompilationStatus.SUCCESS
    assert result.diagnostics is not None
    assert result.diagnostics.preferred_gate_zones
    assert {gate_id for gate_id, _zone_id in result.diagnostics.preferred_gate_zones} == set(gate_order)
    assert result.final_state == result.architecture.replay_schedule(result.schedule)
    assert scheduled_gate_ids == set(gate_order)


def test_compiler_proposes_only_the_selected_hardware_actions() -> None:
    """Generate transport only when the selected subset of the catalog contains it."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [2]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    shuttle_result = LinearCompiler(
        architecture,
        action_types=(Rx, Shuttle),
    ).compile(qasm, initial_placement=[0])
    unavailable_result = LinearCompiler(
        architecture,
        action_types=(Rx,),
    ).compile(qasm, initial_placement=[0])

    assert shuttle_result.status is CompilationStatus.SUCCESS
    assert shuttle_result.path == [Shuttle(ion=0, src=0, dst=1), Shuttle(ion=0, src=1, dst=2), Rx(ion=0, theta=0.5)]
    assert unavailable_result.status is CompilationStatus.FAILED


def test_compiler_defaults_to_all_built_in_hardware_actions() -> None:
    """Expose every built-in hardware capability by default."""
    compiler = LinearCompiler(LinearArchitecture(num_sites=2))

    assert compiler.action_types == DEFAULT_ACTION_TYPES
    assert compiler.action_types == (PhysicalSwap, Shuttle, Rx, Ry, Rz, Rzz)


def test_compiler_requires_explicit_opt_in_for_non_default_gates() -> None:
    """Keep additional supported gates outside the default hardware set."""
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrxx(0.5) q[0],q[1];\n'

    with pytest.raises(ValueError, match="unavailable gate 'rxx'"):
        LinearCompiler(architecture).compile(qasm)

    extended_architecture = replace(architecture, supported_action_types=(*DEFAULT_ACTION_TYPES, Rxx))
    result = LinearCompiler(extended_architecture, action_types=(*DEFAULT_ACTION_TYPES, Rxx)).compile(qasm)
    assert result.status is CompilationStatus.SUCCESS


def test_compiler_rejects_circuit_gates_missing_from_hardware_catalog() -> None:
    """Reject a circuit immediately when its gate is unavailable."""
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrzz(0.5) q[0],q[1];\n'

    with pytest.raises(ValueError, match="unavailable gate 'rzz'"):
        LinearCompiler(architecture, action_types=(PhysicalSwap, Shuttle)).compile(qasm)


@pytest.mark.parametrize("circuit_kind", ["qasm", "qiskit"])
def test_compiler_lowers_an_enabled_gate_from_each_frontend(circuit_kind: str) -> None:
    """Lower an enabled non-default gate from QASM and Qiskit and restore the result."""
    architecture = LinearArchitecture(
        num_sites=2,
        processing_zones={"pz": [0, 1]},
        supported_action_types=(*DEFAULT_ACTION_TYPES, Rxx),
    )
    if circuit_kind == "qasm":
        circuit: str | QuantumCircuit = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrxx(0.5) q[0],q[1];\n'
    else:
        circuit = QuantumCircuit(2)
        circuit.rxx(0.5, 0, 1)

    result = LinearCompiler(architecture).compile(circuit)
    restored = result_from_json(result.to_json())

    assert result.status is CompilationStatus.SUCCESS
    assert result.path == [Rxx(ion_a=0, ion_b=1, theta=0.5)]
    assert restored.path == result.path
    assert restored.architecture == result.architecture
    assert restored.architecture.supports(Rxx)


def test_compiler_rejects_non_action_types() -> None:
    """Reject catalog entries that do not describe hardware actions."""
    with pytest.raises(TypeError, match="Action subclasses"):
        LinearCompiler(LinearArchitecture(num_sites=2), action_types=(cast("type[Action]", object),))

    with pytest.raises(ValueError, match="must not contain duplicates"):
        LinearCompiler(LinearArchitecture(num_sites=2), action_types=(Rx, Rx))


def test_qasm_and_qiskit_inputs_compile_equivalently() -> None:
    """Give equivalent circuit representations the same schedule."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2, 3]})
    compiler = LinearCompiler(architecture)
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rx(0.1) q[0];
ry(0.2) q[1];
rzz(0.3) q[0],q[1];
"""
    circuit = QuantumCircuit(2)
    circuit.rx(0.1, 0)
    circuit.ry(0.2, 1)
    circuit.rzz(0.3, 0, 1)

    from_qasm = compiler.compile(qasm)
    from_qiskit = compiler.compile(circuit)

    assert from_qasm.status is CompilationStatus.SUCCESS
    assert from_qiskit.status is CompilationStatus.SUCCESS
    assert from_qiskit.path == from_qasm.path
    assert from_qiskit.end_time == from_qasm.end_time
    assert from_qiskit.diagnostics == from_qasm.diagnostics
    assert from_qiskit.final_state == from_qasm.final_state


def test_compiler_accepts_a_qasm_path(tmp_path: Path) -> None:
    """Read an explicitly supplied circuit file as UTF-8."""
    qasm_path = tmp_path / "circuit.qasm"
    qasm_path.write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n',
        encoding="utf-8",
    )

    result = LinearCompiler(LinearArchitecture(num_sites=1)).compile(qasm_path)

    assert result.status is CompilationStatus.SUCCESS
    assert result.end_time == 1


def test_compiler_accepts_explicit_initial_placement() -> None:
    """Start the circuit from caller-selected hardware sites."""
    compiler = LinearCompiler(
        LinearArchitecture(num_sites=5, processing_zones={"pz": [0, 1]}),
    )
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm, initial_placement=[0])

    assert result.status is CompilationStatus.SUCCESS
    assert result.initial_state is not None
    assert result.initial_state.positions == ((0, 0),)


def test_exhaustive_configuration_compiles_through_the_same_facade() -> None:
    """Select complete-circuit search through configuration alone."""
    config = LinearCompilerConfig(
        search=SearchConfig(
            horizon=None,
            committed_gates=1,
            iterative_diving_search=False,
            max_frontier_size=None,
            max_compile_time=None,
        ),
    )
    compiler = LinearCompiler(LinearArchitecture(num_sites=1), config=config)
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.SUCCESS
    assert result.end_time == 1


def test_dependency_setting_controls_parallel_gate_readiness() -> None:
    """Use the configured circuit-dependency policy during compilation."""
    architecture = LinearArchitecture(
        num_sites=2,
        processing_zones={"left": [0], "right": [1]},
    )
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rx(0.1) q[0];
ry(0.2) q[1];
"""
    dependency_result = LinearCompiler(architecture).compile(qasm)
    sequential_config = LinearCompilerConfig(
        search=SearchConfig(use_dependencies=False),
    )
    sequential_result = LinearCompiler(architecture, config=sequential_config).compile(qasm)

    assert dependency_result.status is CompilationStatus.SUCCESS
    assert sequential_result.status is CompilationStatus.SUCCESS
    assert dependency_result.end_time == 1
    assert sequential_result.end_time == 2


def test_barrier_delays_gates_on_its_qubits_until_earlier_gates_finish() -> None:
    """Start a gate after a register barrier only when every earlier gate has finished."""
    architecture = LinearArchitecture(
        num_sites=2,
        processing_zones={"left": [0], "right": [1]},
        gate_timing=GateTiming(rx=5),
    )
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrx(0.1) q[1];\nbarrier q;\nrz(0.2) q[0];\n'

    result = LinearCompiler(architecture).compile(qasm, initial_placement=[0, 1])

    assert result.status is CompilationStatus.SUCCESS
    start_times = {type(item.action): item.start_time for item in result.schedule.scheduled_actions}
    assert start_times == {Rx: 0, Rz: 5}
    result.validate()


def test_zero_time_budget_returns_timeout() -> None:
    """Return a partial result when no search time is available."""
    config = LinearCompilerConfig(search=SearchConfig(max_compile_time=0))
    compiler = LinearCompiler(LinearArchitecture(num_sites=1), config=config)
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.TIMEOUT
    assert result.path == []
    assert result.final_state is not None
    assert result.final_state.positions == result.initial_state.positions


def test_failed_search_result_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass an unsuccessful search outcome through the facade unchanged."""
    architecture = LinearArchitecture(num_sites=1)
    schedule = schedule_from_path([], create_initial_state(1, architecture), architecture)
    failed = CompilationResult(
        status=CompilationStatus.FAILED,
        schedule=schedule,
        architecture=architecture,
        final_state=schedule.initial_state,
    )

    def fail_search(*_args: object, **_kwargs: object) -> CompilationResult:
        return failed

    monkeypatch.setattr(compiler_module, "search", fail_search)
    compiler = LinearCompiler(LinearArchitecture(num_sites=1))
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result is failed


def test_interruption_returns_an_interrupted_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the best state when search is interrupted."""

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(search_module, "_run_search", interrupt)
    compiler = LinearCompiler(LinearArchitecture(num_sites=1))
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.INTERRUPTED
    assert result.path == []


def test_invalid_input_fails_before_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported operations before constructing a search problem."""

    def unexpected_search(*_args: object, **_kwargs: object) -> None:
        pytest.fail("search must not run for unsupported circuit input")

    monkeypatch.setattr(compiler_module, "search", unexpected_search)
    compiler = LinearCompiler(LinearArchitecture(num_sites=1))
    circuit = QuantumCircuit(1)
    circuit.h(0)

    with pytest.raises(ValueError, match="unavailable gate 'h'"):
        compiler.compile(circuit)


def test_compilation_does_not_write_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ordinary compilation free of filesystem output."""
    monkeypatch.chdir(tmp_path)
    compiler = LinearCompiler(LinearArchitecture(num_sites=1))
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.SUCCESS
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("horizon", [None, 1])
@pytest.mark.parametrize("limit", [1.0, 3.0, None])
def test_partition_time_counts_toward_compile_budget(
    monkeypatch: pytest.MonkeyPatch, horizon: int | None, limit: float | None
) -> None:
    """Charge partition time to both search modes and elapsed-time metadata."""
    now = [0.0]

    def partition(*_args: object, **_kwargs: object) -> dict[int, str]:
        now[0] += 2.0
        return {0: "left"}

    monkeypatch.setattr(compiler_module, "perf_counter", lambda: now[0])
    monkeypatch.setattr(search_module, "perf_counter", lambda: now[0])
    monkeypatch.setattr(compiler_module, "compute_gate_zone_assignment", partition)
    config = LinearCompilerConfig(search=SearchConfig(horizon=horizon, committed_gates=1, max_compile_time=limit))
    compiler = LinearCompiler(
        LinearArchitecture(num_sites=4, processing_zones={"left": [0, 1], "right": [2, 3]}), config
    )
    circuit = QuantumCircuit(2)
    circuit.rzz(0.3, 0, 1)

    result = compiler.compile(circuit, pre_partition=True)

    expired = limit is not None and limit < 2.0
    assert result.status is (CompilationStatus.TIMEOUT if expired else CompilationStatus.SUCCESS)
    assert result.wall_clock_s == pytest.approx(2.0)
    assert compiler.config.search.max_compile_time == limit
    if expired:
        assert result.path == []
