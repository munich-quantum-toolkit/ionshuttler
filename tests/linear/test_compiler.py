# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end tests for the Linear compiler facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from qiskit import QuantumCircuit

import mqt.ionshuttler.linear.compiler as compiler_module
import mqt.ionshuttler.linear.search as search_module
from mqt.ionshuttler.linear import Architecture, LinearCompiler
from mqt.ionshuttler.linear.config import LinearCompilerConfig, SearchConfig
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus

if TYPE_CHECKING:
    from pathlib import Path


def test_compiler_matches_the_compact_frozen_result(
    production_default_golden: dict[str, object],
) -> None:
    """Match the exact production schedule through the public facade."""
    inputs = cast("dict[str, object]", production_default_golden["input"])
    architecture = Architecture.from_dict(inputs["architecture"])
    compiler = LinearCompiler(architecture)

    result = compiler.compile(cast("str", inputs["qasm"]))

    actual = result.to_dict()
    actual.pop("wall_clock_s")
    expected = cast("dict[str, object]", production_default_golden["expected_result"])
    assert actual == expected
    assert result.final_state is not None
    assert result.final_state.time == 5
    assert result.final_state.completed_gates == frozenset({0, 1, 2})


def test_qasm_and_qiskit_inputs_compile_equivalently() -> None:
    """Give equivalent circuit representations the same schedule."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [2, 3]})
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
    assert from_qiskit.num_timesteps == from_qasm.num_timesteps
    assert from_qiskit.score == from_qasm.score
    assert from_qiskit.final_state == from_qasm.final_state


def test_compiler_accepts_a_qasm_path(tmp_path: Path) -> None:
    """Read an explicitly supplied circuit file as UTF-8."""
    qasm_path = tmp_path / "circuit.qasm"
    qasm_path.write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n',
        encoding="utf-8",
    )

    result = LinearCompiler(Architecture(num_sites=1)).compile(qasm_path)

    assert result.status is CompilationStatus.SUCCESS
    assert result.num_timesteps == 1


def test_compiler_accepts_explicit_initial_positions() -> None:
    """Start the circuit from caller-selected hardware sites."""
    compiler = LinearCompiler(
        Architecture(num_sites=5, processing_zones={"pz": [0, 1]}),
    )
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm, initial_positions=[0])

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
    compiler = LinearCompiler(Architecture(num_sites=1), config=config)
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.SUCCESS
    assert result.num_timesteps == 1


def test_dependency_setting_controls_parallel_gate_readiness() -> None:
    """Use the configured circuit-dependency policy during normalization."""
    architecture = Architecture(
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
    assert dependency_result.num_timesteps == 1
    assert sequential_result.num_timesteps == 2


def test_zero_time_budget_returns_timeout() -> None:
    """Return a partial result when no search time is available."""
    config = LinearCompilerConfig(search=SearchConfig(max_compile_time=0))
    compiler = LinearCompiler(Architecture(num_sites=1), config=config)
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.TIMEOUT
    assert result.path == []
    assert result.final_state == result.initial_state


def test_failed_search_result_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass an unsuccessful search outcome through the facade unchanged."""
    failed = CompilationResult(
        status=CompilationStatus.FAILED,
        path=[],
        num_timesteps=0,
    )

    def fail_search(*_args: object, **_kwargs: object) -> CompilationResult:
        return failed

    monkeypatch.setattr(compiler_module, "search", fail_search)
    compiler = LinearCompiler(Architecture(num_sites=1))
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result is failed


def test_interruption_returns_an_interrupted_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the best state when search is interrupted."""

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(search_module, "_run_search", interrupt)
    compiler = LinearCompiler(Architecture(num_sites=1))
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.INTERRUPTED
    assert result.path == []


def test_invalid_input_fails_before_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported operations before constructing a search problem."""

    def unexpected_search(*_args: object, **_kwargs: object) -> None:
        pytest.fail("search must not run for unsupported circuit input")

    monkeypatch.setattr(compiler_module, "search", unexpected_search)
    compiler = LinearCompiler(Architecture(num_sites=1))
    circuit = QuantumCircuit(1)
    circuit.h(0)

    with pytest.raises(ValueError, match="unsupported circuit operation"):
        compiler.compile(circuit)


def test_compilation_does_not_write_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ordinary compilation free of filesystem output."""
    monkeypatch.chdir(tmp_path)
    compiler = LinearCompiler(Architecture(num_sites=1))
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.SUCCESS
    assert list(tmp_path.iterdir()) == []
