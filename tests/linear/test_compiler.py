# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end tests for the Linear compiler facade."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, cast

import pytest
from qiskit import QuantumCircuit

import mqt.ionshuttler.linear.compiler as compiler_module
import mqt.ionshuttler.linear.search as search_module
from mqt.ionshuttler.linear import DEFAULT_ACTION_TYPES, Architecture, LinearCompiler
from mqt.ionshuttler.linear.actions import (
    Action,
    PhysicalSwap,
    Rx,
    Rxx,
    Ry,
    Rz,
    Rzz,
    Shuttle,
    TransportAction,
    TwoQubitGate,
)
from mqt.ionshuttler.linear.config import LinearCompilerConfig, SearchConfig, TransportTiming
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from mqt.ionshuttler.linear.state import State


@dataclass(frozen=True)
class _LongShuttle(TransportAction):
    """Move one ion two sites in a single hardware operation."""

    ion: int
    src: int
    dst: int
    duration: int = 1

    @classmethod
    def available_actions(
        cls,
        state: State,
        architecture: Architecture,
        transport_timing: TransportTiming,
    ) -> Iterable[Action]:
        """Return forward two-site moves that remain on the device."""
        del transport_timing
        occupied = {position for _, position in state.positions}
        return (
            cls(ion=ion, src=source, dst=source + 2)
            for ion, source in state.positions
            if source + 2 < architecture.num_sites and source + 2 not in occupied
        )

    def is_valid(self, state: State, architecture: Architecture) -> bool:
        """Return whether the ion can make the requested two-site move."""
        positions = dict(state.positions)
        return (
            positions.get(self.ion) == self.src
            and self.dst == self.src + 2
            and self.dst < architecture.num_sites
            and self.dst not in positions.values()
            and dict(state.ions_busy_until).get(self.ion, state.time + 1) <= state.time
        )

    def apply(self, state: State, architecture: Architecture) -> State:
        """Move the ion and reserve it for the configured duration."""
        del architecture
        positions = dict(state.positions)
        busy_until = dict(state.ions_busy_until)
        positions[self.ion] = self.dst
        busy_until[self.ion] = state.time + self.duration
        return replace(
            state,
            positions=tuple(sorted(positions.items())),
            ions_busy_until=tuple(sorted(busy_until.items())),
        )


@dataclass(frozen=True)
class _CX(TwoQubitGate):
    """Controlled-X gate used to exercise catalog-driven circuit lowering."""

    circuit_name: ClassVar[str] = "cx"
    duration: int = 2


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
    actual_action_types = actual.pop("action_types")
    expected = cast("dict[str, object]", production_default_golden["expected_result"])
    assert actual == expected
    assert actual_action_types == ["PhysicalSwap", "Shuttle", "Rx", "Ry", "Rz", "Rzz"]
    assert result.final_state is not None
    assert result.final_state.time == 5
    assert result.final_state.completed_gates == frozenset({0, 1, 2})


def test_compiler_uses_the_hardware_action_catalog() -> None:
    """Discover a custom action through its class-owned availability method."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [2]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    custom_result = LinearCompiler(
        architecture,
        action_types=(Rx, _LongShuttle),
    ).compile(qasm, initial_positions=[0])
    unavailable_result = LinearCompiler(
        architecture,
        action_types=(Rx,),
    ).compile(qasm, initial_positions=[0])

    assert custom_result.status is CompilationStatus.SUCCESS
    assert any(isinstance(action, _LongShuttle) for action in custom_result.path)
    assert unavailable_result.status is CompilationStatus.FAILED


def test_compiler_defaults_to_all_built_in_hardware_actions() -> None:
    """Expose every built-in hardware capability by default."""
    compiler = LinearCompiler(Architecture(num_sites=2))

    assert compiler.action_types == DEFAULT_ACTION_TYPES
    assert compiler.action_types == (PhysicalSwap, Shuttle, Rx, Ry, Rz, Rzz)


def test_compiler_requires_explicit_opt_in_for_non_default_gates() -> None:
    """Keep additional supported gates outside the default hardware set."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrxx(0.5) q[0],q[1];\n'

    with pytest.raises(ValueError, match="unavailable gate 'rxx'"):
        LinearCompiler(architecture).compile(qasm)

    result = LinearCompiler(architecture, action_types=(*DEFAULT_ACTION_TYPES, Rxx)).compile(qasm)
    assert result.status is CompilationStatus.SUCCESS


def test_compiler_rejects_circuit_gates_missing_from_hardware_catalog() -> None:
    """Reject a circuit immediately when its gate is unavailable."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrzz(0.5) q[0],q[1];\n'

    with pytest.raises(ValueError, match="unavailable gate 'rzz'"):
        LinearCompiler(architecture, action_types=(PhysicalSwap, Shuttle)).compile(qasm)


@pytest.mark.parametrize("circuit_kind", ["qasm", "qiskit"])
def test_compiler_lowers_custom_gate_types_from_each_frontend(circuit_kind: str) -> None:
    """Use one action-owned lowerer for QASM and Qiskit circuit inputs."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    if circuit_kind == "qasm":
        circuit: str | QuantumCircuit = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncx q[0],q[1];\n'
    else:
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)

    result = LinearCompiler(
        architecture,
        action_types=(*DEFAULT_ACTION_TYPES, _CX),
    ).compile(circuit)

    assert result.status is CompilationStatus.SUCCESS
    assert any(isinstance(action, _CX) for action in result.path)
    assert result.action_types[-1] == "_CX"
    with pytest.raises(ValueError, match="unknown action type"):
        CompilationResult.from_json(result.to_json())
    restored = CompilationResult.from_json(result.to_json(), action_types=(_CX,))
    assert restored.path == result.path
    assert restored.action_types == result.action_types


def test_compiler_rejects_non_action_types() -> None:
    """Reject catalog entries that do not describe hardware actions."""
    with pytest.raises(TypeError, match="Action subclasses"):
        LinearCompiler(Architecture(num_sites=2), action_types=(cast("type[Action]", object),))

    with pytest.raises(ValueError, match="unique class names"):
        LinearCompiler(Architecture(num_sites=2), action_types=(Rx, Rx))


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

    with pytest.raises(ValueError, match="unavailable gate 'h'"):
        compiler.compile(circuit)


def test_compilation_does_not_write_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ordinary compilation free of filesystem output."""
    monkeypatch.chdir(tmp_path)
    compiler = LinearCompiler(Architecture(num_sites=1))
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.5) q[0];\n'

    result = compiler.compile(qasm)

    assert result.status is CompilationStatus.SUCCESS
    assert list(tmp_path.iterdir()) == []
