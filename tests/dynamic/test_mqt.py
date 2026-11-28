# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end tests that read OpenQASM 3 through MQT Core."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.core.gates import Measure, Reset, Rx, Rxx, Ry, Rz
from mqt.ionshuttler.dynamic import Branch, Loop, compile_program, parse_program, program_from_qasm
from mqt.ionshuttler.grid import CompilationStatus, GridCompiler, square_grid

if TYPE_CHECKING:
    from mqt.ionshuttler.dynamic.program import Step

pytest.importorskip("mqt.core.mlir")

FIXTURES = Path(__file__).parent / "fixtures"
# Synthesis on another platform may round an angle differently in the last digit.
ANGLE_TOLERANCE = 1e-12


def _assert_same_steps(actual: tuple[Step, ...], expected: tuple[Step, ...]) -> None:
    """Require the same structure, conditions, gates, ions, and dependencies, and angles within the tolerance."""
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert type(got) is type(want)
        if isinstance(want, Circuit):
            assert isinstance(got, Circuit)
            assert (got.num_ions, got.predecessors, got.barriers) == (want.num_ions, want.predecessors, want.barriers)
            assert len(got.gates) == len(want.gates)
            for got_gate, want_gate in zip(got.gates, want.gates, strict=True):
                assert type(got_gate) is type(want_gate)
                assert got_gate.ions == want_gate.ions
                assert getattr(got_gate, "outcome", None) == getattr(want_gate, "outcome", None)
                for name in want_gate.parameter_names:
                    assert getattr(got_gate, name) == pytest.approx(
                        getattr(want_gate, name), rel=0, abs=ANGLE_TOLERANCE
                    )
        elif isinstance(want, Branch):
            assert isinstance(got, Branch)
            assert got.condition == want.condition
            _assert_same_steps(got.then_steps, want.then_steps)
            _assert_same_steps(got.else_steps, want.else_steps)
        else:
            assert isinstance(got, Loop)
            assert isinstance(want, Loop)
            assert (got.count, got.entry, got.repeat) == (want.count, want.entry, want.repeat)
            _assert_same_steps(got.steps, want.steps)


def test_program_from_qasm_matches_the_stored_qco_fixture() -> None:
    """Synthesize native gates with MQT Core like the stored fixtures."""
    from_qasm = program_from_qasm(FIXTURES / "teleportation.qasm")
    stored = parse_program((FIXTURES / "teleportation.qco").read_text(encoding="utf-8"))

    assert from_qasm.num_ions == stored.num_ions
    _assert_same_steps(from_qasm.steps, stored.steps)
    assert {type(gate) for gate in from_qasm.gates()} <= {Rx, Ry, Rz, Rxx, Measure, Reset}


def test_a_qasm_program_compiles_end_to_end() -> None:
    """Read, synthesize, compile, and validate a program with reset and a branch."""
    program = program_from_qasm((FIXTURES / "reset_and_reuse.qasm").read_text(encoding="utf-8"))

    result = compile_program(
        program, GridCompiler(square_grid(2, processing_zone_segments=("h:0:0",), segment_capacity=2))
    )

    assert result.status is CompilationStatus.SUCCESS
    result.validate()


def test_integer_counters_decide_branches() -> None:
    """Read counters that add measured bits and compare the sum, as in adaptive error correction."""
    source = """OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
bit[3] m;
m[0] = measure q[0];
m[1] = measure q[1];
m[2] = measure q[2];
int[8] ones = 0;
if (m[0]) { ones = ones + 1; }
if (m[1]) { ones = ones + 1; }
if (m[2]) { ones = ones + 1; }
if (ones - 1 >= 1) { x q[3]; }
"""
    program = program_from_qasm(source)

    *_, branch = program.steps
    assert isinstance(branch, Branch)
    for bits in product((0, 1), repeat=3):
        assert branch.condition.holds(dict(enumerate(bits))) == (sum(bits) >= 2)


def test_a_condition_after_a_loop_cannot_read_a_bit_that_the_loop_writes() -> None:
    """Reject a condition whose value depends on which iteration wrote it."""
    source = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[1] c = "0";
for int r in [0:1] {
  h q[0];
  c[0] = measure q[0];
}
if (c[0]) { x q[1]; }
"""

    with pytest.raises(ValueError, match="a loop writes in another iteration"):
        program_from_qasm(source)
