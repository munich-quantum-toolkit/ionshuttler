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


def _control_flow(steps: tuple[Step, ...]) -> tuple[object, ...]:
    """Return the steps with their conditions and with the measurements and resets of each circuit, in order.

    The native rotations are left out: MQT Core may synthesize a different but equivalent rotation sequence on
    another platform. On Windows, it emits two rotations fewer in one circuit of the teleportation program.
    """
    shape: list[object] = []
    for step in steps:
        if isinstance(step, Circuit):
            operations = tuple(
                (type(gate).__name__, gate.ions, getattr(gate, "outcome", None))
                for gate in step.gates
                if isinstance(gate, Measure | Reset)
            )
            shape.append(("circuit", step.num_ions, operations))
        elif isinstance(step, Branch):
            shape.append(("branch", step.condition, _control_flow(step.then_steps), _control_flow(step.else_steps)))
        else:
            assert isinstance(step, Loop)
            shape.append(("loop", step.count, step.entry, step.repeat, _control_flow(step.steps)))
    return tuple(shape)


def test_program_from_qasm_matches_the_stored_qco_fixture() -> None:
    """Read the same control flow, conditions, measurements, and resets as the stored fixture, with native gates."""
    from_qasm = program_from_qasm(FIXTURES / "teleportation.qasm")
    stored = parse_program((FIXTURES / "teleportation.qco").read_text(encoding="utf-8"))

    assert from_qasm.num_ions == stored.num_ions
    assert _control_flow(from_qasm.steps) == _control_flow(stored.steps)
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
