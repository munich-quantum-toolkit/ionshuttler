# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end tests that read OpenQASM 3 through MQT Core."""

from __future__ import annotations

from pathlib import Path

import pytest

from mqt.ionshuttler.core.gates import Measure, Reset, Rx, Rxx, Ry, Rz
from mqt.ionshuttler.dynamic import compile_program, parse_program, program_from_qasm
from mqt.ionshuttler.grid import CompilationStatus, GridCompiler, square_grid

pytest.importorskip("mqt.core.mlir")

FIXTURES = Path(__file__).parent / "fixtures"


def test_program_from_qasm_matches_the_stored_qco_fixture() -> None:
    """Synthesize native gates with MQT Core like the stored fixtures."""
    from_qasm = program_from_qasm(FIXTURES / "teleportation.qasm")
    stored = parse_program((FIXTURES / "teleportation.qco").read_text(encoding="utf-8"))

    assert from_qasm == stored
    assert {type(gate) for gate in from_qasm.gates()} <= {Rx, Ry, Rz, Rxx, Measure, Reset}


def test_a_qasm_program_compiles_end_to_end() -> None:
    """Read, synthesize, compile, and validate a program with reset and a branch."""
    program = program_from_qasm((FIXTURES / "reset_and_reuse.qasm").read_text(encoding="utf-8"))

    result = compile_program(
        program, GridCompiler(square_grid(2, processing_zone_segments=("h:0:0",), segment_capacity=2))
    )

    assert result.status is CompilationStatus.SUCCESS
    result.validate()


def test_a_condition_after_a_loop_cannot_read_a_bit_that_the_loop_writes() -> None:
    """Reject a decision whose value depends on which iteration wrote it."""
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
