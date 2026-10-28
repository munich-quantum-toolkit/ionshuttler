# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for reading QCO programs.

The ``.qco`` fixtures are MQT Core 4.1.1 output for the ``.qasm`` file with the
same name after native-gate synthesis with an all-to-all target.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.core.gates import Measure, Reset, Rx, Rxx, Rz
from mqt.ionshuttler.dynamic import Branch, Condition, Loop, parse_program

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / f"{name}.qco").read_text(encoding="utf-8")


def test_teleportation_becomes_a_circuit_and_two_branches() -> None:
    """Keep the measurement outcomes that decide each correction."""
    program = parse_program(_read("teleportation"))

    prefix, x_correction, z_correction = program.steps
    assert program.num_ions == 3
    assert isinstance(prefix, Circuit)
    assert {type(gate) for gate in prefix.gates} == {Rx, Rz, Rxx, Measure}
    assert [gate.outcome for gate in prefix.gates if isinstance(gate, Measure)] == [0, 1]
    assert isinstance(x_correction, Branch)
    assert x_correction.condition == Condition((1,), frozenset({(1,)}))
    assert x_correction.else_steps == ()
    assert isinstance(z_correction, Branch)
    assert z_correction.condition == Condition((0,), frozenset({(1,)}))
    correction = z_correction.then_steps[0]
    assert isinstance(correction, Circuit)
    assert all(gate.ions == (1,) for gate in correction.gates)


def test_nested_branch_reads_an_outcome_measured_inside_its_parent() -> None:
    """Number measurements in program order, also inside branches."""
    program = parse_program(_read("nested_branches"))

    _prefix, outer, _suffix = program.steps
    assert isinstance(outer, Branch)
    inner = outer.then_steps[1]
    assert isinstance(inner, Branch)
    assert inner.condition == Condition((1,), frozenset({(1,)}))
    entangle = inner.else_steps[0]
    assert isinstance(entangle, Circuit)
    assert any(isinstance(gate, Rxx) for gate in entangle.gates)
    assert list(program.paths()) == [{0: 0}, {0: 1, 1: 0}, {0: 1, 1: 1}]


def test_reset_keeps_the_ion_for_later_gates() -> None:
    """Treat reset as an operation on the same ion."""
    program = parse_program(_read("reset_and_reuse"))
    prefix = program.steps[0]

    assert isinstance(prefix, Circuit)
    operations = [type(gate) for gate in prefix.gates]
    reset_position = operations.index(Reset)
    assert operations[reset_position - 1] is Measure
    assert prefix.gates[reset_position].ions == prefix.gates[reset_position - 1].ions
    assert prefix.predecessors[reset_position] == frozenset({reset_position - 1})


def test_register_comparisons_become_truth_tables() -> None:
    """Translate whole-register and bitwise comparisons into outcome combinations."""
    program = parse_program(_read("register_conditions"))

    both, different = (step for step in program.steps if isinstance(step, Branch))
    assert both.condition == Condition((0, 1), frozenset({(1, 1)}))
    assert different.condition == Condition((0, 1), frozenset({(0, 1), (1, 0)}))


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("qco.rz(%cst) %q :", "qco.h %q :", "line 11: operation 'qco.h' is not a native gate"),
        ("qco.sink %q_21", "scf.index_switch %q_21", "unsupported operation 'scf.index_switch'"),
        ("qco.sink %q_21", "func.call %q_21", "unsupported operation 'func.call'"),
    ],
)
def test_reader_names_unsupported_operations(original: str, replacement: str, message: str) -> None:
    """Report the line of an operation outside the supported subset."""
    text = _read("reset_and_reuse").replace(original, replacement, 1)

    with pytest.raises(ValueError, match=message):
        parse_program(text)


def test_reader_rejects_a_condition_on_an_unset_bit() -> None:
    """Refuse a branch that reads a classical bit that no measurement sets."""
    text = _read("teleportation").replace("    cbit.store %b, %c[%c0] : !cbit.reg<2>\n", "", 1)

    with pytest.raises(ValueError, match="never set"):
        parse_program(text)


def test_counted_loop_keeps_its_body_and_count() -> None:
    """Read scf.for with constant bounds as one loop instead of unrolling it."""
    program = parse_program(_read("counted_loop"))

    (loop,) = program.steps
    assert isinstance(loop, Loop)
    assert loop.count == 3
    _body, correction = loop.steps
    assert isinstance(correction, Branch)
    assert correction.condition == Condition((0,), frozenset({(1,)}))
    assert len(list(program.paths())) == 8


def test_while_loop_reads_entry_and_repeat_conditions() -> None:
    """Evaluate the condition region with the register values before and after an iteration."""
    program = parse_program(_read("repeat_until_success"))

    (loop,) = program.steps
    assert isinstance(loop, Loop)
    assert loop.count is None
    assert loop.entry == Condition((), frozenset({()}))
    assert loop.repeat == Condition((0,), frozenset({(1,)}))
