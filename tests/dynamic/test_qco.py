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

from itertools import product
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
    assert x_correction.condition == Condition.measured(1)
    assert x_correction.else_steps == ()
    assert isinstance(z_correction, Branch)
    assert z_correction.condition == Condition.measured(0)
    correction = z_correction.then_steps[0]
    assert isinstance(correction, Circuit)
    # MQT places q[2] on site 1; the ion number follows the program qubit.
    assert correction.gates
    assert all(gate.ions == (2,) for gate in correction.gates)


def test_ions_follow_sites_without_a_recorded_placement() -> None:
    """Use site numbers as ion numbers when the module records no qubit placement."""
    header, body = _read("teleportation").split("\n", 1)
    assert header.startswith("module attributes {mqt.layout")
    program = parse_program("module {\n" + body)

    z_correction = program.steps[2]
    assert isinstance(z_correction, Branch)
    correction = z_correction.then_steps[0]
    assert isinstance(correction, Circuit)
    assert correction.gates
    assert all(gate.ions == (1,) for gate in correction.gates)


def test_reader_rejects_a_malformed_qubit_placement() -> None:
    """Refuse a recorded placement that puts two qubits on one site."""
    text = _read("teleportation").replace("initial = array<i64: 2, 0, 1>", "initial = array<i64: 2, 2, 1>")

    with pytest.raises(ValueError, match=r"malformed mqt\.layout"):
        parse_program(text)


def test_nested_branch_reads_an_outcome_measured_inside_its_parent() -> None:
    """Number measurements in program order, also inside branches."""
    program = parse_program(_read("nested_branches"))

    _prefix, outer, _suffix = program.steps
    assert isinstance(outer, Branch)
    inner = outer.then_steps[1]
    assert isinstance(inner, Branch)
    assert inner.condition == Condition.measured(1)
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


def test_register_comparisons_become_conditions() -> None:
    """Translate whole-register and bitwise comparisons into conditions on the measured bits."""
    program = parse_program(_read("register_conditions"))

    both, different = (step for step in program.steps if isinstance(step, Branch))
    for first, second in product((0, 1), repeat=2):
        values = {0: first, 1: second}
        assert both.condition.holds(values) == (first == second == 1)
        assert different.condition.holds(values) == (first != second)


def test_counters_and_register_comparisons_read_many_outcomes() -> None:
    """Read integer counters and a comparison of two 9-bit registers over 18 outcomes."""
    program = parse_program(_read("counted_outcomes"))

    at_least_nine, equal, fewer_than_five = (step.condition for step in program.steps if isinstance(step, Branch))
    assert at_least_nine.outcomes == equal.outcomes == fewer_than_five.outcomes == tuple(range(18))
    # Every 997th assignment of the 2^18 assignments, plus the edge cases.
    samples = [[(number >> position) & 1 for position in range(18)] for number in range(0, 1 << 18, 997)]
    for bits in [*samples, [1] * 18, [1] * 4 + [0] * 14, [1, 0, 1] * 6]:
        values = dict(enumerate(bits))
        assert at_least_nine.holds(values) == (sum(bits) >= 9)
        assert equal.holds(values) == (bits[:9] == bits[9:])
        assert fewer_than_five.holds(values) == (sum(bits) - 3 < 2)


def test_a_counter_changed_inside_a_branch_decides_a_nested_branch() -> None:
    """Carry a counter out of a branch side and read it in a branch nested in that side."""
    program = parse_program(_read("counter_in_branch"))

    _prefix, outer = program.steps
    assert isinstance(outer, Branch)
    assert outer.condition == Condition.measured(0)
    inner = next(step for step in outer.else_steps if isinstance(step, Branch))
    assert inner.condition.outcomes == (1,)
    # The else side counts one, plus one more when outcome 1 is 1.
    assert inner.condition.holds({1: 1})
    assert not inner.condition.holds({1: 0})


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
    lines = _read("teleportation").splitlines()
    store = lines.index("    cbit.store %b, %c[%c0] : !cbit.reg<2>")
    text = "\n".join((*lines[:store], *lines[store + 1 :]))

    with pytest.raises(ValueError, match="never set"):
        parse_program(text)


def test_reader_rejects_a_register_write_in_a_loop_condition() -> None:
    """Refuse a while-loop condition region that changes a register, because the program would lose the write."""
    lines = _read("repeat_until_success").splitlines()
    load = lines.index("      %b = cbit.load %c[%c0] : !cbit.reg<1>")
    text = "\n".join((*lines[: load + 1], "      cbit.store %b, %c[%c0] : !cbit.reg<1>", *lines[load + 1 :]))

    with pytest.raises(ValueError, match="must not write classical registers"):
        parse_program(text)


def test_counted_loop_keeps_its_body_and_count() -> None:
    """Read scf.for with constant bounds as one loop instead of unrolling it."""
    program = parse_program(_read("counted_loop"))

    (loop,) = program.steps
    assert isinstance(loop, Loop)
    assert loop.count == 3
    _body, correction = loop.steps
    assert isinstance(correction, Branch)
    assert correction.condition == Condition.measured(0)
    assert len(list(program.paths())) == 8


def test_while_loop_reads_entry_and_repeat_conditions() -> None:
    """Evaluate the condition region with the register values before and after an iteration."""
    program = parse_program(_read("repeat_until_success"))

    (loop,) = program.steps
    assert isinstance(loop, Loop)
    assert loop.count is None
    assert loop.entry == Condition.always()
    assert loop.repeat == Condition.measured(0)
