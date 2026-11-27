# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the dynamic program model."""

from __future__ import annotations

import json
import operator
from functools import reduce
from typing import TYPE_CHECKING, Any, cast

import pytest

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.core.gates import Measure, Rx
from mqt.ionshuttler.dynamic import Branch, Condition, Loop, Program

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(ion: int, outcome: int) -> Circuit:
    return Circuit(
        num_ions=2,
        gates=(Measure(ion=ion, outcome=outcome, gate_id=0),),
        predecessors=(frozenset(),),
    )


def _rotate(ion: int) -> Circuit:
    return Circuit(num_ions=2, gates=(Rx(ion=ion, theta=0.5, gate_id=0),), predecessors=(frozenset(),))


def test_condition_combines_measured_outcomes() -> None:
    """Evaluate logic over measurement outcomes and simplify constant parts."""
    first, last = Condition.measured(0), Condition.measured(2)
    parity = first ^ last

    assert parity.outcomes == (0, 2)
    assert parity.holds({0: 1, 1: 1, 2: 0})
    assert not parity.holds({0: 1, 2: 1})
    assert (first & ~last).holds({0: 1, 2: 0})
    assert not (first | last).holds({0: 0, 2: 0})
    assert Condition.from_dict(json.loads(json.dumps(parity.to_dict()))) == parity
    assert first & Condition.always() == first
    assert first | Condition.always() == Condition.always()
    assert first ^ Condition.always() == ~first
    assert ~~first == first
    with pytest.raises(ValueError, match="non-negative integers"):
        Condition.measured(-1)
    with pytest.raises(KeyError):
        parity.holds({0: 1})


def test_condition_keeps_shared_parts_once() -> None:
    """Serialize and compare a condition with shared parts in size linear in its parts."""
    # Each step reads the previous step twice, so writing it out as a tree would double its size per step.
    condition = Condition.measured(0)
    for outcome in range(1, 40):
        condition = (condition & Condition.measured(outcome)) | (condition ^ Condition.measured(outcome))
    rebuilt = Condition.from_dict(json.loads(json.dumps(condition.to_dict())))
    values = {outcome: outcome % 3 == 0 for outcome in range(40)}

    assert condition.outcomes == tuple(range(40))
    parts = condition.to_dict()["parts"]
    assert isinstance(parts, list)
    assert len(parts) < 200
    assert rebuilt == condition
    assert hash(rebuilt) == hash(condition)
    assert rebuilt.holds(values) == condition.holds(values)
    with pytest.raises(ValueError, match="only combine earlier parts"):
        Condition.from_dict({"parts": [{"operation": "not", "operands": [0]}]})


def test_program_enumerates_one_assignment_per_path() -> None:
    """Assign only the outcomes that decide branches on each path."""
    inner = Branch(Condition.measured(1), (_rotate(1),))
    program = Program(2, (_measure(0, 0), Branch(Condition.measured(0), (_measure(1, 1), inner), (_rotate(0),))))

    assert list(program.paths()) == [{0: 0}, {0: 1, 1: 0}, {0: 1, 1: 1}]
    assert program.outcomes == (0, 1)
    assert [len(circuit.gates) for circuit in program.path({0: 1, 1: 1})] == [1, 1, 1]
    assert program.path({0: 0})[-1] == _rotate(0)
    assert Program.from_dict(json.loads(json.dumps(program.to_dict()))) == program


def test_program_round_trip_preserves_circuit_barriers() -> None:
    """Preserve barrier groups and target sets in serialized circuits."""
    circuit = Circuit(
        num_ions=2,
        gates=(Rx(ion=0, theta=0.5, gate_id=0),),
        predecessors=(frozenset(),),
        barriers=((frozenset({0, 1}), frozenset({1})), (frozenset({0}),)),
    )
    program = Program(2, (circuit,))

    serialized = json.loads(json.dumps(program.to_dict()))

    assert serialized["steps"][0]["barriers"] == [[[0, 1], [1]], [[0]]]
    assert Program.from_dict(serialized) == program


def test_program_requires_outcomes_measured_before_each_branch() -> None:
    """Reject a branch that can read an outcome that its path has not measured."""
    only_then = Branch(Condition.measured(0), (_measure(1, 1),))
    with pytest.raises(ValueError, match=r"reads outcomes \[1\] before they are measured"):
        Program(2, (_measure(0, 0), only_then, Branch(Condition.measured(1), (_rotate(0),))))
    with pytest.raises(ValueError, match="exactly one measurement"):
        Program(2, (_measure(0, 0), _measure(1, 0)))
    with pytest.raises(ValueError, match="act on num_ions ions"):
        Program(3, (_rotate(0),))


ALWAYS = Condition.always()
OUTCOME_ZERO_IS_ONE = Condition.measured(0)


def test_loop_repeats_with_a_count_or_with_two_conditions() -> None:
    """Accept exactly one way to repeat a loop."""
    with pytest.raises(ValueError, match="at least 1"):
        Loop((_rotate(0),), count=0)
    with pytest.raises(TypeError, match="needs entry and repeat conditions"):
        Loop((_rotate(0),))
    with pytest.raises(ValueError, match="must not have conditions"):
        Loop((_rotate(0),), count=2, entry=ALWAYS)


def test_paths_through_a_counted_loop_key_outcomes_by_iteration() -> None:
    """Give each iteration its own measured value."""
    body = (_measure(0, 0), Branch(OUTCOME_ZERO_IS_ONE, (_rotate(1),)))
    program = Program(2, (Loop(body, count=2),))

    assert list(program.paths()) == [
        {(0, 0): 0, (0, 1): 0},
        {(0, 0): 0, (0, 1): 1},
        {(0, 0): 1, (0, 1): 0},
        {(0, 0): 1, (0, 1): 1},
    ]
    assert program.path({(0, 0): 1, (0, 1): 0}) == (_measure(0, 0), _rotate(1), _measure(0, 0))
    assert Program.from_dict(json.loads(json.dumps(program.to_dict()))) == program


def test_paths_through_a_conditional_loop_stop_at_max_iterations() -> None:
    """Leave out paths that need more iterations than requested."""
    program = Program(2, (Loop((_measure(0, 0),), entry=ALWAYS, repeat=OUTCOME_ZERO_IS_ONE),))

    assert list(program.paths(max_iterations=2)) == [{(0, 0): 0}, {(0, 0): 1, (0, 1): 0}]
    with pytest.raises(ValueError, match="needs max_iterations"):
        list(program.paths())


def test_outcomes_measured_inside_a_loop_stay_inside_it() -> None:
    """Reject conditions that read a loop outcome outside its iteration."""
    with pytest.raises(ValueError, match=r"a branch reads outcomes \[0\]"):
        Program(2, (Loop((_measure(0, 0),), count=2), Branch(OUTCOME_ZERO_IS_ONE, (_rotate(1),))))
    with pytest.raises(ValueError, match="a loop repeat condition reads outcomes"):
        Program(2, (Loop((_rotate(0),), entry=ALWAYS, repeat=OUTCOME_ZERO_IS_ONE),))


@pytest.mark.parametrize(
    ("build", "error", "message"),
    [
        (lambda: Condition("all", (Condition.measured(0), cast("Condition", 1))), TypeError, "must be conditions"),
        (lambda: Condition("nand", (Condition.measured(0),)), ValueError, "unknown condition operation"),
        (lambda: Condition("always", outcome=0), ValueError, "reads no outcome"),
        (lambda: Condition("not"), ValueError, "wrong number of operands"),
        (lambda: Condition("any"), ValueError, "wrong number of operands"),
        (lambda: Condition.from_dict({"parts": []}), ValueError, "at least one part"),
        (lambda: Condition.measured(0) & 1, TypeError, "unsupported operand"),
    ],
    ids=[
        "operand-not-a-condition",
        "unknown-operation",
        "outcome-on-a-constant",
        "not-without-operand",
        "any-without-operands",
        "document-without-parts",
        "combined-with-an-integer",
    ],
)
def test_condition_rejects_malformed_parts(build: Callable[[], object], error: type[Exception], message: str) -> None:
    """Reject condition parts whose operation, operands, and outcome do not fit together."""
    with pytest.raises(error, match=message):
        build()


def test_condition_shows_its_expression() -> None:
    """Show a short condition as an expression and a long one as a summary."""
    short = Condition.measured(0) & ~Condition.measured(1)
    long = reduce(operator.and_, (Condition.measured(outcome) for outcome in range(50)))

    assert "measured(0)" in repr(short)
    assert "~measured(1)" in repr(short)
    assert "reading 50 outcomes" in repr(long)


def test_program_lists_nested_branches_loops_and_conditions() -> None:
    """Find branches, loops, and their conditions at every nesting depth."""
    inner = Branch(Condition.measured(1), (_rotate(0),))
    loop = Loop((_measure(1, 1), inner), entry=ALWAYS, repeat=Condition.measured(1))
    outer = Branch(OUTCOME_ZERO_IS_ONE, (loop,))
    program = Program(2, (_measure(0, 0), outer))

    assert list(program.branches()) == [outer, inner]
    assert list(program.loops()) == [loop]
    assert list(program.conditions()) == [OUTCOME_ZERO_IS_ONE, ALWAYS, Condition.measured(1), Condition.measured(1)]


def test_program_and_loop_reject_wrong_types() -> None:
    """Reject an ion count or a loop count that is not an integer."""
    with pytest.raises(TypeError, match="num_ions must be an integer"):
        Program(cast("int", 2.0), (_rotate(0),))
    with pytest.raises(TypeError, match="loop count must be an integer"):
        Loop((_rotate(0),), count=cast("int", "2"))


def _set(document: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    target: Any = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("version",), 0, "unsupported program schema or version"),
        (("steps", 0, "gates", 0, "type"), "gate.unknown", "unknown program gate type"),
        (("steps", 0, "predecessors"), [["first"]], "predecessor set must be a list of integers"),
        (("steps", 0, "barriers", 0), 0, "barriers must contain lists of target sets"),
        (("steps", 1, "count"), "2", "loop count must be an integer or null"),
        (("steps", 1, "type"), "switch", "unknown program step type"),
    ],
)
def test_program_document_rejects_malformed_data(path: tuple[str | int, ...], value: object, message: str) -> None:
    """Name the malformed part of a serialized program."""
    document = json.loads(json.dumps(Program(2, (_measure(0, 0), Loop((_rotate(1),), count=2))).to_dict()))
    _set(document, path, value)

    with pytest.raises(ValueError, match=message):
        Program.from_dict(document)
