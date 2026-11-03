# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the dynamic program model."""

from __future__ import annotations

import json

import pytest

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.core.gates import Measure, Rx
from mqt.ionshuttler.dynamic import Branch, Condition, Loop, Program


def _measure(ion: int, outcome: int) -> Circuit:
    return Circuit(
        num_ions=2,
        gates=(Measure(ion=ion, outcome=outcome, gate_id=0),),
        predecessors=(frozenset(),),
    )


def _rotate(ion: int) -> Circuit:
    return Circuit(num_ions=2, gates=(Rx(ion=ion, theta=0.5, gate_id=0),), predecessors=(frozenset(),))


def test_condition_holds_for_its_listed_outcome_combinations() -> None:
    """Evaluate a condition as a truth table over measurement outcomes."""
    parity = Condition((0, 2), frozenset({(0, 1), (1, 0)}))

    assert parity.holds({0: 1, 1: 1, 2: 0})
    assert not parity.holds({0: 1, 2: 1})
    assert Condition.from_dict(json.loads(json.dumps(parity.to_dict()))) == parity
    with pytest.raises(ValueError, match="unique and ascending"):
        Condition((2, 0), frozenset())
    with pytest.raises(ValueError, match="one 0/1 value per condition outcome"):
        Condition((0,), frozenset({(2,)}))


def test_program_enumerates_one_assignment_per_path() -> None:
    """Assign only the outcomes that decide branches on each path."""
    inner = Branch(Condition((1,), frozenset({(1,)})), (_rotate(1),))
    program = Program(
        2,
        (_measure(0, 0), Branch(Condition((0,), frozenset({(1,)})), (_measure(1, 1), inner), (_rotate(0),))),
    )

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
    only_then = Branch(Condition((0,), frozenset({(1,)})), (_measure(1, 1),))
    with pytest.raises(ValueError, match=r"reads outcomes \[1\] before they are measured"):
        Program(2, (_measure(0, 0), only_then, Branch(Condition((1,), frozenset({(1,)})), (_rotate(0),))))
    with pytest.raises(ValueError, match="exactly one measurement"):
        Program(2, (_measure(0, 0), _measure(1, 0)))
    with pytest.raises(ValueError, match="act on num_ions ions"):
        Program(3, (_rotate(0),))


ALWAYS = Condition((), frozenset({()}))
OUTCOME_ZERO_IS_ONE = Condition((0,), frozenset({(1,)}))


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


def test_paths_through_a_loop_without_count_stop_at_max_iterations() -> None:
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
