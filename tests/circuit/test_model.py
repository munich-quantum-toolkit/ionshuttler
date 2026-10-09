# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the internal circuit model."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.circuit.parser import parse_circuit
from mqt.ionshuttler.core.gates import Rx, Rxx

if TYPE_CHECKING:
    from collections.abc import Callable


def test_circuit_exposes_stable_gate_ids() -> None:
    """Gate identifiers follow immutable circuit order."""
    circuit = Circuit(
        num_ions=2,
        gates=(
            Rx(ion=0, theta=0.5, gate_id=0),
            Rxx(ion_a=0, ion_b=1, theta=1.0, gate_id=1),
        ),
        predecessors=(frozenset(), frozenset({0})),
    )

    assert circuit.gate_ids == range(2)


def test_circuit_rejects_forward_dependencies() -> None:
    """A gate can depend only on an earlier gate occurrence."""
    gate = Rx(ion=0, theta=0.5, gate_id=0)

    with pytest.raises(ValueError, match="each predecessor must identify an earlier gate"):
        Circuit(num_ions=1, gates=(gate,), predecessors=(frozenset({0}),))


@pytest.mark.parametrize(
    ("build", "error", "message"),
    [
        pytest.param(
            lambda: Circuit(num_ions=cast("int", object()), gates=(), predecessors=()),
            TypeError,
            "num_ions must be an integer",
            id="boolean-ion-count",
        ),
        pytest.param(
            lambda: Circuit(
                num_ions=1,
                gates=(Rx(ion=0, theta=0.5, gate_id=0),),
                predecessors=(),
            ),
            ValueError,
            "one entry for each gate",
            id="dependency-count",
        ),
        pytest.param(
            lambda: Circuit(
                num_ions=1,
                gates=(Rx(ion=0, theta=0.5, gate_id=1),),
                predecessors=(frozenset(),),
            ),
            ValueError,
            "stable circuit-order gate_id",
            id="gate-id-order",
        ),
    ],
)
def test_circuit_rejects_malformed_structure(
    build: Callable[[], object],
    error: type[Exception],
    message: str,
) -> None:
    """Reject circuit values that violate stable compiler identities."""
    with pytest.raises(error, match=message):
        build()


def test_parsed_circuit_always_retains_dependencies() -> None:
    """Parsing preserves the dependency graph for compiler policies."""
    circuit = parse_circuit(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        rx(0.5) q[0];
        ry(0.25) q[1];
        rzz(1.0) q[0], q[1];
        """
    )

    assert circuit.predecessors == (frozenset(), frozenset(), frozenset({0, 1}))
    assert tuple(gate.gate_id for gate in circuit.gates) == (0, 1, 2)
