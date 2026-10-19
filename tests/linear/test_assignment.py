# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear gate-to-processing-zone assignment."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.core.gates import GlobalGate, Rz, Rzz
from mqt.ionshuttler.linear import assignment
from mqt.ionshuttler.linear.architecture import LinearArchitecture


def test_single_zone_assignment_does_not_call_partitioner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the no-op assignment before invoking tabu search."""

    def unexpected_partitioner(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the partitioner must not run for one processing zone")

    monkeypatch.setattr(assignment, "compute_fine_grained_gate_partition", unexpected_partitioner)
    architecture = LinearArchitecture(num_sites=4, processing_zones={"only": [1, 2]})
    circuit = Circuit(
        num_ions=2,
        gates=(Rzz(ion_a=0, ion_b=1, theta=0.5, gate_id=0),),
        predecessors=(frozenset(),),
    )

    assert assignment.compute_gate_zone_assignment(circuit, architecture) == {}


def test_assignment_ignores_gates_that_need_no_processing_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return no preferences when no gate needs a processing zone."""

    def unexpected_partitioner(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the partitioner must not run without physical gate operands")

    monkeypatch.setattr(assignment, "compute_fine_grained_gate_partition", unexpected_partitioner)
    architecture = LinearArchitecture(num_sites=4, processing_zones={"left": [0, 1], "right": [2, 3]})
    circuit = Circuit(
        num_ions=2,
        gates=(
            GlobalGate(gate_name="rx", theta=0.5, ions=(0, 1), gate_id=0),
            Rz(ion=0, theta=0.5, gate_id=1),
        ),
        predecessors=(frozenset(), frozenset()),
    )

    assert assignment.compute_gate_zone_assignment(circuit, architecture) == {}


def test_assignment_passes_each_processing_zone_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Derive the partitioner's soft capacities from each zone's sites."""
    received_capacities: dict[str, int] | None = None

    def partition(*_args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal received_capacities
        received_capacities = cast("dict[str, int]", kwargs["processing_zone_capacities"])
        return SimpleNamespace(gate_assignment={0: "left"})

    monkeypatch.setattr(assignment, "compute_fine_grained_gate_partition", partition)
    architecture = LinearArchitecture(
        num_sites=6,
        processing_zones={"left": [0, 1, 2], "right": [4, 5]},
    )
    circuit = Circuit(
        num_ions=2,
        gates=(Rzz(ion_a=0, ion_b=1, theta=0.5, gate_id=0),),
        predecessors=(frozenset(),),
    )

    assert assignment.compute_gate_zone_assignment(circuit, architecture) == {0: "left"}
    assert received_capacities == {"left": 3, "right": 2}


def test_zone_site_pairs_are_scoped_to_each_processing_zone() -> None:
    """Exclude cross-zone pairs from a zone-specific heuristic target."""
    architecture = LinearArchitecture(
        num_sites=8,
        processing_zones={"left": [1, 2, 3], "right": [6, 7]},
    )

    assert assignment.zone_site_pairs(architecture) == {
        "left": ((1, 2), (1, 3), (2, 3)),
        "right": ((6, 7),),
    }
