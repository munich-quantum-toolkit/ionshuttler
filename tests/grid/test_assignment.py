# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Grid gate-to-processing-zone assignment."""

from __future__ import annotations

from mqt.ionshuttler.circuit import parse_circuit
from mqt.ionshuttler.grid import GridArchitecture, Junction, ProcessingZone, Segment
from mqt.ionshuttler.grid.assignment import compute_gate_zone_assignment


def test_partitioning_excludes_processing_zone_segments_that_are_too_small() -> None:
    """Pass the partitioner only zones that can contain the largest gate."""
    undersized = Segment("undersized", processing_zones=(ProcessingZone("undersized-pz"),))
    left = Segment("left", capacity=2, processing_zones=(ProcessingZone("left-pz"),))
    right = Segment("right", capacity=2, processing_zones=(ProcessingZone("right-pz"),))
    architecture = GridArchitecture(
        (undersized, left, right),
        (Junction("junction", (undersized.end, left.start, right.start)),),
    )
    circuit = parse_circuit(
        "\n".join((
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "qreg q[2];",
            "rx(0.5) q[0];",
            "ry(0.25) q[1];",
            "rxx(0.125) q[0],q[1];",
        ))
    )

    assignment = compute_gate_zone_assignment(circuit, architecture)

    assert assignment
    assert set(assignment.values()) <= {"left-pz", "right-pz"}
