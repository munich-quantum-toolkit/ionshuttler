# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared topology cases for supported Grid compiler configurations."""

from __future__ import annotations

from dataclasses import dataclass

from mqt.ionshuttler.core.result import CompilationStatus
from mqt.ionshuttler.grid import GridArchitecture, GridStopReason, Junction, ProcessingZone, Segment


@dataclass(frozen=True)
class TopologyCase:
    """Define one compiler-independent topology and its expected outcome."""

    name: str
    architecture: GridArchitecture
    initial_placement: dict[str, tuple[int, ...]]
    qasm: str
    expected_status: CompilationStatus
    expected_stop_reason: GridStopReason


def topology_cases() -> tuple[TopologyCase, ...]:
    """Return representative topology cases for Grid compiler acceptance."""
    return (
        _bridge_case(),
        _dead_end_case(),
        _irregular_junction_case(),
        _saturated_acyclic_case(),
    )


def _bridge_case() -> TopologyCase:
    source = Segment("source")
    bridge = Segment("bridge")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (source, bridge, processor),
        (
            Junction("source-bridge", (source.end, bridge.start)),
            Junction("bridge-processor", (bridge.end, processor.start)),
        ),
    )
    return TopologyCase(
        name="bridge",
        architecture=architecture,
        initial_placement={"source": (0,)},
        qasm=_single_rx_qasm(),
        expected_status=CompilationStatus.SUCCESS,
        expected_stop_reason=GridStopReason.COMPLETED,
    )


def _dead_end_case() -> TopologyCase:
    source = Segment("source")
    dead_end = Segment("dead-end")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (source, dead_end, processor),
        (Junction("branch", (source.end, dead_end.start, processor.start)),),
    )
    return TopologyCase(
        name="dead-end-branch",
        architecture=architecture,
        initial_placement={"source": (0,)},
        qasm=_single_rx_qasm(),
        expected_status=CompilationStatus.SUCCESS,
        expected_stop_reason=GridStopReason.COMPLETED,
    )


def _irregular_junction_case() -> TopologyCase:
    source = Segment("source")
    first_branch = Segment("first-branch")
    second_branch = Segment("second-branch")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (source, first_branch, second_branch, processor),
        (Junction("four-way", (source.end, first_branch.start, second_branch.end, processor.start)),),
    )
    return TopologyCase(
        name="irregular-four-way-junction",
        architecture=architecture,
        initial_placement={"source": (0,)},
        qasm=_single_rx_qasm(),
        expected_status=CompilationStatus.SUCCESS,
        expected_stop_reason=GridStopReason.COMPLETED,
    )


def _saturated_acyclic_case() -> TopologyCase:
    source = Segment("source")
    middle = Segment("middle")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (source, middle, processor),
        (
            Junction("source-middle", (source.end, middle.start)),
            Junction("middle-processor", (middle.end, processor.start)),
        ),
    )
    return TopologyCase(
        name="saturated-acyclic-layout",
        architecture=architecture,
        initial_placement={"source": (0,), "middle": (1,), "processor": (2,)},
        qasm=_qasm(3, "rx(0.5) q[0];"),
        expected_status=CompilationStatus.FAILED,
        expected_stop_reason=GridStopReason.NO_PROGRESS,
    )


def _single_rx_qasm() -> str:
    return _qasm(1, "rx(0.5) q[0];")


def _qasm(num_ions: int, *instructions: str) -> str:
    return "\n".join(("OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{num_ions}];", *instructions))


__all__ = ["TopologyCase", "topology_cases"]
