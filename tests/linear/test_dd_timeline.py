# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for dynamical-decoupling timeline reconstruction."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, GlobalPulse, PhysicalSwap, Rz, Rzz, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.state import State, create_initial_state


def test_timeline_reconstructs_positions_resources_and_action_order() -> None:
    """Match source schedule-boundary and duration-occupancy semantics."""
    architecture = Architecture(num_sites=4, processing_zones={"pz": [1, 2]})
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Rzz(ion_a=0, ion_b=1, theta=0.5, duration=2),
            AdvanceTime(),
            AdvanceTime(),
            PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2),
            AdvanceTime(),
        ],
        num_timesteps=4,
        architecture=architecture,
        initial_state=create_initial_state(2, architecture, initial_positions=[0, 2]),
    )

    timeline = build_timeline(result)

    assert [timeline.ion_position(0, timestep) for timestep in range(5)] == [1, 1, 1, 2, 2]
    assert [timeline.ion_position(1, timestep) for timestep in range(5)] == [2, 2, 2, 1, 1]
    assert {timestep for timestep in range(5) if timeline.ion_busy(0, timestep)} == {0, 1, 2, 3}
    assert {timestep for timestep in range(5) if timeline.ion_busy(1, timestep)} == {1, 2, 3}
    assert {timestep for timestep in range(5) if timeline.pz_busy("pz", timestep)} == {1, 2}
    assert timeline.action_at(1) == (Rzz(ion_a=0, ion_b=1, theta=0.5, duration=2), AdvanceTime())


def test_timeline_preserves_same_boundary_and_terminal_action_order() -> None:
    """Keep global pulses and terminal virtual gates in path order."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    pulse = GlobalPulse(gate=GateSpec("Rx", theta=pi))
    terminal = Rz(ion=0, theta=0.25)
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[pulse, Shuttle(ion=0, src=0, dst=1), AdvanceTime(), terminal],
        num_timesteps=1,
        architecture=architecture,
        initial_state=create_initial_state(1, architecture, initial_positions=[0]),
    )

    timeline = build_timeline(result)

    assert timeline.action_at(0) == (pulse, Shuttle(ion=0, src=0, dst=1), AdvanceTime())
    assert timeline.action_at(1) == (terminal,)
    assert timeline.ion_position(0, 0) == 1
    assert timeline.ion_position(0, 1) == 1


def test_timeline_distinguishes_virtual_and_physical_rz_resources() -> None:
    """Treat only virtual rotations as resource-free in the Linear model."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    initial_state = create_initial_state(1, architecture)
    virtual = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[Rz(ion=0, theta=0.2), AdvanceTime()],
        num_timesteps=1,
        architecture=architecture,
        initial_state=initial_state,
    )
    physical = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[Rz(ion=0, theta=0.2, duration=1, virtual=False), AdvanceTime()],
        num_timesteps=1,
        architecture=architecture,
        initial_state=initial_state,
    )

    assert not build_timeline(virtual).ion_gate_busy(0, 0)
    assert build_timeline(physical).ion_gate_busy(0, 0)
    assert build_timeline(physical).pz_busy("pz", 0)


@pytest.mark.parametrize(
    ("num_timesteps", "path", "message"),
    [
        (-1, [], "non-negative"),
        (0, [AdvanceTime(), AdvanceTime()], "advances beyond"),
        (2, [AdvanceTime()], "does not match"),
    ],
)
def test_timeline_rejects_malformed_makespans(
    num_timesteps: int,
    path: list[Action],
    message: str,
) -> None:
    """Reject negative, overlong, and underlong schedule clocks."""
    architecture = Architecture(num_sites=1)
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=path,
        num_timesteps=num_timesteps,
        architecture=architecture,
        initial_state=create_initial_state(1, architecture),
    )

    with pytest.raises(ValueError, match=message):
        build_timeline(result)


def test_timeline_requires_metadata_and_valid_query_boundaries() -> None:
    """Report missing reconstruction inputs and out-of-range queries clearly."""
    result = CompilationResult(status=CompilationStatus.SUCCESS, path=[], num_timesteps=0)
    with pytest.raises(ValueError, match="architecture"):
        build_timeline(result)

    architecture = Architecture(num_sites=1)
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[],
        num_timesteps=0,
        architecture=architecture,
        initial_state=State(
            positions=((0, 0),),
            completed_gates=frozenset(),
            in_progress_gates=(),
            ions_busy_until=((0, 0),),
            pzs_busy_until=architecture.initial_pzs_busy_until(),
            time=0,
        ),
    )
    timeline = build_timeline(result)
    invalid_timestep: int = True
    with pytest.raises(ValueError, match="within"):
        timeline.state_at(1)
    with pytest.raises(TypeError, match="integer"):
        timeline.action_at(invalid_timestep)
