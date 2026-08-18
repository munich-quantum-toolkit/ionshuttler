# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for rebuilding and validating transformed DD schedules."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import AdvanceTime, GateSpec, GlobalPulse, Rx, Rz, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.schedule_transform import (
    insert_action_at_time,
    rebuild_result,
    validate_rebuilt_schedule,
)
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus, DDInsertionRecord, GlobalDDRecord
from mqt.ionshuttler.linear.state import create_initial_state


def _base_result() -> CompilationResult:
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1, 2]})
    return CompilationResult(
        status=CompilationStatus.TIMEOUT,
        path=[Shuttle(ion=0, src=0, dst=1), AdvanceTime(), AdvanceTime()],
        num_timesteps=2,
        wall_clock_s=1.25,
        architecture=architecture,
        initial_state=create_initial_state(1, architecture, initial_positions=[0]),
        dd_insertions=(DDInsertionRecord(ion=0, window=(1, 2), scheme_name="Hahn", gate_timesteps=()),),
        global_dd_records=(GlobalDDRecord(scheme_name="global", pulse_timesteps=(), spacing=2),),
        explored_nodes=17,
        action_types=("Shuttle", "Rx"),
    )


def test_insert_action_rebuilds_without_mutating_source() -> None:
    """Insert immediately before a boundary advance and preserve diagnostics."""
    original = _base_result()
    inserted = Rx(ion=0, theta=pi)

    rebuilt = insert_action_at_time(original, 1, inserted)

    assert original.path == [Shuttle(ion=0, src=0, dst=1), AdvanceTime(), AdvanceTime()]
    assert rebuilt.path == [Shuttle(ion=0, src=0, dst=1), AdvanceTime(), inserted, AdvanceTime()]
    assert rebuilt.status is original.status
    assert rebuilt.wall_clock_s == original.wall_clock_s
    assert rebuilt.explored_nodes == original.explored_nodes
    assert rebuilt.action_types == original.action_types
    assert rebuilt.dd_insertions == original.dd_insertions
    assert rebuilt.global_dd_records == original.global_dd_records
    assert CompilationResult.from_json(rebuilt.to_json()).to_dict() == rebuilt.to_dict()


def test_rebuild_result_reconstructs_makespan_and_final_placement() -> None:
    """Derive schedule state from the replacement action path."""
    original = _base_result()
    rebuilt = rebuild_result(
        original,
        [
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Shuttle(ion=0, src=1, dst=2),
            AdvanceTime(),
        ],
    )

    assert rebuilt.num_timesteps == 2
    assert rebuilt.final_state is not None
    assert rebuilt.final_state.positions == ((0, 2),)
    assert rebuilt.final_state.time == 2


def test_schedule_validation_accepts_concurrent_and_terminal_actions() -> None:
    """Accept a valid transport layer, destination gate, and terminal pulses."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1, 2]})
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[
            Shuttle(ion=0, src=0, dst=1),
            Rx(ion=1, theta=pi),
            AdvanceTime(),
            GlobalPulse(GateSpec("Rx", pi)),
            Rz(ion=0, theta=0.2),
        ],
        num_timesteps=1,
        architecture=architecture,
        initial_state=create_initial_state(2, architecture, initial_positions=[0, 2]),
    )

    assert validate_rebuilt_schedule(result)


def test_schedule_validation_rejects_conflicts_and_physical_terminal_gate() -> None:
    """Reject colliding transport and unfinished physical resource use."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [0, 1, 2]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 2])
    conflict = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[Shuttle(ion=0, src=0, dst=1), Shuttle(ion=1, src=2, dst=1), AdvanceTime()],
        num_timesteps=1,
        architecture=architecture,
        initial_state=initial_state,
    )
    terminal = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[Rx(ion=0, theta=pi)],
        num_timesteps=0,
        architecture=architecture,
        initial_state=initial_state,
    )

    assert not validate_rebuilt_schedule(conflict)
    assert not validate_rebuilt_schedule(terminal)


def test_transform_rejects_invalid_metadata_time_and_action() -> None:
    """Report malformed transform inputs without partially rebuilding them."""
    result = CompilationResult(status=CompilationStatus.SUCCESS, path=[], num_timesteps=0)
    with pytest.raises(ValueError, match="architecture"):
        rebuild_result(result, [])

    base = _base_result()
    with pytest.raises(ValueError, match="within"):
        insert_action_at_time(base, 3, Rz(ion=0, theta=0.2))
    with pytest.raises(ValueError, match="not valid"):
        insert_action_at_time(base, 0, Rx(ion=0, theta=pi))
