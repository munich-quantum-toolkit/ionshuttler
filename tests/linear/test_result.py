# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for action schedules and compiler results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi
from typing import TYPE_CHECKING

import pytest

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, GateSpec, GlobalPulse, Rx, Rz, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.schedule import ActionSchedule, MachineState
from mqt.ionshuttler.linear.state import State, create_initial_state

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class _Marker(Action):
    """Inert custom action used to exercise explicit action codecs."""

    label: str

    def apply(self, state: State, architecture: Architecture) -> State:
        """Return an unchanged copy of the state."""
        del architecture
        return replace(state)


def _program(actions: list[Action] | None = None) -> ActionSchedule:
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    return ActionSchedule.from_actions(
        actions or [AdvanceTime()],
        architecture,
        create_initial_state(1, architecture, initial_positions=[0]),
        action_types=("Shuttle", "Rx"),
    )


def test_action_schedule_round_trips_identity_and_machine_metadata() -> None:
    """Preserve the complete action-level execution boundary."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    program = ActionSchedule.from_actions(
        [
            Shuttle(ion=0, src=0, dst=1),
            Rx(ion=0, theta=pi),
            GlobalPulse(GateSpec("Rx", pi)),
            AdvanceTime(),
            Rz(ion=0, theta=0.2),
        ],
        architecture,
        create_initial_state(1, architecture),
    )

    restored = ActionSchedule.from_json(program.to_json())

    assert restored == program
    assert [item.action_id for item in restored.scheduled_actions] == list(range(5))
    serialized = restored.to_dict()
    actions = serialized["actions"]
    assert isinstance(actions, list)
    assert isinstance(actions[0], dict)
    assert "purpose" not in actions[0]
    assert "schema" not in serialized
    assert "version" not in serialized


def test_machine_state_strips_compiler_progress_and_canonicalizes_availability() -> None:
    """Expose replay state without leaking circuit-search progress."""
    state = State(
        positions=((0, 0),),
        completed_gates=frozenset({3}),
        in_progress_gates=((4, 7),),
        ions_busy_until=((0, 2),),
        pzs_busy_until=(("pz", 1),),
        time=5,
    )

    machine = MachineState.from_compiler_state(state)
    replay = machine.to_replay_state()

    assert machine.ions_busy_until == ((0, 5),)
    assert machine.pzs_busy_until == (("pz", 5),)
    assert replay.completed_gates == frozenset()
    assert replay.in_progress_gates == ()


def test_compilation_result_round_trips_only_compiler_diagnostics() -> None:
    """Keep compiler outcome fields outside the nested scheduled program."""
    program = _program()
    final_state = replace(program.initial_state.to_replay_state(), completed_gates=frozenset({0}), time=1)
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        wall_clock_s=0.5,
        score=1,
        final_state=final_state,
        explored_nodes=7,
    )

    restored = CompilationResult.from_json(result.to_json())

    assert restored == result
    assert "schedule" in restored.to_dict()
    assert "schema" not in restored.to_dict()
    assert "version" not in restored.to_dict()
    assert "dd_insertions" not in restored.to_dict()


def test_artifacts_support_custom_action_types_and_files(tmp_path: Path) -> None:
    """Use explicit codecs for extensions and load UTF-8 artifacts from disk."""
    architecture = Architecture(num_sites=1)
    program = ActionSchedule.from_actions([_Marker("probe")], architecture, create_initial_state(1, architecture))
    path = program.save(tmp_path / "program")

    with pytest.raises(ValueError, match="unknown action type"):
        ActionSchedule.load(path)
    assert ActionSchedule.load(path, action_types=(_Marker,)) == program
