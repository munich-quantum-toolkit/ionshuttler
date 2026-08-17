# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear compilation results and serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from mqt.ionshuttler.linear.actions import (
    Action,
    AdvanceTime,
    GateSpec,
    GlobalPulse,
    PhysicalSwap,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    Shuttle,
)
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.result import (
    CompilationResult,
    CompilationStatus,
    DDInsertionRecord,
    GlobalDDRecord,
)
from mqt.ionshuttler.linear.state import State, create_initial_state


@dataclass(frozen=True)
class _Marker(Action):
    """Inert action used to exercise an explicitly supplied result decoder."""

    label: str

    def apply(self, state: State, architecture: Architecture) -> State:
        """Return an unchanged copy of the state."""
        del architecture
        return replace(state)


def test_result_exports_schedule_and_metadata() -> None:
    """Attach schedule times and portable architecture metadata."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [1, 2]})
    initial_state = create_initial_state(2, architecture, initial_positions=[0, 3])
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[Shuttle(ion=0, src=0, dst=1), AdvanceTime(), Rzz(ion_a=0, ion_b=1, theta=0.5)],
        num_timesteps=1,
        architecture=architecture,
        initial_state=initial_state,
    )

    exported = result.to_dict()
    loaded = CompilationResult.from_dict(exported)

    assert exported == {
        "metadata": {
            "architecture": {
                "num_sites": 5,
                "processing_zones": {"pz": [1, 2]},
            },
            "initial_state": {"site_occupancy": [0, None, None, 1, None]},
        },
        "status": "SUCCESS",
        "num_timesteps": 1,
        "wall_clock_s": 0.0,
        "score": None,
        "action_types": [],
        "actions": [
            {
                "type": "Shuttle",
                "ion": 0,
                "src": 0,
                "dst": 1,
                "start_time": 0,
                "duration": 1,
            },
            {"type": "AdvanceTime", "start_time": 0, "duration": 1},
            {
                "type": "Rzz",
                "ion_a": 0,
                "ion_b": 1,
                "theta": 0.5,
                "start_time": 1,
                "duration": 2,
            },
        ],
        "dd_insertions": [],
        "global_dd_records": [],
    }
    assert loaded.architecture == architecture
    assert loaded.initial_state == initial_state


def test_result_round_trips_every_builtin_action() -> None:
    """Restore built-in actions including nondefault virtual implementations."""
    actions: list[Action] = [
        Shuttle(ion=0, src=0, dst=1, duration=2),
        PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2, duration=3),
        Rx(ion=0, theta=0.1, duration=2),
        Ry(ion=1, theta=0.2),
        Rz(ion=0, theta=0.3, duration=1, virtual=False),
        Rx(ion=1, theta=0.4, duration=0, virtual=True),
        Rxx(ion_a=0, ion_b=1, theta=0.5, duration=2),
        Ryy(ion_a=0, ion_b=1, theta=0.6, duration=2),
        Rzz(ion_a=0, ion_b=1, theta=0.7, duration=2),
        GlobalPulse(gate=GateSpec("Rx", theta=0.8), duration=2),
        AdvanceTime(),
    ]
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=actions,
        num_timesteps=1,
        wall_clock_s=0.25,
        score=7,
    )

    loaded = CompilationResult.from_json(result.to_json())

    assert loaded.path == actions
    assert loaded.status is CompilationStatus.SUCCESS
    assert loaded.num_timesteps == 1
    assert loaded.wall_clock_s == pytest.approx(0.25)
    assert loaded.score == 7
    assert loaded.to_dict() == result.to_dict()


def test_two_qubit_action_without_serialized_duration_uses_hardware_default() -> None:
    """Use the standard two-timestep duration when serialized data omits the field."""
    loaded = CompilationResult.from_dict({
        "actions": [
            {
                "type": "Rzz",
                "ion_a": 0,
                "ion_b": 1,
                "theta": 0.5,
            },
        ],
    })

    assert loaded.path == [Rzz(ion_a=0, ion_b=1, theta=0.5)]


def test_default_virtuality_is_omitted_but_nondefault_modes_are_explicit() -> None:
    """Keep ordinary JSON compact without losing hardware distinctions."""
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[Rx(ion=0, theta=0.1), Rz(ion=0, theta=0.2), Rz(ion=0, theta=0.3, duration=1, virtual=False)],
        num_timesteps=0,
    )

    actions = cast("list[dict[str, object]]", result.to_dict()["actions"])

    assert isinstance(actions, list)
    assert "virtual" not in actions[0]
    assert "virtual" not in actions[1]
    assert actions[2]["virtual"] is False
    assert CompilationResult.from_json(result.to_json()).path == result.path


def test_zero_duration_physical_gate_round_trips_with_warning() -> None:
    """Preserve an explicitly physical instantaneous implementation."""
    with pytest.warns(UserWarning, match="physical single-qubit gate has zero duration"):
        gate = Rz(ion=0, theta=0.3, duration=0, virtual=False)
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[gate],
        num_timesteps=0,
    )

    with pytest.warns(UserWarning, match="physical single-qubit gate has zero duration"):
        loaded = CompilationResult.from_json(result.to_json())

    assert loaded.path == [gate]


def test_result_round_trips_passive_control_records() -> None:
    """Preserve control metadata without invoking any control algorithm."""
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[],
        num_timesteps=0,
        dd_insertions=(
            DDInsertionRecord(
                ion=1,
                window=(2, 8),
                scheme_name="echo",
                gate_timesteps=(4, 6),
                remaining_phase=0.25,
                phase_reduction=1.5,
                residual_phase_at_window_end_before=2.0,
                residual_phase_at_window_end_after=0.5,
                residual_phase_at_window_end_reduction=1.5,
            ),
        ),
        global_dd_records=(
            GlobalDDRecord(
                scheme_name="periodic_x",
                pulse_timesteps=(1, 5),
                spacing=4,
                sum_abs_residual_phase=1.0,
                sum_squared_residual_phase=0.75,
                max_abs_residual_phase=0.5,
            ),
        ),
    )

    loaded = CompilationResult.from_json(result.to_json())

    assert loaded.dd_insertions == result.dd_insertions
    assert loaded.global_dd_records == result.global_dd_records


def test_result_reads_older_control_record_keys() -> None:
    """Accept earlier names retained by saved schedule files."""
    loaded = CompilationResult.from_dict({
        "dd_insertions": [
            {
                "ion": 0,
                "window": [1, 3],
                "scheme_name": "echo",
                "pulse_timesteps": [2],
                "mismatch": 0.75,
            },
        ],
    })

    assert loaded.status is CompilationStatus.FAILED
    assert loaded.dd_insertions == (
        DDInsertionRecord(
            ion=0,
            window=(1, 3),
            scheme_name="echo",
            gate_timesteps=(2,),
            remaining_phase=0.75,
        ),
    )


@pytest.mark.parametrize("status", list(CompilationStatus))
def test_every_status_round_trips(status: CompilationStatus) -> None:
    """Preserve every way compilation can finish."""
    result = CompilationResult(status=status, path=[AdvanceTime()], num_timesteps=1)

    assert CompilationResult.from_json(result.to_json()).status is status


def test_result_save_and_load(tmp_path: Path) -> None:
    """Write explicit result files and restore them unchanged."""
    result = CompilationResult(
        status=CompilationStatus.INTERRUPTED,
        path=[AdvanceTime()],
        num_timesteps=1,
        wall_clock_s=0.1,
    )

    output_path = result.save("schedule", directory=tmp_path)
    loaded = CompilationResult.load(output_path)

    assert output_path == tmp_path / "schedule.json"
    assert output_path.read_text(encoding="utf-8") == result.to_json()
    assert loaded.to_dict() == result.to_dict()


def test_result_save_uses_a_stable_working_directory_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep default result files beneath the current working directory."""
    monkeypatch.chdir(tmp_path)
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[],
        num_timesteps=0,
    )

    output_path = result.save("schedule")

    assert output_path == Path("outputs/results/json/schedule.json")
    assert (tmp_path / output_path).is_file()


def test_result_requires_architecture_for_initial_state() -> None:
    """Reject initial-state metadata that has no hardware interpretation."""
    initial_state = create_initial_state(1, Architecture(num_sites=1))

    with pytest.raises(ValueError, match="architecture is required"):
        CompilationResult(
            status=CompilationStatus.SUCCESS,
            path=[],
            num_timesteps=0,
            initial_state=initial_state,
        )


def test_result_rejects_duplicate_capability_names() -> None:
    """Keep serialized capability provenance unambiguous."""
    with pytest.raises(ValueError, match="must not contain duplicates"):
        CompilationResult(
            status=CompilationStatus.SUCCESS,
            path=[],
            num_timesteps=0,
            action_types=("Rx", "Rx"),
        )


def test_result_omits_search_only_state_from_json() -> None:
    """Keep runtime search state outside the portable result schema."""
    architecture = Architecture(num_sites=1)
    state = create_initial_state(1, architecture)
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[],
        num_timesteps=0,
        final_state=state,
        explored_nodes=12,
    )

    exported = result.to_dict()
    loaded = CompilationResult.from_dict(exported)

    assert "final_state" not in exported
    assert "explored_nodes" not in exported
    assert loaded.final_state is None
    assert loaded.explored_nodes is None


def test_custom_action_uses_its_action_owned_decoder() -> None:
    """Round-trip a custom action by supplying its class catalog."""
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[_Marker("calibration")],
        num_timesteps=0,
        action_types=("_Marker",),
    )

    with pytest.raises(ValueError, match="unknown action type"):
        CompilationResult.from_json(result.to_json())

    loaded = CompilationResult.from_json(
        result.to_json(),
        action_types=(_Marker,),
    )

    assert loaded.path == result.path
    assert loaded.action_types == ("_Marker",)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "JSON object"),
        ({"actions": {}}, "actions must be a list"),
        ({"status": "UNKNOWN"}, "unknown compilation status"),
        ({"num_timesteps": 1.5}, "num_timesteps must be an integer"),
        ({"wall_clock_s": "slow"}, "wall_clock_s must be numeric"),
        ({"score": 1.5}, "score must be an integer or null"),
        ({"action_types": ["Rx", 1]}, "action_types must be a list of strings"),
        ({"actions": [None]}, "each action must be a JSON object"),
        ({"actions": [{}]}, "type must be a string"),
        ({"actions": [{"type": "Rx", "ion": False, "theta": 0.1}]}, "ion must be an integer"),
        ({"actions": [{"type": "Rx", "ion": 0, "theta": "bad"}]}, "theta must be numeric"),
        (
            {"actions": [{"type": "Rx", "ion": 0, "theta": 0.1, "duration": "bad"}]},
            "duration must be an integer",
        ),
        (
            {"actions": [{"type": "Rx", "ion": 0, "theta": 0.1, "virtual": "yes"}]},
            "virtual must be a boolean",
        ),
        (
            {"actions": [{"type": "Rx", "ion": 0, "theta": 0.1, "duration": -1}]},
            "action duration must be >= 0",
        ),
        (
            {"actions": [{"type": "Shuttle", "ion": 0, "src": 0, "dst": 1, "duration": 0}]},
            "action duration must be >= 1",
        ),
        ({"dd_insertions": [{}]}, "window must be a list of integers"),
        (
            {
                "dd_insertions": [
                    {"ion": 0, "window": [0], "scheme_name": "echo", "gate_timesteps": []},
                ],
            },
            "window must have length 2",
        ),
        (
            {
                "dd_insertions": [
                    {
                        "ion": 0,
                        "window": [0, 1],
                        "scheme_name": "echo",
                        "gate_timesteps": [],
                        "remaining_phase": "bad",
                    },
                ],
            },
            "remaining_phase must be numeric",
        ),
        (
            {
                "dd_insertions": [
                    {
                        "ion": 0,
                        "window": [0, 1],
                        "scheme_name": "echo",
                        "gate_timesteps": [],
                        "residual_phase_at_window_end_before": "bad",
                    },
                ],
            },
            "residual_phase_at_window_end_before must be numeric or null",
        ),
        ({"global_dd_records": [{}]}, "scheme_name must be a string"),
        (
            {
                "metadata": {
                    "architecture": {"num_sites": 1, "processing_zones": {"pz": [0]}},
                    "initial_state": {"site_occupancy": ["ion"]},
                },
            },
            "site_occupancy must contain integers or null",
        ),
    ],
)
def test_result_rejects_malformed_input(data: object, message: str) -> None:
    """Reject incomplete or incorrectly typed serialized fields."""
    with pytest.raises(ValueError, match=message):
        CompilationResult.from_dict(data)


@pytest.mark.parametrize(
    "metadata",
    [
        {"architecture": {"num_sites": 1, "processing_zones": {"pz": [0]}}},
        {
            "architecture": {"num_sites": 1, "processing_zones": {"pz": [0]}},
            "initial_state": {},
        },
    ],
)
def test_incomplete_initial_state_metadata_is_ignored(metadata: dict[str, object]) -> None:
    """Load architecture metadata even when initial placement is absent."""
    loaded = CompilationResult.from_dict({"metadata": metadata})

    assert loaded.architecture == Architecture(num_sites=1, processing_zones={"pz": [0]})
    assert loaded.initial_state is None


def test_to_json_produces_a_json_object() -> None:
    """Emit ordinary JSON consumable without package-specific hooks."""
    raw = CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=[],
        num_timesteps=0,
    ).to_json()

    assert isinstance(json.loads(raw), dict)
