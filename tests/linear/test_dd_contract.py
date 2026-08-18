# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the frozen Linear dynamical-decoupling source contract."""

from __future__ import annotations

from typing import cast

import pytest

from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast("list[object]", value)


def _assert_no_runtime_fields(value: object) -> None:
    if isinstance(value, dict):
        assert "runtime_s" not in value
        assert "wall_clock_s" not in value
        for nested in value.values():
            _assert_no_runtime_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_runtime_fields(nested)


def test_sadd_golden_contract_is_complete_and_loadable(sadd_golden: dict[str, object]) -> None:
    """Keep each deterministic SADD observation normalized and portable."""
    assert sadd_golden["schema_version"] == 1
    assert sadd_golden["case"] in {"local_pulse", "transport_required"}
    assert sadd_golden["config"] == {
        "min_window_length": 2,
        "max_window_length": 16,
        "max_participating_ions": 5,
        "timeout_s": 10.0,
        "ion_preselection": "phase",
        "opportunity_order": "chronological",
        "max_accepted_windows": None,
        "improvement_tolerance": 1e-12,
        "allow_pulses": True,
        "scale": 1000,
        "operation_durations": {"shuttle": 1, "swap": 3, "one_qubit_gate": 1},
    }
    _assert_no_runtime_fields(sadd_golden)

    base = CompilationResult.from_dict(sadd_golden["base_result"])
    assert base.status is CompilationStatus.SUCCESS
    assert base.num_timesteps == 3
    assert base.architecture is not None
    assert [base.architecture.field_at(site) for site in range(3)] == pytest.approx([3.0, 1.0, 0.5])

    expected = _mapping(sadd_golden["expected"])
    for observation_raw in expected.values():
        observation = _mapping(observation_raw)
        result = CompilationResult.from_dict(observation["result"])
        assert result.status is CompilationStatus.SUCCESS
        assert result.architecture == base.architecture
        assert result.initial_state == base.initial_state
        assert observation["unavailable_reason"] is None
        assert len(_list(observation["records"])) == 1


def test_local_pulse_fixture_freezes_pulse_only_insertion(sadd_golden: dict[str, object]) -> None:
    """Freeze the source PulseOnlySADD schedule when local control is available."""
    if sadd_golden["case"] != "local_pulse":
        return

    observation = _mapping(_mapping(sadd_golden["expected"])["pulse_only_sadd"])
    result = CompilationResult.from_dict(observation["result"])
    record = _mapping(_list(observation["records"])[0])

    assert observation["allow_transport"] is False
    assert [type(action).__name__ for action in result.path] == [
        "AdvanceTime",
        "AdvanceTime",
        "Rx",
        "AdvanceTime",
    ]
    assert result.dd_insertions[0].gate_timesteps == (2,)
    assert record["accepted"] is True
    assert record["window"] == [0, 3]
    assert record["trajectories"] == {"0": [1, 1, 1]}
    assert record["pulse_timesteps"] == {"0": [2]}
    assert record["transport_action_count"] == 0
    assert record["objective_before"] == pytest.approx(2.6341463414634156)
    assert record["objective_after"] == pytest.approx(0.2926829268292684)
    assert observation["final_state"] == {
        "positions": [[0, 1]],
        "completed_gates": [],
        "in_progress_gates": [],
        "ions_busy_until": [[0, 3]],
        "pzs_busy_until": [["pz", 3]],
        "time": 3,
    }


def test_transport_fixture_distinguishes_sadd_methods(sadd_golden: dict[str, object]) -> None:
    """Freeze the source distinction between pulse-only and transport-enabled SADD."""
    if sadd_golden["case"] != "transport_required":
        return

    expected = _mapping(sadd_golden["expected"])
    pulse_only = _mapping(expected["pulse_only_sadd"])
    full = _mapping(expected["full_sadd"])
    pulse_only_result = CompilationResult.from_dict(pulse_only["result"])
    full_result = CompilationResult.from_dict(full["result"])
    pulse_only_record = _mapping(_list(pulse_only["records"])[0])
    full_record = _mapping(_list(full["records"])[0])

    assert pulse_only["allow_transport"] is False
    assert pulse_only_result.path == CompilationResult.from_dict(sadd_golden["base_result"]).path
    assert pulse_only_record["accepted"] is False
    assert pulse_only_record["objective_after"] == pytest.approx(pulse_only_record["objective_before"])
    assert pulse_only["final_state"] is None

    assert full["allow_transport"] is True
    assert [type(action).__name__ for action in full_result.path] == [
        "Shuttle",
        "AdvanceTime",
        "Rx",
        "AdvanceTime",
        "Shuttle",
        "AdvanceTime",
    ]
    assert full_result.dd_insertions[0].gate_timesteps == (1,)
    assert full_record["accepted"] is True
    assert full_record["trajectories"] == {"0": [1, 1, 0]}
    assert full_record["transport_action_count"] == 2
    assert full_record["objective_before"] == pytest.approx(23.70731707317073)
    assert full_record["objective_after"] == pytest.approx(2.6341463414634148)
    assert full["final_state"] == {
        "positions": [[0, 0]],
        "completed_gates": [],
        "in_progress_gates": [],
        "ions_busy_until": [[0, 3]],
        "pzs_busy_until": [["pz", 3]],
        "time": 3,
    }
