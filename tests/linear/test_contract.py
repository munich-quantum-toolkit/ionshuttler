# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the frozen Linear compiler source contract."""

from __future__ import annotations

from typing import cast

# Frozen from the read-only IonSwapper source at commit 5446fddbfa0bb161da9919c45bacdc88d38ae5f8.
_SOURCE_COMMIT = "5446fddbfa0bb161da9919c45bacdc88d38ae5f8"


def _as_string_keyed_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def test_production_default_golden_contract(production_default_golden: dict[str, object]) -> None:
    """Keep the deterministic source-derived fixture complete and normalized."""
    assert production_default_golden["source_commit"] == _SOURCE_COMMIT

    expected_result = _as_string_keyed_dict(production_default_golden["expected_result"])
    assert "wall_clock_s" not in expected_result
    assert expected_result["status"] == "SUCCESS"
    assert expected_result["num_timesteps"] == 5
    assert expected_result["score"] == 5
    assert expected_result["dd_insertions"] == []
    assert expected_result["global_dd_records"] == []

    actions = expected_result["actions"]
    assert isinstance(actions, list)
    assert [_as_string_keyed_dict(action)["type"] for action in actions] == [
        "Shuttle",
        "Shuttle",
        "AdvanceTime",
        "Rx",
        "AdvanceTime",
        "Ry",
        "AdvanceTime",
        "Rzz",
        "AdvanceTime",
        "AdvanceTime",
    ]

    final_state = _as_string_keyed_dict(production_default_golden["expected_final_state"])
    assert final_state["completed_gates"] == [0, 1, 2]
    assert final_state["time"] == 5


def test_production_default_golden_contains_no_gratuitous_idle_time(
    production_default_golden: dict[str, object],
) -> None:
    """Freeze the source schedule's pending-work-only time-advance policy."""
    expected_result = _as_string_keyed_dict(production_default_golden["expected_result"])
    actions = expected_result["actions"]
    assert isinstance(actions, list)

    scheduled_until = 0
    advance_times: list[int] = []
    for raw_action in actions:
        action = _as_string_keyed_dict(raw_action)
        start_time = action["start_time"]
        duration = action["duration"]
        assert isinstance(start_time, int)
        assert isinstance(duration, int)
        if action["type"] == "AdvanceTime":
            assert start_time < scheduled_until
            advance_times.append(start_time)
        else:
            scheduled_until = max(scheduled_until, start_time + duration)

    assert advance_times == [0, 1, 2, 3, 4]
