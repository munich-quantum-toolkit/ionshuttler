# Copyright (c) 2026 Munich Quantum Software Company GmbH
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
