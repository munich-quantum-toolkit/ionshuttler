"""Tests for conflict-resolution mode config validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.ionshuttler.multi_shuttler import main as main_module

if TYPE_CHECKING:
    from collections.abc import Callable


def _expect_validator_raises_with_text(
    validator: Callable[[dict[str, object]], object],
    expected_exception: type[Exception],
    expected_text: str,
    payload: dict[str, object],
) -> None:
    """Call validator and verify exception type/message."""
    try:
        validator(payload)
    except expected_exception as exc:
        if expected_text not in str(exc):
            msg = f"Expected '{expected_text}' in exception message, got: {exc!s}"
            raise AssertionError(msg) from exc
    else:
        msg = f"Expected {expected_exception.__name__} to be raised."
        raise AssertionError(msg)


def test_validate_conflict_resolution_mode_accepts_valid_values() -> None:
    cases = [
        ("cycles", "cycles"),
        ("paths", "paths"),
        ("hybrid", "hybrid"),
        (" CyClEs ", "cycles"),
    ]

    for raw_value, expected in cases:
        result = main_module.validate_conflict_resolution_mode({"use_cycle_or_paths": raw_value})
        assert result == expected


def test_validate_conflict_resolution_mode_rejects_invalid_types() -> None:
    for bad_value in [False, True, 0, 1, 1.2, None]:
        _expect_validator_raises_with_text(
            main_module.validate_conflict_resolution_mode,
            TypeError,
            "use_cycle_or_paths",
            {"use_cycle_or_paths": bad_value},
        )


def test_validate_conflict_resolution_mode_rejects_invalid_strings() -> None:
    for bad_value in ["", "cycle", "path", "random"]:
        _expect_validator_raises_with_text(
            main_module.validate_conflict_resolution_mode,
            ValueError,
            "use_cycle_or_paths",
            {"use_cycle_or_paths": bad_value},
        )


def test_validate_conflict_resolution_mode_defaults_to_cycles() -> None:
    result = main_module.validate_conflict_resolution_mode({})
    assert result == "cycles"


def test_validate_gate_pz_assignment_defaults_to_empty_dict() -> None:
    result = main_module.validate_gate_pz_assignment({})
    assert result == {}


def test_validate_gate_pz_assignment_accepts_int_to_str_mapping() -> None:
    result = main_module.validate_gate_pz_assignment({"gate_pz_assignment": {0: "pz1", 3: "pz2"}})
    assert result == {0: "pz1", 3: "pz2"}


def test_validate_gate_pz_assignment_rejects_invalid_mapping_types() -> None:
    for bad_value in [False, True, 0, 1.2, "pz1", [1, 2], {1, 2}]:
        _expect_validator_raises_with_text(
            main_module.validate_gate_pz_assignment,
            TypeError,
            "gate_pz_assignment",
            {"gate_pz_assignment": bad_value},
        )


def test_validate_gate_pz_assignment_rejects_non_int_keys_and_non_str_values() -> None:
    _expect_validator_raises_with_text(
        main_module.validate_gate_pz_assignment,
        TypeError,
        "integer gate ids",
        {"gate_pz_assignment": {"0": "pz1"}},
    )
    _expect_validator_raises_with_text(
        main_module.validate_gate_pz_assignment,
        TypeError,
        "integer gate ids",
        {"gate_pz_assignment": {True: "pz1"}},
    )
    _expect_validator_raises_with_text(
        main_module.validate_gate_pz_assignment,
        TypeError,
        "string PZ names",
        {"gate_pz_assignment": {0: 1}},
    )
