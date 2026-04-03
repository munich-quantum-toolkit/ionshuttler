"""Tests for conflict-resolution mode config validation."""

from mqt.ionshuttler.multi_shuttler import main as main_module


def _expect_raises_with_text(
    expected_exception: type[Exception],
    expected_text: str,
    payload: dict[str, object],
) -> None:
    """Call validator and verify exception type/message."""
    try:
        main_module.validate_conflict_resolution_mode(payload)
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
        _expect_raises_with_text(TypeError, "use_cycle_or_paths", {"use_cycle_or_paths": bad_value})


def test_validate_conflict_resolution_mode_rejects_invalid_strings() -> None:
    for bad_value in ["", "cycle", "path", "random"]:
        _expect_raises_with_text(ValueError, "use_cycle_or_paths", {"use_cycle_or_paths": bad_value})


def test_validate_conflict_resolution_mode_defaults_to_cycles() -> None:
    result = main_module.validate_conflict_resolution_mode({})
    assert result == "cycles"
