# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validation helpers for JSON-compatible package data."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping


def require_mapping(data: object, label: str) -> dict[str, object]:
    """Return a JSON object.

    Raises:
        ValueError: If the value is not a JSON object.
    """
    if not isinstance(data, dict):
        msg = f"{label} must be a JSON object"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
    return cast("dict[str, object]", data)


def require_list(data: Mapping[str, object], key: str) -> list[object]:
    """Return a JSON array field.

    Raises:
        ValueError: If the field is not an array.
    """
    value = data.get(key)
    if not isinstance(value, list):
        msg = f"{key} must be a list"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
    return cast("list[object]", value)


def require_int(data: Mapping[str, object], key: str) -> int:
    """Return an integer field.

    Raises:
        ValueError: If the field is not an integer.
    """
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
    return value


def require_optional_int(data: Mapping[str, object], key: str) -> int | None:
    """Return an optional integer field.

    Raises:
        ValueError: If the field is neither an integer nor null.
    """
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer or null"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
    return value


def require_number(data: Mapping[str, object], key: str, *, default: float | None = None) -> float:
    """Return a numeric field.

    Raises:
        ValueError: If the field is not numeric.
    """
    value = data.get(key) if default is None else data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key} must be numeric"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
    return float(value)


def require_optional_number(data: Mapping[str, object], key: str) -> float | None:
    """Return an optional numeric field.

    Raises:
        ValueError: If the field is neither numeric nor null.
    """
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key} must be numeric or null"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
    return float(value)


def require_str(data: Mapping[str, object], key: str) -> str:
    """Return a string field.

    Raises:
        ValueError: If the field is not a string.
    """
    value = data.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
    return value


def require_bool(data: Mapping[str, object], key: str, *, default: bool) -> bool:
    """Return a Boolean field.

    Raises:
        ValueError: If the field is not Boolean.
    """
    value = data.get(key, default)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
    return value


def require_int_list(data: Mapping[str, object], key: str) -> list[int]:
    """Return an integer-array field.

    Raises:
        ValueError: If the field is not an array of integers.
    """
    values = require_list(data, key)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        msg = f"{key} must contain integers"
        raise ValueError(msg)
    return cast("list[int]", values)


def require_str_list(data: Mapping[str, object], key: str) -> list[str]:
    """Return a string-array field.

    Raises:
        ValueError: If the field is not an array of strings.
    """
    values = require_list(data, key)
    if any(not isinstance(value, str) for value in values):
        msg = f"{key} must contain strings"
        raise ValueError(msg)
    return cast("list[str]", values)


def require_int_pairs(data: Mapping[str, object], key: str) -> list[tuple[int, int]]:
    """Return an array of integer pairs.

    Raises:
        ValueError: If the field is not an array of integer pairs.
    """
    pairs: list[tuple[int, int]] = []
    for value in require_list(data, key):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or isinstance(value[0], bool)
            or not isinstance(value[0], int)
            or isinstance(value[1], bool)
            or not isinstance(value[1], int)
        ):
            msg = f"{key} must contain integer pairs"
            raise ValueError(msg)
        pairs.append((value[0], value[1]))
    return pairs


def require_str_int_pairs(data: Mapping[str, object], key: str) -> list[tuple[str, int]]:
    """Return an array of string/integer pairs.

    Raises:
        ValueError: If the field is not an array of string/integer pairs.
    """
    pairs: list[tuple[str, int]] = []
    for value in require_list(data, key):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or isinstance(value[1], bool)
            or not isinstance(value[1], int)
        ):
            msg = f"{key} must contain string/integer pairs"
            raise ValueError(msg)
        pairs.append((value[0], value[1]))
    return pairs


__all__ = [
    "require_bool",
    "require_int",
    "require_int_list",
    "require_int_pairs",
    "require_list",
    "require_mapping",
    "require_number",
    "require_optional_int",
    "require_optional_number",
    "require_str",
    "require_str_int_pairs",
    "require_str_list",
]
