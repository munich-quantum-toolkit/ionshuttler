# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared action values."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, cast

from .._json_utils import require_mapping

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True)
class Action:
    """Represent one operation in a compiled schedule.

    An action is immutable data. The architecture that executes the action
    decides whether it is supported, how long it takes, which resources it
    occupies, and how it changes the machine state.

    Each concrete action class declares a stable ``serialized_type``, such as
    ``"linear.shuttle"``, that identifies the class in serialized schedules.
    """

    serialized_type: ClassVar[str]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Action:
        """Restore an action from its serialized dataclass fields.

        Returns:
            The restored action.

        Raises:
            ValueError: If the saved fields cannot construct the action.
        """
        values = {model_field.name: data[model_field.name] for model_field in fields(cls) if model_field.name in data}
        constructor = cast("Any", cls)
        try:
            return cast("Action", constructor(**values))
        except TypeError as error:
            msg = f"invalid serialized {cls.__name__} action"
            raise ValueError(msg) from error

    def to_dict(self) -> dict[str, object]:
        """Return a description of this action using JSON-compatible values."""
        result: dict[str, object] = {"type": _serialized_type(type(self))}
        for model_field in fields(self):
            value = getattr(self, model_field.name)
            if value is not None:
                result[model_field.name] = value
        return result


def index_action_types(action_types: Iterable[type[Action]]) -> Mapping[str, type[Action]]:
    """Return action classes keyed by their serialized type.

    Returns:
        An immutable mapping from serialized type to action class.

    Raises:
        TypeError: If an entry is not an action class with a serialized type.
        ValueError: If two different classes use the same serialized type.
    """
    indexed: dict[str, type[Action]] = {}
    for action_type in action_types:
        if not isinstance(action_type, type) or not issubclass(action_type, Action):
            msg = "action_types must contain Action subclasses"
            raise TypeError(msg)
        name = _serialized_type(action_type)
        existing = indexed.get(name)
        if existing is not None and existing is not action_type:
            msg = f"duplicate serialized action type {name!r}"
            raise ValueError(msg)
        indexed[name] = action_type
    return MappingProxyType(indexed)


def decode_action(data: object, action_types: Mapping[str, type[Action]]) -> Action:
    """Restore one serialized action using an explicit set of action classes.

    Returns:
        The restored action.

    Raises:
        ValueError: If the data is malformed or names an unknown action type.
    """
    mapping = require_mapping(data, "action")
    name = mapping.get("type")
    if not isinstance(name, str):
        msg = "action.type must be a string"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Malformed JSON uses ValueError.
    action_type = action_types.get(name)
    if action_type is None:
        msg = f"unknown action type: {name}"
        raise ValueError(msg)
    return action_type.from_dict(mapping)


def _serialized_type(action_type: type[Action]) -> str:
    name = getattr(action_type, "serialized_type", None)
    if not isinstance(name, str) or not name:
        msg = f"action type {action_type.__name__} does not declare a serialized_type"
        raise TypeError(msg)
    return name


__all__ = ["Action", "decode_action", "index_action_types"]
