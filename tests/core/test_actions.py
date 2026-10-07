# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared action values and their serialized types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, cast

import pytest

from mqt.ionshuttler.core import Action
from mqt.ionshuttler.core import Rx as SharedRx
from mqt.ionshuttler.core.actions import decode_action, index_action_types
from mqt.ionshuttler.core.gates import GATE_TYPES, Rzz
from mqt.ionshuttler.linear.actions import Action as LinearAction
from mqt.ionshuttler.linear.actions import Rx


@dataclass(frozen=True)
class _Marker(Action):
    """Custom action that carries one label."""

    label: str
    serialized_type: ClassVar[str] = "test.marker"


@dataclass(frozen=True)
class _Unnamed(Action):
    """Custom action without a serialized type."""


def test_actions_are_value_data_without_architecture_rules() -> None:
    """Leave validity, effects, and candidate generation to architectures and compilers."""
    action = _Marker(label="probe")

    assert {action, _Marker(label="probe")} == {action}
    for name in ("is_valid", "apply", "available_actions"):
        assert not hasattr(action, name)


def test_linear_actions_use_shared_action_base() -> None:
    """Linear action imports use the shared runtime classes."""
    assert LinearAction is Action
    assert isinstance(Rx(ion=0, theta=0.25), Action)
    assert Rx is SharedRx


def test_actions_serialize_with_their_stable_serialized_type() -> None:
    """Identify serialized actions by a namespaced type rather than a class name."""
    marker = _Marker(label="probe")

    assert marker.to_dict() == {"type": "test.marker", "label": "probe"}
    assert decode_action(marker.to_dict(), index_action_types((_Marker,))) == marker


def test_action_type_index_rejects_ambiguous_or_unnamed_types() -> None:
    """Require one action class per serialized type."""

    @dataclass(frozen=True)
    class OtherMarker(Action):
        serialized_type: ClassVar[str] = "test.marker"

    assert index_action_types((_Marker, _Marker)) == {"test.marker": _Marker}
    with pytest.raises(ValueError, match=r"duplicate serialized action type 'test\.marker'"):
        index_action_types((_Marker, OtherMarker))
    with pytest.raises(TypeError, match="does not declare a serialized_type"):
        index_action_types((_Unnamed,))
    with pytest.raises(TypeError, match="Action subclasses"):
        index_action_types((cast("type[Action]", object),))


def test_decoding_uses_only_the_supplied_action_types() -> None:
    """Decode no action type that the caller did not supply."""
    data = Rzz(ion_a=0, ion_b=1, theta=0.5).to_dict()

    assert decode_action(data, GATE_TYPES) == Rzz(ion_a=0, ion_b=1, theta=0.5)
    with pytest.raises(ValueError, match=r"unknown action type: gate\.rzz"):
        decode_action(data, index_action_types((_Marker,)))
    with pytest.raises(ValueError, match=r"action\.type must be a string"):
        decode_action({"label": "probe"}, index_action_types((_Marker,)))
    with pytest.raises(ValueError, match="invalid serialized _Marker action"):
        decode_action({"type": "test.marker"}, index_action_types((_Marker,)))
