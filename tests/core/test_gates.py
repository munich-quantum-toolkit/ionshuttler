# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared gate occurrences and gate timing."""

from __future__ import annotations

import json
from math import pi
from typing import TYPE_CHECKING, cast

import pytest

from mqt.ionshuttler.core.actions import decode_action
from mqt.ionshuttler.core.gates import GATE_TYPES, GateAction, GateTiming, GlobalGate, Rx, Rxx, Ry, Ryy, Rz, Rzz
from mqt.ionshuttler.linear import GateTiming as LinearGateTiming

if TYPE_CHECKING:
    from collections.abc import Callable


def test_gate_timing_round_trips_and_linear_reexports_shared_identity() -> None:
    """Share one gate-duration value between architecture levels."""
    timing = GateTiming(rx=2, ry=2, rz=1, rxx=4, ryy=4, rzz=5, virtual_single_qubit_gates=frozenset())

    assert GateTiming.from_dict(timing.to_dict()) == timing
    assert GateTiming.from_dict({}) == GateTiming()
    assert LinearGateTiming is GateTiming


def test_gate_timing_rejects_malformed_serialized_durations() -> None:
    """Reject serialized durations that are not integers."""
    with pytest.raises(TypeError, match="rx duration must be an integer"):
        GateTiming.from_dict({"rx": 1.5})
    with pytest.raises(TypeError, match="gate_timing must be a JSON object"):
        GateTiming.from_dict([1, 2])


def test_gate_timing_rejects_unsupported_queries_and_malformed_virtual_gates() -> None:
    """Validate direct timing queries and virtual-gate collections."""
    timing = GateTiming()

    with pytest.raises(ValueError, match="unsupported gate name"):
        timing.duration_for("cx")
    with pytest.raises(ValueError, match="single-qubit gates"):
        timing.is_virtual("rzz")
    with pytest.raises(TypeError, match="duration for gate 'rx' must be an integer"):
        GateTiming(rx=cast("int", 1.5))
    with pytest.raises(TypeError, match="must be a collection"):
        GateTiming(virtual_single_qubit_gates=cast("frozenset[str]", "rx"))
    with pytest.raises(TypeError, match="contain only strings"):
        GateTiming(virtual_single_qubit_gates=cast("frozenset[str]", frozenset({1})))


@pytest.mark.parametrize("field_name", ["Rzz", "rzz_duration"])
def test_gate_timing_rejects_unknown_serialized_fields(field_name: str) -> None:
    """Refuse a misspelled duration instead of using the default duration."""
    with pytest.raises(ValueError, match=f"unknown gate_timing fields: {field_name}"):
        GateTiming.from_dict({field_name: 4})


def test_built_in_gate_types_have_stable_serialized_types() -> None:
    """Name each built-in gate type independently of its Python class name."""
    assert dict(GATE_TYPES) == {
        "gate.rx": Rx,
        "gate.ry": Ry,
        "gate.rz": Rz,
        "gate.rxx": Rxx,
        "gate.ryy": Ryy,
        "gate.rzz": Rzz,
        "gate.global": GlobalGate,
    }


def test_global_gate_stores_explicit_targets_in_ascending_order() -> None:
    """Treat the explicit targets as a set of ion identifiers."""
    gate = GlobalGate(gate_name="ry", theta=pi, ions=(3, 0, 2))

    assert gate.ions == (0, 2, 3)
    assert _gate_ions(gate) == (0, 2, 3)
    assert gate == GlobalGate(gate_name="ry", theta=pi, ions=(2, 3, 0))
    assert gate.to_dict() == {"type": "gate.global", "gate_name": "ry", "theta": pi, "ions": [0, 2, 3]}
    assert decode_action(gate.to_dict(), GATE_TYPES) == gate


def _gate_ions(gate: GateAction) -> tuple[int, ...]:
    """Read targets through the common gate contract."""
    return gate.ions


@pytest.mark.parametrize(
    ("ions", "error", "message"),
    [
        ((), ValueError, "must not be empty"),
        ((0, 0), ValueError, "unique"),
        ((-1,), ValueError, "non-negative"),
        ((True,), TypeError, "integers"),
    ],
)
def test_global_gate_rejects_invalid_targets(ions: tuple[int, ...], error: type[Exception], message: str) -> None:
    """Require unique, non-negative ion identifiers."""
    with pytest.raises(error, match=message):
        GlobalGate(gate_name="rx", theta=pi, ions=ions)


def test_global_gate_supports_only_single_ion_rotations() -> None:
    """Apply one single-ion rotation at a time through global control."""
    with pytest.raises(ValueError, match="global gates support only"):
        GlobalGate(gate_name="rzz", theta=pi, ions=(0,))
    with pytest.raises(ValueError, match="global gates support only"):
        GlobalGate(gate_name="Rx", theta=pi, ions=(0,))
    with pytest.raises(TypeError, match="gate_name must be a string"):
        GlobalGate(gate_name=cast("str", None), theta=pi, ions=(0,))


@pytest.mark.parametrize(
    ("build", "error", "message"),
    [
        pytest.param(lambda: Rx(ion=-1, theta=pi), ValueError, "ion must be non-negative", id="negative-ion"),
        pytest.param(lambda: Rx(ion=True, theta=pi), TypeError, "ion must be an integer", id="boolean-ion"),
        pytest.param(lambda: Rx(ion=cast("int", 0.0), theta=pi), TypeError, "ion must be an integer", id="float-ion"),
        pytest.param(
            lambda: Rzz(ion_a=0, ion_b=-2, theta=pi), ValueError, "ion_b must be non-negative", id="negative-ion-b"
        ),
        pytest.param(
            lambda: Rzz(ion_a=False, ion_b=1, theta=pi),
            TypeError,
            "ion_a must be an integer",
            id="boolean-ion-a",
        ),
        pytest.param(lambda: Rxx(ion_a=1, ion_b=1, theta=pi), ValueError, "must be distinct", id="identical-ions"),
        pytest.param(
            lambda: Ry(ion=0, theta=cast("float", "pi")), TypeError, "theta must be a real number", id="string-angle"
        ),
        pytest.param(lambda: Ry(ion=0, theta=True), TypeError, "theta must be a real number", id="boolean-angle"),
        pytest.param(lambda: Rz(ion=0, theta=float("inf")), ValueError, "theta must be finite", id="infinite-angle"),
        pytest.param(
            lambda: Ryy(ion_a=0, ion_b=1, theta=float("nan")), ValueError, "theta must be finite", id="nan-angle"
        ),
        pytest.param(
            lambda: GlobalGate(gate_name="rx", theta=float("-inf"), ions=(0,)),
            ValueError,
            "theta must be finite",
            id="infinite-global-angle",
        ),
        pytest.param(
            lambda: GlobalGate(gate_name="rx", theta=cast("float", "pi"), ions=(0,)),
            TypeError,
            "theta must be a real number",
            id="string-global-angle",
        ),
    ],
)
def test_gates_reject_invalid_ions_and_parameters(
    build: Callable[[], object],
    error: type[Exception],
    message: str,
) -> None:
    """Require non-negative integer ions, distinct operands, and finite angles."""
    with pytest.raises(error, match=message):
        build()


def test_gate_parameters_are_stored_as_floats_and_round_trip() -> None:
    """Store integer angles as floats so that serialized gates restore equally."""
    gate = Rzz(ion_a=2, ion_b=0, theta=1, gate_id=4)

    assert type(gate.theta) is float
    assert type(GlobalGate(gate_name="rz", theta=1, ions=(0,)).theta) is float
    assert decode_action(json.loads(json.dumps(gate.to_dict())), GATE_TYPES) == gate


def test_gate_equality_ignores_gate_id_and_serialization_keeps_it() -> None:
    """Compare gates by operation, and keep their circuit identity in JSON."""
    gate = Rx(ion=0, theta=pi, gate_id=3)
    restored = cast("Rx", decode_action(json.loads(json.dumps(gate.to_dict())), GATE_TYPES))

    assert gate == Rx(ion=0, theta=pi)
    assert hash(gate) == hash(Rx(ion=0, theta=pi))
    assert restored.gate_id == 3


@pytest.mark.parametrize("gate_id", [-1, True, 1.5])
def test_gate_ids_must_be_non_negative_in_memory_and_json(gate_id: object) -> None:
    """Apply the same gate-identity contract at construction and decoding boundaries."""
    with pytest.raises(ValueError, match="gate_id must be a non-negative integer"):
        Rx(ion=0, theta=pi, gate_id=cast("int", gate_id))
    with pytest.raises(ValueError, match="gate_id must be a non-negative integer"):
        decode_action({"type": "gate.rx", "ion": 0, "theta": pi, "gate_id": gate_id}, GATE_TYPES)


def test_gates_reject_non_finite_serialized_parameters() -> None:
    """Reject the non-standard JSON constants that Python accepts as numbers."""
    with pytest.raises(ValueError, match="theta must be finite"):
        decode_action(json.loads('{"type": "gate.rx", "ion": 0, "theta": Infinity}'), GATE_TYPES)
    with pytest.raises(ValueError, match="theta must be finite"):
        decode_action(json.loads('{"type": "gate.global", "gate_name": "rx", "theta": NaN, "ions": [0]}'), GATE_TYPES)
    with pytest.raises(ValueError, match="ion must be non-negative"):
        decode_action({"type": "gate.rx", "ion": -1, "theta": pi}, GATE_TYPES)


def test_global_gate_rejects_malformed_serialized_targets() -> None:
    """Report missing or malformed serialized target lists as data errors."""
    with pytest.raises(ValueError, match="ions must be a list"):
        decode_action({"type": "gate.global", "gate_name": "rx", "theta": pi}, GATE_TYPES)
    with pytest.raises(ValueError, match="ions must be a list"):
        decode_action({"type": "gate.global", "gate_name": "rx", "theta": pi, "ions": 3}, GATE_TYPES)
    with pytest.raises(ValueError, match="invalid serialized GlobalGate"):
        decode_action({"type": "gate.global", "gate_name": "rx", "theta": pi, "ions": ["a"]}, GATE_TYPES)
