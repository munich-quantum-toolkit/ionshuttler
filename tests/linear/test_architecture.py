# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Linear architecture and field profile."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import] - The tests unpickle only data they create.
from typing import TYPE_CHECKING, cast

import pytest

from mqt.ionshuttler.linear import GateTiming, TransportTiming
from mqt.ionshuttler.linear.actions import (
    DEFAULT_ACTION_TYPES,
    GlobalGate,
    Rx,
    Rz,
    Rzz,
    Shuttle,
)
from mqt.ionshuttler.linear.architecture import IMPLICIT_PROCESSING_ZONE, LinearArchitecture
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mqt.ionshuttler.linear.actions import Action


def test_processing_zones_are_validated_and_normalized() -> None:
    """Normalize zone sites and derive every within-zone interaction pair."""
    architecture = LinearArchitecture(
        num_sites=10,
        processing_zones={"pz_1": [4, 2, 3], "pz_2": [8, 9]},
    )

    assert architecture.processing_zones == {"pz_1": (2, 3, 4), "pz_2": (8, 9)}
    assert architecture.valid_two_qubit_site_pairs == ((2, 3), (2, 4), (3, 4), (8, 9))
    assert architecture.get_processing_zone(3) == "pz_1"
    assert architecture.get_processing_zone(7) is None


@pytest.mark.parametrize(
    ("processing_zones", "message"),
    [
        ({"pz_1": []}, "must not be empty"),
        ({"pz_1": [2, 2]}, "duplicate"),
        ({"pz_1": [2, 4]}, "contiguous"),
        ({"pz_1": [0, 1], "pz_2": [1, 2]}, "overlaps"),
        ({"pz_1": [5]}, "invalid site"),
    ],
)
def test_invalid_processing_zones_raise_value_error(
    processing_zones: dict[str, list[int]],
    message: str,
) -> None:
    """Reject malformed processing-zone definitions."""
    with pytest.raises(ValueError, match=message):
        LinearArchitecture(num_sites=5, processing_zones=processing_zones)


def test_processing_zones_cannot_change_after_construction() -> None:
    """Keep zone lookups and derived site pairs consistent with the stored zones."""
    supplied_zones = {"pz": [0, 1]}
    architecture = LinearArchitecture(num_sites=4, processing_zones=supplied_zones)
    supplied_zones["pz"].append(2)
    supplied_zones["other"] = [3]
    stored_zones = cast("dict[str, tuple[int, ...]]", architecture.processing_zones)

    with pytest.raises(TypeError):
        stored_zones["other"] = (3,)
    assert architecture.processing_zones == {"pz": (0, 1)}
    assert architecture.valid_two_qubit_site_pairs == ((0, 1),)
    assert architecture.get_processing_zone(3) is None


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"num_sites": True}, TypeError, "num_sites must be an integer"),
        ({"num_sites": 2.0}, TypeError, "num_sites must be an integer"),
        ({"num_sites": 2, "processing_zones": [[0, 1]]}, TypeError, "processing_zones must be a mapping"),
        ({"num_sites": 2, "processing_zones": {0: [0, 1]}}, TypeError, "zone names must be strings"),
        ({"num_sites": 2, "processing_zones": {"": [0, 1]}}, ValueError, "zone names must not be empty"),
        ({"num_sites": 2, "processing_zones": {"pz": [True]}}, TypeError, "sequence of integer sites"),
        ({"num_sites": 2, "processing_zones": {"pz": 1}}, TypeError, "sequence of integer sites"),
        ({"num_sites": 2, "field_profile": {"0": 0.5}}, TypeError, "field_profile must be FieldProfile"),
    ],
)
def test_constructor_rejects_values_that_cannot_round_trip(
    arguments: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Reject the values that serialized architecture data cannot represent."""
    constructor = cast("Callable[..., LinearArchitecture]", LinearArchitecture)

    with pytest.raises(error, match=message):
        constructor(**arguments)


def test_from_dict_rejects_boolean_zone_sites() -> None:
    """Treat Boolean zone sites as malformed serialized data."""
    with pytest.raises(ValueError, match="processing zone sites must be a list of integers"):
        LinearArchitecture.from_dict({"num_sites": 2, "processing_zones": {"pz": [False, True]}})


def test_missing_processing_zones_create_implicit_full_array_zone() -> None:
    """Use one all-sites processing zone when no zones are supplied."""
    architecture = LinearArchitecture(num_sites=4)

    assert architecture.processing_zones == {IMPLICIT_PROCESSING_ZONE: (0, 1, 2, 3)}
    assert architecture.valid_two_qubit_site_pairs == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )
    assert architecture.sites_share_processing_zone(0, 3)
    assert not architecture.sites_share_processing_zone()


def test_field_profile_defaults_and_architecture_integration() -> None:
    """Fill unspecified field values without changing the supplied scale."""
    profile = FieldProfile(num_sites=5, site_field=((1, 0.25), (3, -0.5)))
    architecture = LinearArchitecture(num_sites=5, processing_zones={"A": [0, 1]}, field_profile=profile)

    assert profile.field_at(1) == pytest.approx(0.25)
    assert profile.field_at(2) == pytest.approx(1.0)
    assert architecture.field_at(3) == pytest.approx(-0.5)
    assert architecture.has_nontrivial_field_profile()
    zero_profile_architecture = LinearArchitecture(
        num_sites=4,
        field_profile=FieldProfile(num_sites=4, site_field=(), default_field=0.0),
    )
    assert zero_profile_architecture.has_nontrivial_field_profile()
    assert "field_profile" in zero_profile_architecture.to_dict()
    assert not LinearArchitecture(num_sites=4).has_nontrivial_field_profile()


def test_architecture_round_trips_through_json() -> None:
    """Preserve architecture and structured field metadata through JSON."""
    architecture = LinearArchitecture(
        num_sites=5,
        processing_zones={"A": [0, 1], "B": [3, 4]},
        field_profile=FieldProfile(
            num_sites=5,
            site_field=((1, 0.25), (3, -0.5)),
            default_field=0.9,
        ),
        gate_timing=GateTiming(rx=2, rzz=3),
        transport_timing=TransportTiming(shuttle=2, swap=4),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )

    assert LinearArchitecture.from_json(architecture.to_json()) == architecture
    assert LinearArchitecture.from_json(LinearArchitecture(num_sites=3).to_json()) == LinearArchitecture(num_sites=3)
    assert architecture.supports(GlobalGate)
    assert architecture.to_dict()["supported_action_types"] == [
        "linear.physical_swap",
        "linear.shuttle",
        "gate.rx",
        "gate.ry",
        "gate.rz",
        "gate.rzz",
        "gate.global",
    ]
    assert architecture.to_dict()["transport_timing"] == {"shuttle": 2, "swap": 4}


def test_architecture_round_trips_through_pickle() -> None:
    """Restore an equal architecture whose processing zones stay read-only."""
    architecture = LinearArchitecture(
        num_sites=5,
        processing_zones={"A": [0, 1], "B": [3, 4]},
        field_profile=FieldProfile(num_sites=5, site_field=((1, 0.25),)),
        gate_timing=GateTiming(rx=2),
        transport_timing=TransportTiming(shuttle=2),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )

    restored = pickle.loads(pickle.dumps(architecture))  # ruff: ignore[suspicious-pickle-usage] - Test-created data.
    restored_zones = cast("dict[str, tuple[int, ...]]", restored.processing_zones)

    assert restored == architecture
    with pytest.raises(TypeError):
        restored_zones["C"] = (2,)


def test_architecture_rejects_unknown_serialized_fields() -> None:
    """Refuse serialized fields that the architecture would otherwise ignore."""
    with pytest.raises(ValueError, match="unknown architecture fields: timing"):
        LinearArchitecture.from_dict({"num_sites": 2, "timing": {}})


def test_architecture_load_reads_utf8_json(tmp_path: Path) -> None:
    """Load architecture metadata from an explicit UTF-8 file boundary."""
    config_path = tmp_path / "architecture.json"
    config_path.write_text(
        '{"num_sites": 4, "processing_zones": {"pz": [0, 1, 2, 3]}, '
        '"field_profile": {"site_field": {"1": 0.5}, "default_field": 1.0}}',
        encoding="utf-8",
    )

    assert LinearArchitecture.load(config_path) == LinearArchitecture(
        num_sites=4,
        processing_zones={"pz": [0, 1, 2, 3]},
        field_profile=FieldProfile(num_sites=4, site_field=((1, 0.5),)),
    )


def test_bare_field_mapping_infers_its_site_count(tmp_path: Path) -> None:
    """Size a standalone field profile from its largest site index."""
    profile_path = tmp_path / "field_profile.json"
    profile_path.write_text('{"0": 0.25, "3": -0.5}', encoding="utf-8")

    profile = FieldProfile.load(profile_path)

    assert profile.num_sites == 4
    assert profile.field_at(0) == pytest.approx(0.25)
    assert profile.field_at(1) == pytest.approx(1.0)
    assert profile.field_at(3) == pytest.approx(-0.5)


def test_empty_field_mapping_requires_an_explicit_site_count() -> None:
    """Reject a standalone empty mapping whose size cannot be inferred."""
    with pytest.raises(ValueError, match="num_sites is required"):
        FieldProfile.from_dict({})


def test_architecture_and_field_profile_reject_invalid_shapes() -> None:
    """Reject malformed hardware layouts and field profiles early."""
    with pytest.raises(ValueError, match="num_sites"):
        LinearArchitecture(num_sites=0)
    with pytest.raises(ValueError, match="invalid site"):
        FieldProfile(num_sites=4, site_field=((4, 1.0),))
    with pytest.raises(ValueError, match=r"field_profile\.num_sites"):
        LinearArchitecture(num_sites=5, field_profile=FieldProfile(num_sites=4, site_field=()))
    with pytest.raises(TypeError, match="JSON object"):
        LinearArchitecture.from_dict([])
    with pytest.raises(TypeError, match=r"architecture\.num_sites"):
        LinearArchitecture.from_dict({"num_sites": "4"})
    with pytest.raises(TypeError, match=r"architecture\.num_sites"):
        LinearArchitecture.from_dict({"num_sites": True})
    with pytest.raises(TypeError, match="Action subclasses"):
        LinearArchitecture(num_sites=1, supported_action_types=(1,))  # ty: ignore[invalid-argument-type] - Intentionally invalid entry to test runtime validation.
    with pytest.raises(TypeError, match="list of strings"):
        LinearArchitecture.from_dict({"num_sites": 1, "supported_action_types": "gate.global"})
    with pytest.raises(ValueError, match=r"unknown architecture action type: 'gate\.cx'"):
        LinearArchitecture.from_dict({"num_sites": 1, "supported_action_types": ["gate.cx"]})
    with pytest.raises(TypeError, match="Action subclasses"):
        LinearArchitecture(
            num_sites=1,
            supported_action_types=(*DEFAULT_ACTION_TYPES, cast("type[Action]", AdvanceTime)),
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        LinearArchitecture(num_sites=1, supported_action_types=(Rx, Rx))
    with pytest.raises(TypeError, match="gate_timing must be GateTiming"):
        LinearArchitecture(num_sites=1, gate_timing=cast("GateTiming", TransportTiming()))


def test_action_processing_zone_follows_ion_positions() -> None:
    """Report the zone an action occupies, and none for zone-free actions."""
    architecture = LinearArchitecture(num_sites=6, processing_zones={"pz_1": [0, 1], "pz_2": [4, 5]})
    state = create_initial_state(3, architecture, initial_positions=[0, 1, 3])

    assert architecture.action_processing_zone(Rx(ion=0, theta=0.5), state) == "pz_1"
    assert architecture.action_processing_zone(Rx(ion=2, theta=0.5), state) is None
    assert architecture.action_processing_zone(Rzz(ion_a=1, ion_b=0, theta=0.5), state) == "pz_1"
    assert architecture.action_processing_zone(Rz(ion=0, theta=0.5), state) is None
    assert architecture.action_processing_zone(GlobalGate(gate_name="rx", theta=0.5, ions=(0, 1, 2)), state) is None
    assert architecture.action_processing_zone(Shuttle(ion=2, src=3, dst=4), state) is None
