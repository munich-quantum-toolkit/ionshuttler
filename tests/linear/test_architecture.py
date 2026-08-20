# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Linear architecture and field profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES, AdvanceTime, GlobalPulse
from mqt.ionshuttler.linear.architecture import IMPLICIT_PROCESSING_ZONE, Architecture
from mqt.ionshuttler.linear.field_profile import FieldProfile

if TYPE_CHECKING:
    from pathlib import Path


def test_processing_zones_are_validated_and_normalized() -> None:
    """Normalize zone sites and derive every within-zone interaction pair."""
    architecture = Architecture(
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
        Architecture(num_sites=5, processing_zones=processing_zones)


def test_missing_processing_zones_create_implicit_full_array_zone() -> None:
    """Use one all-sites processing zone when no zones are supplied."""
    architecture = Architecture(num_sites=4)

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
    architecture = Architecture(num_sites=5, processing_zones={"A": [0, 1]}, field_profile=profile)

    assert profile.field_at(1) == pytest.approx(0.25)
    assert profile.field_at(2) == pytest.approx(1.0)
    assert architecture.field_at(3) == pytest.approx(-0.5)
    assert architecture.has_nontrivial_field_profile()
    zero_profile_architecture = Architecture(
        num_sites=4,
        field_profile=FieldProfile(num_sites=4, site_field=(), default_field=0.0),
    )
    assert zero_profile_architecture.has_nontrivial_field_profile()
    assert "field_profile" in zero_profile_architecture.to_dict()
    assert not Architecture(num_sites=4).has_nontrivial_field_profile()


def test_architecture_round_trips_through_json() -> None:
    """Preserve architecture and structured field metadata through JSON."""
    architecture = Architecture(
        num_sites=5,
        processing_zones={"A": [0, 1], "B": [3, 4]},
        field_profile=FieldProfile(
            num_sites=5,
            site_field=((1, 0.25), (3, -0.5)),
            default_field=0.9,
        ),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalPulse),
    )

    assert Architecture.from_json(architecture.to_json()) == architecture
    assert architecture.supports(GlobalPulse)
    assert architecture.to_dict()["supported_action_types"] == [
        *(action_type.__name__ for action_type in DEFAULT_ACTION_TYPES),
        "GlobalPulse",
    ]


def test_architecture_load_reads_utf8_json(tmp_path: Path) -> None:
    """Load architecture metadata from an explicit UTF-8 file boundary."""
    config_path = tmp_path / "architecture.json"
    config_path.write_text(
        '{"num_sites": 4, "processing_zones": {"pz": [0, 1, 2, 3]}, '
        '"field_profile": {"site_field": {"1": 0.5}, "default_field": 1.0}}',
        encoding="utf-8",
    )

    assert Architecture.load(config_path) == Architecture(
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
        Architecture(num_sites=0)
    with pytest.raises(ValueError, match="invalid site"):
        FieldProfile(num_sites=4, site_field=((4, 1.0),))
    with pytest.raises(ValueError, match=r"field_profile\.num_sites"):
        Architecture(num_sites=5, field_profile=FieldProfile(num_sites=4, site_field=()))
    with pytest.raises(TypeError, match="JSON object"):
        Architecture.from_dict([])
    with pytest.raises(TypeError, match=r"architecture\.num_sites"):
        Architecture.from_dict({"num_sites": "4"})
    with pytest.raises(TypeError, match=r"architecture\.num_sites"):
        Architecture.from_dict({"num_sites": True})
    with pytest.raises(TypeError, match="Action subclasses"):
        Architecture(num_sites=1, supported_action_types=(1,))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="list of strings"):
        Architecture.from_dict({"num_sites": 1, "supported_action_types": "GlobalPulse"})
    with pytest.raises(ValueError, match="scheduler operation"):
        Architecture(num_sites=1, supported_action_types=(*DEFAULT_ACTION_TYPES, AdvanceTime))
