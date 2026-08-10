# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hardware layout and processing capabilities for a linear ion chain."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations, pairwise
from pathlib import Path
from typing import TYPE_CHECKING, cast

from mqt.ionshuttler.linear.field_profile import FieldProfile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

IMPLICIT_PROCESSING_ZONE = "all_sites"


@dataclass(frozen=True)
class Architecture:
    """Describe the available sites, processing zones, and site-dependent field."""

    num_sites: int
    processing_zones: Mapping[str, Sequence[int]] | None = None
    field_profile: FieldProfile | None = None
    valid_two_qubit_site_pairs: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        """Check the hardware description and store zone sites in order.

        Raises:
            ValueError: If the site count, zones, or field profile are invalid.
        """
        if self.num_sites < 1:
            msg = "num_sites must be >= 1"
            raise ValueError(msg)

        processing_zones = _normalize_processing_zones(self.num_sites, self.processing_zones)
        object.__setattr__(self, "processing_zones", processing_zones)
        field_profile = self.field_profile
        if field_profile is None:
            field_profile = FieldProfile(num_sites=self.num_sites, site_field=())
        if field_profile.num_sites != self.num_sites:
            msg = "field_profile.num_sites must match architecture.num_sites"
            raise ValueError(msg)
        object.__setattr__(self, "field_profile", field_profile)
        object.__setattr__(
            self,
            "valid_two_qubit_site_pairs",
            _valid_two_qubit_site_pairs(processing_zones),
        )

    def get_processing_zone(self, site: int) -> str | None:
        """Return the processing zone containing a site, if any."""
        for zone_name, zone_sites in self._processing_zones().items():
            if site in zone_sites:
                return zone_name
        return None

    def field_at(self, site: int) -> float:
        """Return the configured field value at one site."""
        _validate_site(site, self.num_sites)
        return self._field_profile().field_at(site)

    def has_nontrivial_field_profile(self) -> bool:
        """Return whether any site has a nonzero configured field."""
        return any(value for _, value in self._field_profile().site_field)

    def sites_share_processing_zone(self, *sites: int) -> bool:
        """Return whether all supplied sites belong to one processing zone."""
        if not sites:
            return False
        first_zone = self.get_processing_zone(sites[0])
        if first_zone is None:
            return False
        return all(self.get_processing_zone(site) == first_zone for site in sites[1:])

    def initial_pzs_busy_until(self) -> tuple[tuple[str, int], ...]:
        """Return every processing zone as available at the start of a schedule."""
        return tuple((zone_name, 0) for zone_name in self._processing_zones())

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible architecture metadata."""
        result: dict[str, object] = {
            "num_sites": self.num_sites,
            "processing_zones": {
                zone_name: list(zone_sites) for zone_name, zone_sites in self._processing_zones().items()
            },
        }
        field_profile = self._field_profile()
        if field_profile.default_field - 1.0 or any(value - 1.0 for _, value in field_profile.site_field):
            result["field_profile"] = field_profile.to_dict()
        return result

    def to_json(self) -> str:
        """Serialize architecture metadata as JSON.

        Returns:
            The serialized architecture object.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> Architecture:
        """Construct an architecture from a JSON-style mapping.

        Args:
            data: Architecture mapping.

        Returns:
            A validated architecture.

        Raises:
            ValueError: If the mapping has an invalid shape or values.
        """
        if not isinstance(data, dict):
            msg = "architecture must be a JSON object"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Input validation uses ValueError.
        mapping = cast("dict[str, object]", data)

        num_sites = mapping.get("num_sites")
        if not isinstance(num_sites, int):
            msg = "architecture.num_sites must be an integer"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Input validation uses ValueError.

        processing_zones_raw = mapping.get("processing_zones")
        processing_zones = None
        if processing_zones_raw is not None:
            if not isinstance(processing_zones_raw, dict):
                msg = "architecture.processing_zones must be a JSON object"
                raise ValueError(msg)
            zone_mapping = cast("dict[str, object]", processing_zones_raw)
            processing_zones = {
                zone_name: _require_int_sequence(zone_sites, "processing zone sites")
                for zone_name, zone_sites in zone_mapping.items()
            }

        field_profile_raw = mapping.get("field_profile")
        field_profile = None
        if field_profile_raw is not None:
            if not isinstance(field_profile_raw, dict):
                msg = "architecture.field_profile must be a JSON object"
                raise ValueError(msg)
            field_profile = FieldProfile.from_dict(field_profile_raw, num_sites=num_sites)

        return cls(num_sites=num_sites, processing_zones=processing_zones, field_profile=field_profile)

    @classmethod
    def from_json(cls, raw: str) -> Architecture:
        """Deserialize an architecture from JSON.

        Returns:
            A validated architecture.
        """
        return cls.from_dict(json.loads(raw))

    @classmethod
    def load(cls, filename: str | Path) -> Architecture:
        """Load an architecture from a UTF-8 JSON file.

        Returns:
            A validated architecture.
        """
        return cls.from_json(Path(filename).read_text(encoding="utf-8"))

    def _processing_zones(self) -> dict[str, tuple[int, ...]]:
        return cast("dict[str, tuple[int, ...]]", self.processing_zones)

    def _field_profile(self) -> FieldProfile:
        return cast("FieldProfile", self.field_profile)


def _normalize_processing_zones(
    num_sites: int,
    processing_zones: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[int, ...]]:
    if not processing_zones:
        return {IMPLICIT_PROCESSING_ZONE: tuple(range(num_sites))}

    normalized: dict[str, tuple[int, ...]] = {}
    seen_sites: set[int] = set()
    for zone_name, zone_sites in processing_zones.items():
        if not zone_sites:
            msg = f"processing zone '{zone_name}' must not be empty"
            raise ValueError(msg)
        sorted_sites = tuple(sorted(zone_sites))
        if len(set(sorted_sites)) != len(sorted_sites):
            msg = f"processing zone '{zone_name}' contains duplicate sites"
            raise ValueError(msg)
        for site in sorted_sites:
            if not 0 <= site < num_sites:
                msg = (
                    f"processing zone '{zone_name}' contains invalid site {site}; "
                    f"expected sites within [0, {num_sites - 1}]"
                )
                raise ValueError(msg)
            if site in seen_sites:
                msg = f"processing zone '{zone_name}' overlaps with another zone at site {site}"
                raise ValueError(msg)
            seen_sites.add(site)
        if any(right - left != 1 for left, right in pairwise(sorted_sites)):
            msg = f"processing zone '{zone_name}' must contain contiguous sites"
            raise ValueError(msg)
        normalized[zone_name] = sorted_sites
    return normalized


def _valid_two_qubit_site_pairs(
    processing_zones: dict[str, tuple[int, ...]],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right) for zone_sites in processing_zones.values() for left, right in combinations(zone_sites, 2)
    )


def _validate_site(site: int, num_sites: int) -> None:
    if not 0 <= site < num_sites:
        msg = f"site must be within [0, {num_sites - 1}]"
        raise ValueError(msg)


def _require_int_sequence(value: object, label: str) -> list[int]:
    if not isinstance(value, list | tuple) or not all(isinstance(item, int) for item in value):
        msg = f"{label} must be a list of integers"
        raise ValueError(msg)
    return list(cast("list[int] | tuple[int, ...]", value))


__all__ = ["IMPLICIT_PROCESSING_ZONE", "Architecture"]
