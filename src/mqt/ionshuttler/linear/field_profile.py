# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Site-dependent field values associated with a Linear architecture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class FieldProfile:
    """Store the scalar field assigned to each hardware site.

    Unspecified sites receive ``default_field``. Values retain their supplied
    scale; this class does not perform RMS normalization. The compiler stores
    the profile for downstream analysis but does not use it during search.
    """

    num_sites: int
    site_field: tuple[tuple[int, float], ...]
    default_field: float = 1.0

    def __post_init__(self) -> None:
        """Fill unspecified sites and store the profile in site order.

        Raises:
            ValueError: If the number of sites or a site index is invalid.
        """
        if self.num_sites < 1:
            msg = "num_sites must be >= 1"
            raise ValueError(msg)

        normalized = {site: float(self.default_field) for site in range(self.num_sites)}
        for site, field_value in self.site_field:
            if not 0 <= site < self.num_sites:
                msg = f"field profile contains invalid site {site}; expected sites within [0, {self.num_sites - 1}]"
                raise ValueError(msg)
            normalized[site] = float(field_value)

        object.__setattr__(self, "site_field", tuple(sorted(normalized.items())))

    def field_at(self, site: int) -> float:
        """Return the field at one site.

        Args:
            site: Zero-based architecture site.

        Returns:
            The configured field value.

        Raises:
            ValueError: If ``site`` lies outside the architecture.
        """
        if not 0 <= site < self.num_sites:
            msg = f"site must be within [0, {self.num_sites - 1}]"
            raise ValueError(msg)
        return dict(self.site_field).get(site, self.default_field)

    @classmethod
    def from_dict(cls, data: object, num_sites: int | None = None) -> FieldProfile:
        """Construct a profile from structured or site-to-value mapping data.

        Args:
            data: JSON-style field-profile mapping.
            num_sites: Architecture size for structured mappings.

        Returns:
            A complete field profile covering every site.

        Raises:
            ValueError: If the mapping has an invalid shape or lacks a required size.
        """
        if not isinstance(data, dict):
            msg = "field_profile must be a JSON object"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - Input validation uses ValueError.
        mapping = cast("dict[str, object]", data)

        if "site_field" in mapping or "default_field" in mapping:
            if num_sites is None:
                msg = "num_sites is required when loading structured field_profile data"
                raise ValueError(msg)
            site_field_raw = mapping.get("site_field", {})
            if not isinstance(site_field_raw, dict):
                msg = "field_profile.site_field must be a JSON object"
                raise ValueError(msg)
            site_field = cast("dict[str, object]", site_field_raw)
            default_field = mapping.get("default_field", 1.0)
            if not isinstance(default_field, int | float):
                msg = "field_profile.default_field must be numeric"
                raise ValueError(msg)
            return cls(
                num_sites=num_sites,
                site_field=tuple(
                    (int(site), float(cast("str | int | float", field_value)))
                    for site, field_value in site_field.items()
                ),
                default_field=float(default_field),
            )

        if num_sites is None:
            if not mapping:
                msg = "num_sites is required when loading an empty field_profile mapping"
                raise ValueError(msg)
            inferred_num_sites = max(int(site) for site in mapping) + 1
        else:
            inferred_num_sites = num_sites
        return cls(
            num_sites=inferred_num_sites,
            site_field=tuple(
                (int(site), float(cast("str | int | float", field_value))) for site, field_value in mapping.items()
            ),
        )

    @classmethod
    def from_json(cls, raw: str, num_sites: int | None = None) -> FieldProfile:
        """Deserialize a field profile from JSON.

        Args:
            raw: Serialized JSON object.
            num_sites: Architecture size for structured mappings.

        Returns:
            A complete field profile covering every site.

        """
        return cls.from_dict(json.loads(raw), num_sites=num_sites)

    @classmethod
    def load(cls, filename: str | Path, num_sites: int | None = None) -> FieldProfile:
        """Load a field profile from a UTF-8 JSON file.

        Args:
            filename: File to read.
            num_sites: Architecture size for structured mappings.

        Returns:
            A complete field profile covering every site.
        """
        return cls.from_json(Path(filename).read_text(encoding="utf-8"), num_sites=num_sites)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible profile metadata."""
        return {
            "site_field": {str(site): field_value for site, field_value in self.site_field},
            "default_field": self.default_field,
        }

    def to_json(self) -> str:
        """Serialize the profile as JSON.

        Returns:
            The serialized profile object.
        """
        return json.dumps(self.to_dict())


__all__ = ["FieldProfile"]
