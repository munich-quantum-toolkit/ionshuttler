# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hardware layout, timing, and operation rules for a linear qccd architecture."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations, pairwise
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from mqt.ionshuttler.core.gates import (
    GATE_NAMES,
    SINGLE_QUBIT_GATE_NAMES,
    GateAction,
    GateTiming,
    GlobalGate,
    SingleQubitGate,
    TwoQubitGate,
)
from mqt.ionshuttler.linear.actions import (
    DEFAULT_ACTION_TYPES,
    LINEAR_ACTION_TYPES,
    Action,
    PhysicalSwap,
    Shuttle,
    TransportTiming,
    is_adjacent,
)
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.replay import replay_schedule as replay_linear_schedule
from mqt.ionshuttler.linear.schedule import LinearMachineState

if TYPE_CHECKING:
    from mqt.ionshuttler.core.schedule import Schedule
    from mqt.ionshuttler.linear.state import State

IMPLICIT_PROCESSING_ZONE = "all_sites"
_SERIALIZED_FIELDS = frozenset({
    "num_sites",
    "processing_zones",
    "field_profile",
    "gate_timing",
    "transport_timing",
    "supported_action_types",
})


@dataclass(frozen=True)
class LinearArchitecture:
    """Describe the sites, processing zones, timing, and supported operations.

    The architecture owns the complete meaning of its operations: their
    durations, the ions and processing zones they occupy, when they may start,
    and how they change the machine state.

    ``supported_action_types`` selects which of the operations implemented by
    Linear architectures this device offers, for example to compare schedules
    with and without physical swaps. It does not add new operation types.

    The architecture stores ``processing_zones`` as a read-only mapping from
    zone name to sorted site tuple, so later changes to the supplied mapping
    do not affect it.
    """

    num_sites: int
    processing_zones: Mapping[str, Sequence[int]] | None = None
    field_profile: FieldProfile | None = None
    gate_timing: GateTiming = field(default_factory=GateTiming)
    transport_timing: TransportTiming = field(default_factory=TransportTiming)
    supported_action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES
    valid_two_qubit_site_pairs: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        """Check the hardware description and store zone sites in order.

        Raises:
            TypeError: If the site count, a zone, the field profile, timing, or
                an action type has the wrong type.
            ValueError: If the site count, zones, operations, or field profile are invalid.
        """
        if isinstance(self.num_sites, bool) or not isinstance(self.num_sites, int):
            msg = "num_sites must be an integer"
            raise TypeError(msg)
        if self.num_sites < 1:
            msg = "num_sites must be >= 1"
            raise ValueError(msg)
        if self.field_profile is not None and not isinstance(self.field_profile, FieldProfile):
            msg = "field_profile must be FieldProfile or None"
            raise TypeError(msg)
        if not isinstance(self.gate_timing, GateTiming):
            msg = "gate_timing must be GateTiming"
            raise TypeError(msg)
        if not isinstance(self.transport_timing, TransportTiming):
            msg = "transport_timing must be TransportTiming"
            raise TypeError(msg)
        supported_action_types = tuple(self.supported_action_types)
        if any(
            not isinstance(action_type, type) or not issubclass(action_type, Action)
            for action_type in supported_action_types
        ):
            msg = "supported_action_types must contain Action subclasses"
            raise TypeError(msg)
        if len(set(supported_action_types)) != len(supported_action_types):
            msg = "supported_action_types must not contain duplicates"
            raise ValueError(msg)
        implemented = set(LINEAR_ACTION_TYPES.values())
        unknown = [action_type.__name__ for action_type in supported_action_types if action_type not in implemented]
        if unknown:
            msg = f"LinearArchitecture implements no action types named: {', '.join(unknown)}"
            raise ValueError(msg)
        object.__setattr__(self, "supported_action_types", supported_action_types)

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

    def __reduce__(self) -> tuple[type[LinearArchitecture], tuple[object, ...]]:
        """Rebuild the architecture from its constructor arguments when pickled.

        :mod:`pickle` cannot store the read-only ``processing_zones`` mapping,
        so the zones pass through a regular dictionary.

        Returns:
            The architecture class and its constructor arguments.
        """
        return (
            type(self),
            (
                self.num_sites,
                dict(self._processing_zones()),
                self.field_profile,
                self.gate_timing,
                self.transport_timing,
                self.supported_action_types,
            ),
        )

    def get_processing_zone(self, site: int) -> str | None:
        """Return the processing zone containing a site, if any."""
        for zone_name, zone_sites in self._processing_zones().items():
            if site in zone_sites:
                return zone_name
        return None

    def action_processing_zone(self, action: Action, state: State) -> str | None:
        """Return the processing zone an action occupies in a machine state.

        Args:
            action: Action whose zone is requested.
            state: Machine state that holds the ion positions.

        Returns:
            The zone name, or ``None`` if the action occupies no zone. Global
            gates, virtual single-ion gates, and transport occupy none.
        """
        if isinstance(action, GlobalGate):
            return None
        if isinstance(action, SingleQubitGate):
            if self.is_virtual_gate(action):
                return None
            return self.get_processing_zone(dict(state.positions)[action.ion])
        if isinstance(action, TwoQubitGate):
            return self.get_processing_zone(dict(state.positions)[action.ion_a])
        return None

    def field_at(self, site: int) -> float:
        """Return the configured field value at one site."""
        _validate_site(site, self.num_sites)
        return self._field_profile().field_at(site)

    def has_nontrivial_field_profile(self) -> bool:
        """Return whether any site differs from the unit field profile."""
        return _has_nontrivial_field_profile(self._field_profile())

    def supports(self, action_type: type[Action]) -> bool:
        """Return whether the hardware exposes an action type."""
        return action_type in self.supported_action_types

    def action_duration(self, action: Action) -> int:
        """Return the architecture's duration for one action.

        Gates use :attr:`gate_timing`, and a global gate takes the duration of
        its rotation. Transport uses :attr:`transport_timing`.

        Returns:
            The duration in timesteps.

        Raises:
            TypeError: If the architecture defines no duration for the action.
        """
        if isinstance(action, Shuttle):
            return self.transport_timing.shuttle
        if isinstance(action, PhysicalSwap):
            return self.transport_timing.swap
        if isinstance(action, GlobalGate):
            return self.gate_timing.duration_for(action.gate_name)
        gate_name = action.circuit_name if isinstance(action, GateAction) else None
        if gate_name is not None and gate_name in GATE_NAMES:
            return self.gate_timing.duration_for(gate_name)
        msg = f"LinearArchitecture defines no duration for {type(action).__name__}"
        raise TypeError(msg)

    def is_virtual_gate(self, gate: SingleQubitGate) -> bool:
        """Return whether this architecture implements a single-ion gate virtually."""
        gate_name = gate.circuit_name
        if gate_name is None or gate_name not in SINGLE_QUBIT_GATE_NAMES:
            return False
        return self.gate_timing.is_virtual(gate_name)

    def is_action_valid(self, state: State, action: Action) -> bool:
        """Return whether one action can start in a machine state.

        This checks only the action's own start conditions. Use
        :func:`~mqt.ionshuttler.linear.validation.is_transport_layer_valid`
        when several transports start together.

        Raises:
            TypeError: If the architecture defines no rules for the action.
        """
        if isinstance(action, Shuttle):
            return self._is_shuttle_valid(state, action)
        if isinstance(action, PhysicalSwap):
            return _is_swap_valid(state, action)
        if isinstance(action, GateAction):
            return self._is_gate_valid(state, action)
        msg = f"LinearArchitecture defines no rules for {type(action).__name__}"
        raise TypeError(msg)

    def apply_action(self, state: State, action: Action) -> State:
        """Apply the hardware effect of one valid action.

        The effect changes ion positions and resource availability. Circuit
        progress stays unchanged; the compiler records it separately.

        Returns:
            The updated state.

        Raises:
            TypeError: If the architecture defines no rules for the action.
        """
        if isinstance(action, Shuttle):
            positions = dict(state.positions)
            ions_busy = dict(state.ions_busy_until)
            positions[action.ion] = action.dst
            ions_busy[action.ion] = state.time + self.transport_timing.shuttle
            return replace(state, positions=tuple(positions.items()), ions_busy_until=tuple(ions_busy.items()))
        if isinstance(action, PhysicalSwap):
            positions = dict(state.positions)
            ions_busy = dict(state.ions_busy_until)
            positions[action.ion_a], positions[action.ion_b] = positions[action.ion_b], positions[action.ion_a]
            free_time = state.time + self.transport_timing.swap
            ions_busy[action.ion_a] = free_time
            ions_busy[action.ion_b] = free_time
            return replace(state, positions=tuple(positions.items()), ions_busy_until=tuple(ions_busy.items()))
        if isinstance(action, GateAction):
            return self._apply_gate(state, action)
        msg = f"LinearArchitecture defines no rules for {type(action).__name__}"
        raise TypeError(msg)

    def replay_schedule(self, schedule: Schedule) -> LinearMachineState:
        """Replay a schedule and return its final Linear machine state.

        Returns:
            The final machine state.
        """
        final_state = replay_linear_schedule(schedule, self)
        return LinearMachineState.from_compiler_state(final_state)

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
            "gate_timing": self.gate_timing.to_dict(),
            "transport_timing": self.transport_timing.to_dict(),
        }
        field_profile = self._field_profile()
        if _has_nontrivial_field_profile(field_profile):
            result["field_profile"] = field_profile.to_dict()
        if self.supported_action_types != DEFAULT_ACTION_TYPES:
            result["supported_action_types"] = [
                action_type.serialized_type for action_type in self.supported_action_types
            ]
        return result

    def to_json(self) -> str:
        """Serialize architecture metadata as JSON.

        Returns:
            The serialized architecture object.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> LinearArchitecture:
        """Construct an architecture from a JSON-style mapping.

        Returns:
            A validated architecture.

        Raises:
            TypeError: If the architecture or site count has the wrong type.
            ValueError: If the mapping has an invalid shape or values.
        """
        if not isinstance(data, dict):
            msg = "architecture must be a JSON object"
            raise TypeError(msg)
        mapping = cast("dict[str, object]", data)
        unknown_fields = sorted(set(mapping).difference(_SERIALIZED_FIELDS))
        if unknown_fields:
            msg = f"unknown architecture fields: {', '.join(unknown_fields)}"
            raise ValueError(msg)

        num_sites = mapping.get("num_sites")
        if isinstance(num_sites, bool) or not isinstance(num_sites, int):
            msg = "architecture.num_sites must be an integer"
            raise TypeError(msg)

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

        gate_timing_raw = mapping.get("gate_timing")
        transport_timing_raw = mapping.get("transport_timing")

        supported_action_names = mapping.get("supported_action_types")
        supported_action_types = DEFAULT_ACTION_TYPES
        if supported_action_names is not None:
            if not isinstance(supported_action_names, list) or any(
                not isinstance(name, str) for name in supported_action_names
            ):
                msg = "architecture.supported_action_types must be a list of strings"
                raise TypeError(msg)
            try:
                supported_action_types = tuple(LINEAR_ACTION_TYPES[name] for name in supported_action_names)
            except KeyError as error:
                msg = f"unknown architecture action type: {error.args[0]!r}"
                raise ValueError(msg) from error

        return cls(
            num_sites=num_sites,
            processing_zones=processing_zones,
            field_profile=field_profile,
            gate_timing=GateTiming() if gate_timing_raw is None else GateTiming.from_dict(gate_timing_raw),
            transport_timing=(
                TransportTiming() if transport_timing_raw is None else TransportTiming.from_dict(transport_timing_raw)
            ),
            supported_action_types=supported_action_types,
        )

    @classmethod
    def from_json(cls, raw: str) -> LinearArchitecture:
        """Deserialize an architecture from JSON.

        Returns:
            A validated architecture.
        """
        return cls.from_dict(json.loads(raw))

    @classmethod
    def load(cls, filename: str | Path) -> LinearArchitecture:
        """Load an architecture from a UTF-8 JSON file.

        Returns:
            A validated architecture.
        """
        return cls.from_json(Path(filename).read_text(encoding="utf-8"))

    def _is_shuttle_valid(self, state: State, action: Shuttle) -> bool:
        positions = dict(state.positions)
        return (
            positions.get(action.ion) == action.src
            and 0 <= action.dst < self.num_sites
            and is_adjacent(action.src, action.dst)
            and action.dst not in positions.values()
            and dict(state.ions_busy_until).get(action.ion, state.time + 1) <= state.time
        )

    def _is_gate_valid(self, state: State, gate: GateAction) -> bool:
        positions = dict(state.positions)
        if isinstance(gate, GlobalGate):
            return all(ion in positions for ion in gate.ions)
        ions_busy = dict(state.ions_busy_until)
        if isinstance(gate, SingleQubitGate):
            if self.is_virtual_gate(gate):
                return gate.ion in positions
            position = positions.get(gate.ion)
            if position is None or ions_busy.get(gate.ion, state.time + 1) > state.time:
                return False
            zone = self.get_processing_zone(position)
            return zone is not None and dict(state.pzs_busy_until).get(zone, state.time + 1) <= state.time
        if isinstance(gate, TwoQubitGate):
            pos_a = positions.get(gate.ion_a)
            pos_b = positions.get(gate.ion_b)
            if (
                pos_a is None
                or pos_b is None
                or ions_busy.get(gate.ion_a, state.time + 1) > state.time
                or ions_busy.get(gate.ion_b, state.time + 1) > state.time
            ):
                return False
            zone_a = self.get_processing_zone(pos_a)
            zone_b = self.get_processing_zone(pos_b)
            return (
                zone_a is not None
                and zone_a == zone_b
                and dict(state.pzs_busy_until).get(zone_a, state.time + 1) <= state.time
            )
        msg = f"LinearArchitecture defines no rules for {type(gate).__name__}"
        raise TypeError(msg)

    def _apply_gate(self, state: State, gate: GateAction) -> State:
        # Global gates act through a schedule-wide control field and reserve no
        # ions or processing zones in the Linear model.
        if isinstance(gate, GlobalGate) or (isinstance(gate, SingleQubitGate) and self.is_virtual_gate(gate)):
            return state
        positions = dict(state.positions)
        ions_busy = dict(state.ions_busy_until)
        pzs_busy = dict(state.pzs_busy_until)
        free_time = state.time + self.action_duration(gate)
        if isinstance(gate, SingleQubitGate):
            ions_busy[gate.ion] = free_time
            zone = self.get_processing_zone(positions[gate.ion])
        elif isinstance(gate, TwoQubitGate):
            ions_busy[gate.ion_a] = free_time
            ions_busy[gate.ion_b] = free_time
            zone = self.get_processing_zone(positions[gate.ion_a])
        else:
            msg = f"LinearArchitecture defines no rules for {type(gate).__name__}"
            raise TypeError(msg)
        if zone is not None:
            pzs_busy[zone] = free_time
        return replace(state, ions_busy_until=tuple(ions_busy.items()), pzs_busy_until=tuple(pzs_busy.items()))

    def _processing_zones(self) -> Mapping[str, tuple[int, ...]]:
        return cast("Mapping[str, tuple[int, ...]]", self.processing_zones)

    def _field_profile(self) -> FieldProfile:
        return cast("FieldProfile", self.field_profile)


def _is_swap_valid(state: State, action: PhysicalSwap) -> bool:
    positions = dict(state.positions)
    ions_busy = dict(state.ions_busy_until)
    return (
        action.ion_a != action.ion_b
        and positions.get(action.ion_a) == action.pos_a
        and positions.get(action.ion_b) == action.pos_b
        and ions_busy.get(action.ion_a, state.time + 1) <= state.time
        and ions_busy.get(action.ion_b, state.time + 1) <= state.time
        and is_adjacent(action.pos_a, action.pos_b)
    )


def _normalize_processing_zones(
    num_sites: int,
    processing_zones: Mapping[str, Sequence[int]] | None,
) -> Mapping[str, tuple[int, ...]]:
    """Return validated processing zones as a read-only mapping.

    Returns:
        Sorted zone sites keyed by zone name.

    Raises:
        TypeError: If the mapping, a zone name, or a site has the wrong type.
        ValueError: If a zone name is empty or the zone sites are invalid.
    """
    if processing_zones is not None and not isinstance(processing_zones, Mapping):
        msg = "processing_zones must be a mapping or None"
        raise TypeError(msg)
    if not processing_zones:
        return MappingProxyType({IMPLICIT_PROCESSING_ZONE: tuple(range(num_sites))})

    normalized: dict[str, tuple[int, ...]] = {}
    seen_sites: set[int] = set()
    for zone_name, zone_sites in processing_zones.items():
        if not isinstance(zone_name, str):
            msg = "processing zone names must be strings"
            raise TypeError(msg)
        if not zone_name:
            msg = "processing zone names must not be empty"
            raise ValueError(msg)
        if (
            isinstance(zone_sites, str)
            or not isinstance(zone_sites, Sequence)
            or any(isinstance(site, bool) or not isinstance(site, int) for site in zone_sites)
        ):
            msg = f"processing zone '{zone_name}' must be a sequence of integer sites"
            raise TypeError(msg)
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
    return MappingProxyType(normalized)


def _valid_two_qubit_site_pairs(
    processing_zones: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right) for zone_sites in processing_zones.values() for left, right in combinations(zone_sites, 2)
    )


def _has_nontrivial_field_profile(field_profile: FieldProfile) -> bool:
    return any(
        value != 1.0  # ruff: ignore[float-equality-comparison] - One is the exact default field value.
        for _, value in field_profile.site_field
    )


def _validate_site(site: int, num_sites: int) -> None:
    if not 0 <= site < num_sites:
        msg = f"site must be within [0, {num_sites - 1}]"
        raise ValueError(msg)


def _require_int_sequence(value: object, label: str) -> list[int]:
    if not isinstance(value, list | tuple) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        msg = f"{label} must be a list of integers"
        raise ValueError(msg)
    return list(cast("list[int] | tuple[int, ...]", value))


__all__ = ["IMPLICIT_PROCESSING_ZONE", "LinearArchitecture"]
