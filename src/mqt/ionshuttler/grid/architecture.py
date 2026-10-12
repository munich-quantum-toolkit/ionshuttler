# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hardware rules for segment-graph architectures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.gates import GATE_TYPES, GateAction, GateTiming, SingleQubitGate
from mqt.ionshuttler.grid.actions import (
    DEFAULT_ACTION_TYPES,
    GRID_ACTION_TYPES,
    Cycle,
    JunctionMove,
)
from mqt.ionshuttler.grid.model import (
    Junction,
    ProcessingZone,
    Segment,
    SegmentEndpoint,
    SegmentOccupancy,
    TransportTiming,
)
from mqt.ionshuttler.grid.state import GridMachineState

from .._json_utils import require_list, require_mapping, require_str_list

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction

GRID_ARCHITECTURE_SCHEMA = "mqt.ionshuttler.grid.architecture"
GRID_ARCHITECTURE_VERSION = 1
T = TypeVar("T")


@dataclass(frozen=True)
class GridArchitecture:
    """Describe Grid locations, junction topology, resources, and timing."""

    segments: tuple[Segment, ...]
    junctions: tuple[Junction, ...]
    gate_timing: GateTiming = field(default_factory=GateTiming)
    transport_timing: TransportTiming = field(default_factory=TransportTiming)
    supported_action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES
    _segments_by_id: dict[str, Segment] = field(init=False, repr=False, compare=False)
    _junctions_by_id: dict[str, Junction] = field(init=False, repr=False, compare=False)
    _endpoint_junctions: dict[SegmentEndpoint, str] = field(init=False, repr=False, compare=False)
    _zones_by_id: dict[str, ProcessingZone] = field(init=False, repr=False, compare=False)
    _zone_segments: dict[str, str] = field(init=False, repr=False, compare=False)
    _routing_adjacencies: dict[
        tuple[frozenset[tuple[str, str]] | None, bool],
        dict[str, tuple[str, ...]],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _routing_distances: dict[
        tuple[str, frozenset[tuple[str, str]] | None, bool],
        dict[str, int],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate and canonicalize the architecture.

        Raises:
            TypeError: If a field contains values of the wrong type.
            ValueError: If identities, references, or capabilities are invalid.
        """
        segments = _normalize_values(self.segments, Segment, "segments", key=lambda item: item.segment_id)
        junctions = _normalize_values(self.junctions, Junction, "junctions", key=lambda item: item.junction_id)
        zones = tuple(zone for segment in segments for zone in segment.processing_zones)
        zone_ids = [zone.zone_id for zone in zones]
        if len(set(zone_ids)) != len(zone_ids):
            msg = "processing zones must have globally unique identifiers"
            raise ValueError(msg)
        if not segments:
            msg = "GridArchitecture must contain at least one segment"
            raise ValueError(msg)
        segment_ids = {segment.segment_id for segment in segments}
        endpoint_junctions: dict[SegmentEndpoint, str] = {}
        for junction in junctions:
            for endpoint in junction.endpoints:
                if endpoint.segment_id not in segment_ids:
                    msg = f"junction {junction.junction_id!r} refers to an unknown segment"
                    raise ValueError(msg)
                previous = endpoint_junctions.get(endpoint)
                if previous is not None:
                    msg = (
                        f"segment endpoint {endpoint!r} belongs to junctions {previous!r} and {junction.junction_id!r}"
                    )
                    raise ValueError(msg)
                endpoint_junctions[endpoint] = junction.junction_id
        if not isinstance(self.gate_timing, GateTiming):
            msg = "gate_timing must be GateTiming"
            raise TypeError(msg)
        if not isinstance(self.transport_timing, TransportTiming):
            msg = "transport_timing must be TransportTiming"
            raise TypeError(msg)
        action_types = tuple(self.supported_action_types)
        if any(
            not isinstance(action_type, type) or not issubclass(action_type, Action) for action_type in action_types
        ):
            msg = "supported_action_types must contain Action subclasses"
            raise TypeError(msg)
        if len(set(action_types)) != len(action_types):
            msg = "supported_action_types must not contain duplicates"
            raise ValueError(msg)
        implemented = set(GRID_ACTION_TYPES.values())
        unknown = [action_type.__name__ for action_type in action_types if action_type not in implemented]
        if unknown:
            msg = f"GridArchitecture implements no action types named: {', '.join(unknown)}"
            raise ValueError(msg)
        for zone in zones:
            unsupported = [
                gate_type.__name__ for gate_type in zone.supported_gate_types if gate_type not in action_types
            ]
            if unsupported:
                msg = f"processing zone {zone.zone_id!r} declares unsupported gates: {', '.join(unsupported)}"
                raise ValueError(msg)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "junctions", junctions)
        object.__setattr__(self, "supported_action_types", action_types)
        object.__setattr__(self, "_segments_by_id", {segment.segment_id: segment for segment in segments})
        object.__setattr__(self, "_junctions_by_id", {junction.junction_id: junction for junction in junctions})
        object.__setattr__(self, "_endpoint_junctions", endpoint_junctions)
        object.__setattr__(self, "_zones_by_id", {zone.zone_id: zone for zone in zones})
        object.__setattr__(
            self,
            "_zone_segments",
            {zone.zone_id: segment.segment_id for segment in segments for zone in segment.processing_zones},
        )

    def supports(self, action_type: type[Action]) -> bool:
        """Return whether this architecture exposes an action type."""
        return action_type in self.supported_action_types

    def segment(self, segment_id: str) -> Segment:
        """Return a segment by stable identity."""
        return self._segments_by_id[segment_id]

    def junction_for(self, endpoint: SegmentEndpoint) -> Junction:
        """Return the junction attached to a segment endpoint."""
        return self._junctions_by_id[self._endpoint_junctions[endpoint]]

    @property
    def processing_zones(self) -> tuple[ProcessingZone, ...]:
        """All processing zones in segment and within-segment order."""
        return tuple(zone for segment in self.segments for zone in segment.processing_zones)

    def processing_zone(self, zone_id: str) -> ProcessingZone:
        """Return a processing zone by stable identity."""
        return self._zones_by_id[zone_id]

    def processing_zone_segment(self, zone_id: str) -> Segment:
        """Return the segment that owns a processing zone."""
        return self._segments_by_id[self._zone_segments[zone_id]]

    def initial_state(self, placement: Mapping[str, Sequence[int]] | None = None) -> GridMachineState:
        """Create a validated machine state from a segment-to-ion mapping.

        Omitted segments are empty. Ordered segments retain the supplied order.
        Unordered segments use ascending ion order as their canonical storage.

        Returns:
            The initial machine state.

        Raises:
            TypeError: If the placement is not a mapping of sequences.
            ValueError: If a segment, ion, or capacity constraint is invalid.
        """
        if placement is None:
            placement = {}
        if not isinstance(placement, Mapping):
            msg = "placement must be a mapping"
            raise TypeError(msg)
        unknown = sorted(set(placement).difference(self._segments_by_id))
        if unknown:
            msg = f"placement refers to unknown segments: {', '.join(unknown)}"
            raise ValueError(msg)
        occupancy: list[tuple[str, tuple[int, ...]]] = []
        all_ions: list[int] = []
        for segment in self.segments:
            raw_ions = placement.get(segment.segment_id, ())
            if isinstance(raw_ions, str) or not isinstance(raw_ions, Sequence):
                msg = f"placement for segment {segment.segment_id!r} must be a sequence"
                raise TypeError(msg)
            ions = tuple(raw_ions)
            if any(isinstance(ion, bool) or not isinstance(ion, int) for ion in ions):
                msg = f"placement for segment {segment.segment_id!r} must contain integers"
                raise TypeError(msg)
            if any(ion < 0 for ion in ions):
                msg = "ion identifiers must be non-negative"
                raise ValueError(msg)
            if len(ions) > segment.capacity:
                msg = f"placement exceeds capacity of segment {segment.segment_id!r}"
                raise ValueError(msg)
            if segment.occupancy is SegmentOccupancy.UNORDERED:
                ions = tuple(sorted(ions))
            occupancy.append((segment.segment_id, ions))
            all_ions.extend(ions)
        if len(set(all_ions)) != len(all_ions):
            msg = "placement must contain each ion exactly once"
            raise ValueError(msg)
        return GridMachineState(
            occupancy=tuple(occupancy),
            ions_busy_until=tuple((ion, 0) for ion in sorted(all_ions)),
            junctions_busy_until=tuple((junction.junction_id, 0) for junction in self.junctions),
            pzs_busy_until=tuple((zone.zone_id, 0) for zone in self.processing_zones),
        )

    def action_duration(self, action: Action) -> int:
        """Return the duration of one supported Grid action.

        Returns:
            The duration in ticks.

        Raises:
            TypeError: If the architecture defines no duration for the action.
        """
        if isinstance(action, JunctionMove | Cycle):
            return self.transport_timing.junction_move
        if isinstance(action, GateAction) and action.circuit_name is not None:
            return self.gate_timing.duration_for(action.circuit_name)
        msg = f"GridArchitecture defines no duration for {type(action).__name__}"
        raise TypeError(msg)

    def is_virtual_gate(self, action: Action) -> bool:
        """Return whether an action is a configured virtual gate."""
        return (
            isinstance(action, SingleQubitGate)
            and action.circuit_name is not None
            and self.gate_timing.is_virtual(action.circuit_name)
        )

    def apply_layer(
        self,
        state: GridMachineState,
        scheduled_actions: Sequence[ScheduledAction[Any]],
    ) -> GridMachineState:
        """Validate and jointly apply actions with one common start time.

        All actions of the layer are checked against the same state. A move may
        therefore enter a segment that another move of the layer leaves.

        Returns:
            The state after all actions start.

        Raises:
            TypeError: If the architecture defines no rules for an action.
            ValueError: If the state, timing, resources, or action layer is invalid.
        """
        self._validate_state(state)
        items = tuple(scheduled_actions)
        if not items:
            return state
        start_time = items[0].start_time
        if any(item.start_time != start_time for item in items):
            msg = "a Grid action layer must have one start time"
            raise ValueError(msg)
        if start_time < state.time:
            msg = "scheduled actions must not start before the machine state"
            raise ValueError(msg)
        current = state.at_time(start_time)
        occupancy = dict(current.occupancy)
        ion_busy = dict(current.ions_busy_until)
        junction_busy = dict(current.junctions_busy_until)
        pz_busy = dict(current.pzs_busy_until)
        claimed_ions: set[int] = set()
        claimed_junctions: set[str] = set()
        claimed_pzs: set[str] = set()
        moves: list[tuple[JunctionMove, int]] = []

        for item in items:
            action = item.action
            if not self.supports(type(action)):
                msg = f"unsupported Grid action type: {type(action).__name__}"
                raise ValueError(msg)
            expected_duration = self.action_duration(action)
            if item.duration != expected_duration:
                msg = f"action {item.action_id} has duration {item.duration}, expected {expected_duration}"
                raise ValueError(msg)
            if isinstance(action, JunctionMove | Cycle):
                if item.processing_zone_id is not None:
                    msg = "transport actions must not select a processing zone"
                    raise ValueError(msg)
                if isinstance(action, Cycle):
                    self._validate_cycle(action)
                    moves.extend((move, item.end_time) for move in action.moves)
                else:
                    moves.append((action, item.end_time))
            elif isinstance(action, GateAction):
                self._claim_gate(
                    current,
                    item,
                    action,
                    claimed_ions=claimed_ions,
                    claimed_pzs=claimed_pzs,
                    ion_busy=ion_busy,
                    pz_busy=pz_busy,
                )
            else:
                msg = f"GridArchitecture defines no rules for {type(action).__name__}"
                raise TypeError(msg)

        self._apply_moves(
            current,
            moves,
            occupancy=occupancy,
            claimed_ions=claimed_ions,
            claimed_junctions=claimed_junctions,
            ion_busy=ion_busy,
            junction_busy=junction_busy,
        )
        return GridMachineState(
            occupancy=tuple(occupancy.items()),
            ions_busy_until=tuple(ion_busy.items()),
            junctions_busy_until=tuple(junction_busy.items()),
            pzs_busy_until=tuple(pz_busy.items()),
            time=start_time,
        )

    def replay_schedule(self, schedule: Schedule[Action, GridMachineState]) -> GridMachineState:
        """Validate and replay a Grid schedule.

        Actions with the same start time form one layer, which
        :meth:`apply_layer` applies as one state transition. An invalid schedule raises
        :class:`ValueError`.

        Returns:
            The final Grid machine state at the schedule end time.
        """
        state = schedule.initial_state
        self._validate_state(state)
        for _start_time, layer in groupby(schedule.scheduled_actions, key=lambda item: item.start_time):
            state = self.apply_layer(state, tuple(layer))
        return state.at_time(schedule.end_time)

    def is_schedule_valid(self, schedule: Schedule[Action, GridMachineState]) -> bool:
        """Return whether a schedule replays without a validation error."""
        try:
            self.replay_schedule(schedule)
        except ValueError:
            return False
        return True

    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON-compatible architecture."""
        return {
            "schema": GRID_ARCHITECTURE_SCHEMA,
            "version": GRID_ARCHITECTURE_VERSION,
            "segments": [segment.to_dict() for segment in self.segments],
            "junctions": [junction.to_dict() for junction in self.junctions],
            "gate_timing": self.gate_timing.to_dict(),
            "transport_timing": self.transport_timing.to_dict(),
            "supported_action_types": [action_type.serialized_type for action_type in self.supported_action_types],
        }

    @classmethod
    def from_dict(cls, data: object) -> GridArchitecture:
        """Restore a Grid architecture from its versioned representation.

        Returns:
            The restored architecture.

        Raises:
            ValueError: If the schema, version, or action declarations are invalid.
        """
        mapping = require_mapping(data, "Grid architecture")
        if mapping.get("schema") != GRID_ARCHITECTURE_SCHEMA or mapping.get("version") != GRID_ARCHITECTURE_VERSION:
            msg = "unsupported Grid architecture schema or version"
            raise ValueError(msg)
        action_types: list[type[Action]] = []
        for name in require_str_list(mapping, "supported_action_types"):
            action_type = GRID_ACTION_TYPES.get(name)
            if action_type is None:
                msg = f"unknown Grid action type: {name}"
                raise ValueError(msg)
            action_types.append(action_type)
        gate_types = cast(
            "Mapping[str, type[GateAction]]",
            {name: action_type for name, action_type in GATE_TYPES.items() if issubclass(action_type, GateAction)},
        )
        return cls(
            segments=tuple(
                Segment.from_dict(item, gate_types=gate_types) for item in require_list(mapping, "segments")
            ),
            junctions=tuple(Junction.from_dict(item) for item in require_list(mapping, "junctions")),
            gate_timing=GateTiming.from_dict(mapping.get("gate_timing")),
            transport_timing=TransportTiming.from_dict(mapping.get("transport_timing")),
            supported_action_types=tuple(action_types),
        )

    def to_json(self) -> str:
        """Serialize this architecture as JSON text.

        Returns:
            The JSON document.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> GridArchitecture:
        """Restore a Grid architecture from JSON text.

        Returns:
            The restored architecture.
        """
        return cls.from_dict(json.loads(raw))

    @classmethod
    def load(cls, filename: str | Path) -> GridArchitecture:
        """Load a Grid architecture from a UTF-8 JSON file.

        Returns:
            The restored architecture.
        """
        return cls.from_json(Path(filename).read_text(encoding="utf-8"))

    def save(self, filename: str | Path) -> Path:
        """Write this architecture to an explicit UTF-8 JSON file.

        Returns:
            The path written.
        """
        output_path = Path(filename)
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")
        return output_path

    def _validate_state(self, state: GridMachineState) -> None:
        if {segment_id for segment_id, _ions in state.occupancy} != self._segments_by_id.keys():
            msg = "Grid state must contain exactly the architecture segments"
            raise ValueError(msg)
        if {junction_id for junction_id, _time in state.junctions_busy_until} != {
            junction.junction_id for junction in self.junctions
        }:
            msg = "Grid state must contain exactly the architecture junctions"
            raise ValueError(msg)
        if {zone_id for zone_id, _time in state.pzs_busy_until} != self._zones_by_id.keys():
            msg = "Grid state must contain exactly the architecture processing zones"
            raise ValueError(msg)
        for segment_id, ions in state.occupancy:
            segment = self._segments_by_id[segment_id]
            if len(ions) > segment.capacity:
                msg = f"state exceeds capacity of segment {segment_id!r}"
                raise ValueError(msg)
            if segment.occupancy is SegmentOccupancy.UNORDERED and ions != tuple(sorted(ions)):
                msg = f"unordered segment {segment_id!r} must use canonical ion order"
                raise ValueError(msg)

    def _junction_id_for(self, move: JunctionMove) -> str:
        source_junction = self._endpoint_junctions.get(move.source)
        destination_junction = self._endpoint_junctions.get(move.destination)
        if source_junction is None or destination_junction is None or source_junction != destination_junction:
            msg = "junction move endpoints must belong to the same junction"
            raise ValueError(msg)
        return source_junction

    def _claim_gate(
        self,
        state: GridMachineState,
        item: ScheduledAction[Any],
        gate: GateAction,
        *,
        claimed_ions: set[int],
        claimed_pzs: set[str],
        ion_busy: dict[int, int],
        pz_busy: dict[str, int],
    ) -> None:
        ions = gate.ions
        if any(ion not in ion_busy for ion in ions):
            msg = "gate refers to an ion outside the machine state"
            raise ValueError(msg)
        if self.is_virtual_gate(gate):
            if item.processing_zone_id is not None:
                msg = "virtual gates must not select a processing zone"
                raise ValueError(msg)
            return
        if any(ion_busy[ion] > state.time for ion in ions) or claimed_ions.intersection(ions):
            msg = "gate ion is busy or claimed by another action"
            raise ValueError(msg)
        if item.processing_zone_id is None:
            msg = "physical Grid gates must select a processing zone"
            raise ValueError(msg)
        zone = self._zones_by_id.get(item.processing_zone_id)
        if zone is None:
            msg = f"unknown processing zone {item.processing_zone_id!r}"
            raise ValueError(msg)
        if not zone.supports(gate):
            msg = f"processing zone {zone.zone_id!r} does not support {type(gate).__name__}"
            raise ValueError(msg)
        zone_segment_id = self._zone_segments[zone.zone_id]
        if any(state.ion_segment(ion) != zone_segment_id for ion in ions):
            msg = "all gate ions must occupy the selected processing-zone segment"
            raise ValueError(msg)
        if pz_busy[zone.zone_id] > state.time or zone.zone_id in claimed_pzs:
            msg = f"processing zone {zone.zone_id!r} is busy or claimed"
            raise ValueError(msg)
        claimed_pzs.add(zone.zone_id)
        pz_busy[zone.zone_id] = item.end_time
        claimed_ions.update(ions)
        for ion in ions:
            ion_busy[ion] = item.end_time

    def _validate_cycle(self, cycle: Cycle) -> None:
        successor: dict[str, str] = {}
        arrivals: set[str] = set()
        for move in cycle.moves:
            self._junction_id_for(move)
            source, destination = move.source, move.destination
            if source.segment_id in successor or destination.segment_id in arrivals:
                msg = "a cycle must enter and leave each participating segment once"
                raise ValueError(msg)
            successor[source.segment_id] = destination.segment_id
            arrivals.add(destination.segment_id)
        if arrivals != successor.keys():
            msg = "cycle moves must form a closed segment rotation"
            raise ValueError(msg)
        first = next(iter(successor))
        visited = {first}
        segment_id = successor[first]
        while segment_id != first:
            visited.add(segment_id)
            segment_id = successor[segment_id]
        if len(visited) != len(successor):
            msg = "cycle moves must form one closed segment rotation"
            raise ValueError(msg)

    def _apply_moves(
        self,
        state: GridMachineState,
        moves: Sequence[tuple[JunctionMove, int]],
        *,
        occupancy: dict[str, tuple[int, ...]],
        claimed_ions: set[int],
        claimed_junctions: set[str],
        ion_busy: dict[int, int],
        junction_busy: dict[str, int],
    ) -> None:
        removals: dict[str, list[tuple[Literal["start", "end"], tuple[int, ...]]]] = {}
        arrivals: dict[str, list[tuple[Literal["start", "end"], tuple[int, ...]]]] = {}
        for move, end_time in moves:
            source, destination = move.source, move.destination
            junction_id = self._junction_id_for(move)
            if any(ion not in ion_busy for ion in move.ions):
                msg = "junction move refers to an ion outside the machine state"
                raise ValueError(msg)
            if any(ion_busy[ion] > state.time for ion in move.ions) or claimed_ions.intersection(move.ions):
                msg = "transport ion is busy or claimed by another action"
                raise ValueError(msg)
            if junction_busy[junction_id] > state.time or junction_id in claimed_junctions:
                msg = f"junction {junction_id!r} is busy or claimed"
                raise ValueError(msg)
            self._validate_departure(source.segment_id, source.orientation, move.ions, occupancy)
            transferred = move.ions if source.orientation != destination.orientation else tuple(reversed(move.ions))
            removals.setdefault(source.segment_id, []).append((source.orientation, move.ions))
            arrivals.setdefault(destination.segment_id, []).append((destination.orientation, transferred))
            claimed_ions.update(move.ions)
            claimed_junctions.add(junction_id)
            junction_busy[junction_id] = end_time
            for ion in move.ions:
                ion_busy[ion] = end_time
        for entries in (*removals.values(), *arrivals.values()):
            ends = [end for end, _ions in entries]
            if len(set(ends)) != len(ends):
                msg = "an action layer must not use one segment end more than once"
                raise ValueError(msg)
        for segment_id, entries in removals.items():
            current = occupancy[segment_id]
            if self._segments_by_id[segment_id].occupancy is SegmentOccupancy.UNORDERED:
                removed = {ion for _end, ions in entries for ion in ions}
                occupancy[segment_id] = tuple(ion for ion in current if ion not in removed)
            else:
                # Chains from both ends cannot overlap because each ion moves at most once.
                start_count = sum(len(ions) for orientation, ions in entries if orientation == "start")
                end_count = sum(len(ions) for orientation, ions in entries if orientation == "end")
                occupancy[segment_id] = current[start_count : len(current) - end_count]
        for segment_id, entries in arrivals.items():
            segment = self._segments_by_id[segment_id]
            current = occupancy[segment_id]
            start_ions = tuple(ion for orientation, ions in entries if orientation == "start" for ion in ions)
            end_ions = tuple(ion for orientation, ions in entries if orientation == "end" for ion in ions)
            combined = (*start_ions, *current, *end_ions)
            if len(combined) > segment.capacity:
                msg = f"transport exceeds capacity of segment {segment_id!r}"
                raise ValueError(msg)
            occupancy[segment_id] = (
                tuple(sorted(combined)) if segment.occupancy is SegmentOccupancy.UNORDERED else combined
            )

    def _validate_departure(
        self,
        segment_id: str,
        orientation: Literal["start", "end"],
        ions: tuple[int, ...],
        occupancy: Mapping[str, tuple[int, ...]],
    ) -> None:
        current = occupancy[segment_id]
        if self._segments_by_id[segment_id].occupancy is SegmentOccupancy.UNORDERED:
            if not set(ions).issubset(current):
                msg = "junction move ions must occupy the source segment"
                raise ValueError(msg)
            return
        expected = current[: len(ions)] if orientation == "start" else current[-len(ions) :]
        if expected != ions:
            msg = "junction move ions must form the ordered chain at the selected segment end"
            raise ValueError(msg)


def _normalize_values(
    values: Sequence[T],
    expected_type: type[T],
    label: str,
    *,
    key: Callable[[T], str],
) -> tuple[T, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, expected_type) for value in normalized):
        msg = f"{label} must contain {expected_type.__name__} values"
        raise TypeError(msg)
    identifiers = [key(value) for value in normalized]
    if len(set(identifiers)) != len(identifiers):
        msg = f"{label} must have unique identifiers"
        raise ValueError(msg)
    return tuple(sorted(normalized, key=key))


__all__ = ["GRID_ARCHITECTURE_SCHEMA", "GRID_ARCHITECTURE_VERSION", "GridArchitecture"]
