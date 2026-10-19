# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Implement the permutation-based Linear compiler strategy.

Fine-grained pre-partitioning supplies a preferred processing zone for each
gate. The strategy then repeats four steps until every gate has run:

1. Pick a group of ready gates that one static ion placement can run.
2. Build one complete target placement for the group. A placement records the
   site of every ion and therefore includes the ion order and empty sites.
3. Route from the current placement to the target with the fewest adjacent
   exchanges. Ion-ion exchanges are swaps; ion-empty-site exchanges are
   shuttles.
4. Schedule the transports and gates at their earliest legal start times.

Every action starts at its earliest legal time, so moves and gates of
neighboring groups can overlap. For a fixed architecture and lookahead, the
work depends on the number of groups and the transport actions that the
compiler emits. The fixed-placement route is minimal, but the target placements
and final makespan need not be optimal.
"""

from __future__ import annotations

import operator
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from time import perf_counter
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.gates import GlobalGate, SingleQubitGate, TwoQubitGate
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.linear.actions import PhysicalSwap, Shuttle
from mqt.ionshuttler.linear.result import GateGroup, LinearDiagnostics, PermutationDiagnostics
from mqt.ionshuttler.linear.schedule import LinearMachineState
from mqt.ionshuttler.linear.state import normalize_initial_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.circuit import Circuit
    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.linear.actions import TransportAction
    from mqt.ionshuttler.linear.architecture import LinearArchitecture
    from mqt.ionshuttler.linear.config import PermutationConfig
    from mqt.ionshuttler.linear.result import LinearCompilationResult
    from mqt.ionshuttler.linear.state import State


class _GateKind(Enum):
    """Classify gates by the processing-zone space they need."""

    NO_PROCESSING_ZONE = auto()
    ONE_ION = auto()
    TWO_IONS = auto()


@dataclass(frozen=True)
class _ProcessingZone:
    """Describe one processing zone as a range of sites."""

    name: str
    first_site: int
    last_site: int

    @property
    def capacity(self) -> int:
        """Number of ions that the zone can hold."""
        return self.last_site - self.first_site + 1

    def distance(self, site: int) -> int:
        """Return the number of sites between a site and the zone."""
        if site < self.first_site:
            return self.first_site - site
        if site > self.last_site:
            return site - self.last_site
        return 0


class _UnavailableTransportError(Exception):
    """Report that the target placement needs an unavailable transport type."""


@dataclass(frozen=True)
class _ZoneMember:
    """Record an assigned ion and its position among unassigned ions."""

    ion: int
    preceding_unassigned_ions: int


@dataclass(frozen=True)
class _ScheduleEntry:
    """Store one action before final schedule order and IDs are known."""

    start_time: int
    insertion_order: int
    action: Action
    duration: int
    processing_zone_id: str | None

    @property
    def end_time(self) -> int:
        """First timestep after the action."""
        return self.start_time + self.duration


def route_to_placement(
    source_site_of_ion: Sequence[int],
    target_site_of_ion: Sequence[int],
    num_sites: int,
) -> list[TransportAction]:
    """Return transports that move every ion from its source to its target site.

    The route uses the fewest adjacent exchanges. An exchange of two ions is a
    physical swap, and an exchange of an ion with an empty site is a shuttle.
    Empty sites keep their left-to-right order, so two empty sites never
    exchange. The number of physical swaps is therefore the number of ion
    pairs whose order differs between source and target.

    The actions are valid in the given order, one after the other.

    Args:
        source_site_of_ion: Current site indexed by ion.
        target_site_of_ion: Target site indexed by ion.
        num_sites: Number of sites on the line.

    Returns:
        The transports in a valid order.

    Raises:
        ValueError: If a placement repeats a site or uses a site outside the line.
    """
    if len(source_site_of_ion) != len(target_site_of_ion):
        msg = "source and target placements must place the same ions"
        raise ValueError(msg)
    for site_of_ion in (source_site_of_ion, target_site_of_ion):
        if len(set(site_of_ion)) != len(site_of_ion) or any(not 0 <= site < num_sites for site in site_of_ion):
            msg = "placements must use distinct sites on the line"
            raise ValueError(msg)
    ion_at_site = [-1] * num_sites
    for ion, site in enumerate(source_site_of_ion):
        ion_at_site[site] = ion
    return _route_by_adjacent_exchanges(ion_at_site, target_site_of_ion)


def _route_by_adjacent_exchanges(
    ion_at_site: list[int],
    target_site_of_ion: Sequence[int],
) -> list[TransportAction]:
    """Return the exchanges that sort the line by target site.

    This insertion sort performs exactly one exchange per inverted pair, so its
    work is proportional to the number of sites plus the number of actions.
    ``ion_at_site`` holds the ion on each site, or ``-1`` for an empty site. The
    function changes ``ion_at_site`` to the target placement.

    Returns:
        The transports in a valid order.
    """
    num_sites = len(ion_at_site)
    target_is_occupied = [False] * num_sites
    for site in target_site_of_ion:
        target_is_occupied[site] = True
    empty_target_sites = iter([site for site in range(num_sites) if not target_is_occupied[site]])
    target_site_of_occupant = [target_site_of_ion[ion] if ion >= 0 else next(empty_target_sites) for ion in ion_at_site]

    actions: list[TransportAction] = []
    # Insertion sort starts at site 1 because site 0 is already a sorted prefix.
    # Every inversion is one necessary adjacent exchange, which makes the route
    # minimal for this fixed source and target placement.
    for first_unsorted_site in range(1, num_sites):
        right = first_unsorted_site
        while right > 0 and target_site_of_occupant[right - 1] > target_site_of_occupant[right]:
            left = right - 1
            left_ion = ion_at_site[left]
            right_ion = ion_at_site[right]
            # Ion-ion exchanges are physical swaps. An ion next to an empty
            # site can move with one shuttle.
            if left_ion >= 0 and right_ion >= 0:
                actions.append(PhysicalSwap(ion_a=left_ion, ion_b=right_ion, pos_a=left, pos_b=right))
            elif left_ion >= 0:
                actions.append(Shuttle(ion=left_ion, src=left, dst=right))
            else:
                actions.append(Shuttle(ion=right_ion, src=right, dst=left))
            ion_at_site[left], ion_at_site[right] = right_ion, left_ion
            target_site_of_occupant[left], target_site_of_occupant[right] = (
                target_site_of_occupant[right],
                target_site_of_occupant[left],
            )
            right = left
    return actions


@dataclass
class _PermutationCompilation:
    """Hold the mutable progress of one compilation."""

    architecture: LinearArchitecture
    circuit: Circuit
    processing_zones: tuple[_ProcessingZone, ...]
    gate_kind: list[_GateKind]
    gate_duration: list[int]
    preferred_zone: list[_ProcessingZone | None]
    site_of_ion: list[int]
    ion_at_site: list[int]
    ion_available_at: list[int]
    site_vacated_at: list[int]
    zone_available_at: dict[_ProcessingZone, int]
    gate_end_time: list[int]
    gate_is_planned: bytearray
    pending_gate_ids: deque[int]
    swaps_enabled: bool
    shuttles_enabled: bool
    lookahead_gates: int
    initial_time: int
    schedule_entries: list[_ScheduleEntry]
    # The schedule entries and ion sites after the last completed group. An
    # interruption inside a group returns this valid prefix.
    committed_entry_count: int
    group_count: int = 0
    transport_free_group_count: int = 0
    preferred_zone_gate_count: int = 0
    planning_trace: list[GateGroup] | None = None

    def plan_group(self) -> tuple[list[int], dict[int, _ProcessingZone]]:
        """Pick the next group of gates and the processing zone of each used ion.

        The planner inspects up to ``lookahead_gates`` unplanned gates in circuit
        order. A gate joins the group when all its predecessors are planned and
        its ions fit into the processing zones together with the ions that the
        group already uses.

        Returns:
            The gate IDs of the group in circuit order and the processing zone
            of each ion that a group gate uses.
        """
        gate_ids: list[int] = []
        zone_by_ion: dict[int, _ProcessingZone] = {}
        ion_count_by_zone = dict.fromkeys(self.processing_zones, 0)
        skipped_gate_ids: list[int] = []
        predecessors = self.circuit.predecessors
        for _ in range(min(self.lookahead_gates, len(self.pending_gate_ids))):
            gate_id = self.pending_gate_ids.popleft()
            predecessors_are_planned = all(self.gate_is_planned[predecessor] for predecessor in predecessors[gate_id])
            if predecessors_are_planned and self._try_assign_gate_ions(
                gate_id,
                zone_by_ion,
                ion_count_by_zone,
            ):
                self.gate_is_planned[gate_id] = 1
                gate_ids.append(gate_id)
            else:
                skipped_gate_ids.append(gate_id)
        # The skipped gates keep their circuit order at the front of the queue.
        self.pending_gate_ids.extendleft(reversed(skipped_gate_ids))
        return gate_ids, zone_by_ion

    def _try_assign_gate_ions(
        self,
        gate_id: int,
        zone_by_ion: dict[int, _ProcessingZone],
        ion_count_by_zone: dict[_ProcessingZone, int],
    ) -> bool:
        """Assign a gate's ions to zones if the current group has enough room.

        Returns:
            Whether the gate fits into the group.
        """
        kind = self.gate_kind[gate_id]
        if kind is _GateKind.NO_PROCESSING_ZONE:
            return True
        ions = self.circuit.gates[gate_id].ions
        if kind is _GateKind.ONE_ION:
            if ions[0] in zone_by_ion:
                return True
            return self._try_assign_ions_to_zone(
                ions,
                self.preferred_zone[gate_id],
                zone_by_ion,
                ion_count_by_zone,
            )
        first_zone = zone_by_ion.get(ions[0])
        second_zone = zone_by_ion.get(ions[1])
        if first_zone is not None and second_zone is not None:
            return first_zone == second_zone
        if first_zone is None and second_zone is None:
            return self._try_assign_ions_to_zone(
                ions,
                self.preferred_zone[gate_id],
                zone_by_ion,
                ion_count_by_zone,
            )
        zone = first_zone if first_zone is not None else second_zone
        assert zone is not None
        if ion_count_by_zone[zone] >= zone.capacity:
            return False
        unassigned_ion = ions[1] if first_zone is not None else ions[0]
        zone_by_ion[unassigned_ion] = zone
        ion_count_by_zone[zone] += 1
        return True

    def _try_assign_ions_to_zone(
        self,
        ions: tuple[int, ...],
        preferred_zone: _ProcessingZone | None,
        zone_by_ion: dict[int, _ProcessingZone],
        ion_count_by_zone: dict[_ProcessingZone, int],
    ) -> bool:
        """Assign the ions to the preferred zone or the nearest zone with room.

        Returns:
            Whether a zone had room for all ions.
        """
        if preferred_zone is not None and preferred_zone.capacity - ion_count_by_zone[preferred_zone] >= len(ions):
            selected_zone = preferred_zone
        else:
            selected_zone = None
            shortest_distance = 0
            for zone in self.processing_zones:
                if zone.capacity - ion_count_by_zone[zone] < len(ions):
                    continue
                distance = sum(zone.distance(self.site_of_ion[ion]) for ion in ions)
                if selected_zone is None or distance < shortest_distance:
                    selected_zone = zone
                    shortest_distance = distance
            if selected_zone is None:
                return False
        for ion in ions:
            zone_by_ion[ion] = selected_zone
        ion_count_by_zone[selected_zone] += len(ions)
        return True

    def target_placement(self, zone_by_ion: dict[int, _ProcessingZone]) -> list[int]:
        """Return a complete target site for every ion.

        Ions that already sit in their assigned zones keep the current
        placement. Otherwise, the ions of each zone form one block inside the
        zone. Each block enters the current ion order where it needs the fewest
        physical swaps. All other ions keep their order and stay as close to
        their current sites as the blocks permit.

        Returns:
            The target site of each ion, indexed by ion.
        """
        site_of_ion = self.site_of_ion
        if all(zone.distance(site_of_ion[ion]) == 0 for ion, zone in zone_by_ion.items()):
            return site_of_ion

        ion_order = [ion for ion in self.ion_at_site if ion >= 0]
        unassigned_ions: list[int] = []
        members_by_zone: dict[_ProcessingZone, list[_ZoneMember]] = {}
        for ion in ion_order:
            zone = zone_by_ion.get(ion)
            if zone is None:
                unassigned_ions.append(ion)
            else:
                members_by_zone.setdefault(zone, []).append(
                    _ZoneMember(ion=ion, preceding_unassigned_ions=len(unassigned_ions))
                )
        current_site_of_unassigned_ion = [site_of_ion[ion] for ion in unassigned_ions]
        active_zones = sorted(members_by_zone, key=operator.attrgetter("first_site"))

        # Each zone ion records how many unassigned ions currently sit to its
        # left. Moving all zone ions into one block makes each ion cross the
        # difference between that count and the block's insertion point. The
        # median minimizes the total crossings. If several positions tie,
        # choose the one nearest the processing zone.
        insertion_indices: list[int] = []
        previous_insertion_index = 0
        for zone in active_zones:
            preceding_counts = [member.preceding_unassigned_ions for member in members_by_zone[zone]]
            lowest_optimal_index = preceding_counts[(len(preceding_counts) - 1) // 2]
            highest_optimal_index = preceding_counts[len(preceding_counts) // 2]
            preferred_index = bisect_left(
                current_site_of_unassigned_ion,
                zone.first_site,
            )
            insertion_index = max(
                min(max(preferred_index, lowest_optimal_index), highest_optimal_index),
                previous_insertion_index,
            )
            insertion_indices.append(insertion_index)
            previous_insertion_index = insertion_index
        segment_bounds = [0, *insertion_indices, len(unassigned_ions)]
        unassigned_segments = [
            unassigned_ions[segment_bounds[index] : segment_bounds[index + 1]]
            for index in range(len(segment_bounds) - 1)
        ]

        block_starts = self._choose_block_starts(
            active_zones,
            members_by_zone,
            unassigned_segments,
        )
        self._fit_unassigned_segments(
            unassigned_segments,
            active_zones,
            block_starts,
            members_by_zone,
        )

        target_site_of_ion = [0] * len(site_of_ion)
        for block_index, zone in enumerate(active_zones):
            for offset, member in enumerate(members_by_zone[zone]):
                target_site_of_ion[member.ion] = block_starts[block_index] + offset
        num_sites = self.architecture.num_sites
        for segment_index, segment in enumerate(unassigned_segments):
            first_site = (
                0
                if segment_index == 0
                else block_starts[segment_index - 1] + len(members_by_zone[active_zones[segment_index - 1]])
            )
            last_site = num_sites - 1 if segment_index == len(active_zones) else block_starts[segment_index] - 1
            previous_site = first_site - 1
            last_offset = len(segment) - 1
            for offset, ion in enumerate(segment):
                site = min(
                    max(site_of_ion[ion], first_site + offset),
                    last_site - (last_offset - offset),
                )
                site = max(site, previous_site + 1)
                target_site_of_ion[ion] = site
                previous_site = site
        return target_site_of_ion

    def _choose_block_starts(
        self,
        active_zones: list[_ProcessingZone],
        members_by_zone: dict[_ProcessingZone, list[_ZoneMember]],
        unassigned_segments: list[list[int]],
    ) -> list[int]:
        """Return the first site of each block, shifted inside its zone to make room."""
        block_starts: list[int] = []
        # Center each block around its members' current mean position, then
        # clamp the complete block to its processing zone.
        for zone in active_zones:
            members = members_by_zone[zone]
            block_size = len(members)
            mean_site = sum(self.site_of_ion[member.ion] for member in members) / block_size
            latest_start = zone.last_site - block_size + 1
            block_starts.append(min(max(round(mean_site - (block_size - 1) / 2), zone.first_site), latest_start))

        # The forward and backward passes reserve enough sites for each
        # unassigned segment while keeping every block inside its zone.
        previous_end = -1
        for block_index, zone in enumerate(active_zones):
            block_size = len(members_by_zone[zone])
            earliest_start = previous_end + 1 + len(unassigned_segments[block_index])
            start = max(block_starts[block_index], earliest_start)
            block_starts[block_index] = min(
                start,
                zone.last_site - block_size + 1,
            )
            previous_end = block_starts[block_index] + block_size - 1
        next_start = self.architecture.num_sites
        for block_index in reversed(range(len(active_zones))):
            zone = active_zones[block_index]
            block_size = len(members_by_zone[zone])
            latest_start = next_start - len(unassigned_segments[block_index + 1]) - block_size
            start = min(block_starts[block_index], latest_start)
            block_starts[block_index] = max(start, zone.first_site)
            next_start = block_starts[block_index]
        return block_starts

    def _fit_unassigned_segments(
        self,
        unassigned_segments: list[list[int]],
        active_zones: list[_ProcessingZone],
        block_starts: list[int],
        members_by_zone: dict[_ProcessingZone, list[_ZoneMember]],
    ) -> None:
        """Move unassigned ions across blocks until every segment fits."""
        segment_capacities: list[int] = []
        previous_end = -1
        for block_index, zone in enumerate(active_zones):
            segment_capacities.append(block_starts[block_index] - previous_end - 1)
            previous_end = block_starts[block_index] + len(members_by_zone[zone]) - 1
        segment_capacities.append(self.architecture.num_sites - previous_end - 1)

        # First push left-side overflow right. Then push any remaining
        # right-side overflow left. Both passes preserve global ion order.
        for segment_index in range(len(unassigned_segments) - 1):
            overflow = len(unassigned_segments[segment_index]) - segment_capacities[segment_index]
            if overflow > 0:
                unassigned_segments[segment_index + 1] = (
                    unassigned_segments[segment_index][-overflow:] + unassigned_segments[segment_index + 1]
                )
                del unassigned_segments[segment_index][-overflow:]
        for segment_index in reversed(range(1, len(unassigned_segments))):
            overflow = len(unassigned_segments[segment_index]) - segment_capacities[segment_index]
            if overflow > 0:
                unassigned_segments[segment_index - 1].extend(unassigned_segments[segment_index][:overflow])
                del unassigned_segments[segment_index][:overflow]

    def schedule_move_to(self, target_site_of_ion: list[int]) -> tuple[int, int]:
        """Schedule the route to a target placement at the earliest legal times.

        Returns:
            The numbers of physical swaps and shuttles in the route.

        Raises:
            _UnavailableTransportError: If the route needs a transport type that is unavailable.
        """
        if target_site_of_ion == self.site_of_ion:
            return 0, 0
        ion_at_site = list(self.ion_at_site)
        actions = _route_by_adjacent_exchanges(ion_at_site, target_site_of_ion)
        swap_count = sum(isinstance(action, PhysicalSwap) for action in actions)
        shuttle_count = len(actions) - swap_count
        if (swap_count and not self.swaps_enabled) or (shuttle_count and not self.shuttles_enabled):
            raise _UnavailableTransportError
        timing = self.architecture.transport_timing
        schedule_entries = self.schedule_entries
        for action in actions:
            if isinstance(action, Shuttle):
                # The target site must be empty: its last occupant has left.
                start_time = max(
                    self.ion_available_at[action.ion],
                    self.site_vacated_at[action.dst],
                )
                self.site_vacated_at[action.src] = start_time
                self.ion_available_at[action.ion] = start_time + timing.shuttle
                duration = timing.shuttle
            else:
                assert isinstance(action, PhysicalSwap)
                start_time = max(
                    self.ion_available_at[action.ion_a],
                    self.ion_available_at[action.ion_b],
                )
                duration = timing.swap
                end_time = start_time + duration
                self.ion_available_at[action.ion_a] = end_time
                self.ion_available_at[action.ion_b] = end_time
            schedule_entries.append(
                _ScheduleEntry(
                    start_time=start_time,
                    insertion_order=len(schedule_entries),
                    action=action,
                    duration=duration,
                    processing_zone_id=None,
                )
            )
        self.site_of_ion = target_site_of_ion
        self.ion_at_site = ion_at_site
        return swap_count, shuttle_count

    def schedule_gates(self, gate_ids: list[int]) -> None:
        """Schedule the gates of a group at their earliest legal times."""
        gates = self.circuit.gates
        predecessors = self.circuit.predecessors
        for gate_id in gate_ids:
            gate = gates[gate_id]
            ions = gate.ions
            duration = self.gate_duration[gate_id]
            start_time = max(
                (self.gate_end_time[predecessor] for predecessor in predecessors[gate_id]),
                default=self.initial_time,
            )
            processing_zone_id = None
            if self.gate_kind[gate_id] is not _GateKind.NO_PROCESSING_ZONE:
                zone = self._processing_zone_at(self.site_of_ion[ions[0]])
                start_time = max(
                    start_time,
                    self.zone_available_at[zone],
                    *(self.ion_available_at[ion] for ion in ions),
                )
                processing_zone_id = zone.name
                if zone == self.preferred_zone[gate_id]:
                    self.preferred_zone_gate_count += 1
                end_time = start_time + duration
                self.zone_available_at[zone] = end_time
                for ion in ions:
                    self.ion_available_at[ion] = end_time
            else:
                # Global and virtual gates use no local ion or zone resource
                # in the Linear hardware model, so they may overlap transport.
                end_time = start_time + duration
            self.gate_end_time[gate_id] = end_time
            self.schedule_entries.append(
                _ScheduleEntry(
                    start_time=start_time,
                    insertion_order=len(self.schedule_entries),
                    action=gate,
                    duration=duration,
                    processing_zone_id=processing_zone_id,
                )
            )

    def commit_group(
        self,
        gate_ids: list[int],
        zone_by_ion: dict[int, _ProcessingZone],
        swap_count: int,
        shuttle_count: int,
    ) -> None:
        """Count a completed group, record it when tracing, and keep the schedule prefix."""
        self.group_count += 1
        if swap_count == shuttle_count == 0:
            self.transport_free_group_count += 1
        if self.planning_trace is not None:
            group_entries = self.schedule_entries[self.committed_entry_count :]
            self.planning_trace.append(
                GateGroup(
                    gate_ids=tuple(gate_ids),
                    zone_by_ion=tuple((ion, zone.name) for ion, zone in sorted(zone_by_ion.items())),
                    target_site_of_ion=tuple(self.site_of_ion),
                    swaps=swap_count,
                    shuttles=shuttle_count,
                    start_time=min(entry.start_time for entry in group_entries),
                    end_time=max(entry.end_time for entry in group_entries),
                )
            )
        self.committed_entry_count = len(self.schedule_entries)

    def _processing_zone_at(self, site: int) -> _ProcessingZone:
        # The target placement puts every gate ion in a processing zone.
        return next(zone for zone in self.processing_zones if zone.first_site <= site <= zone.last_site)


def compile_with_permutations(
    initial_state: State,
    circuit: Circuit,
    architecture: LinearArchitecture,
    config: PermutationConfig,
    *,
    preferred_zone_by_gate: Mapping[int, str] | None = None,
) -> LinearCompilationResult:
    """Compile a circuit by moving ions through a planned series of placements.

    Args:
        initial_state: Starting ion placement and resource availability.
        circuit: Parsed circuit to compile.
        architecture: Linear hardware model.
        config: Compiler settings. The permutation settings and the time budget
            apply.
        preferred_zone_by_gate: Preferred processing zone for each gate, for example from
            pre-partitioning. The planner uses another zone only when the
            preferred zone has no room.

    Returns:
        The schedule and completion status. A timeout, interruption, or
        failed group returns every completed group as a valid partial
        schedule. The architecture replays the schedule before it is returned.
    """
    compilation_started_at = perf_counter()
    deadline = None if config.max_compile_time is None else compilation_started_at + config.max_compile_time
    normalized_state = normalize_initial_state(initial_state, architecture)
    action_types = architecture.supported_action_types if config.action_types is None else config.action_types
    compilation = _initialize_compilation(
        normalized_state,
        circuit,
        architecture,
        config,
        action_types,
        preferred_zone_by_gate or {},
    )
    try:
        status = _run_compilation(compilation, deadline)
    except KeyboardInterrupt:
        status = CompilationStatus.INTERRUPTED
    return _build_result(compilation, normalized_state, status, compilation_started_at)


def _run_compilation(
    compilation: _PermutationCompilation,
    deadline: float | None,
) -> CompilationStatus:
    """Plan, route, and schedule gate groups until no gate is left.

    Returns:
        The completion status.
    """
    while compilation.pending_gate_ids:
        if deadline is not None and perf_counter() >= deadline:
            return CompilationStatus.TIMEOUT
        gate_ids, zone_by_ion = compilation.plan_group()
        if not gate_ids:
            return CompilationStatus.FAILED
        try:
            swap_count, shuttle_count = compilation.schedule_move_to(compilation.target_placement(zone_by_ion))
        except _UnavailableTransportError:
            return CompilationStatus.FAILED
        compilation.schedule_gates(gate_ids)
        compilation.commit_group(
            gate_ids,
            zone_by_ion,
            swap_count,
            shuttle_count,
        )
    return CompilationStatus.SUCCESS


def _initialize_compilation(
    state: State,
    circuit: Circuit,
    architecture: LinearArchitecture,
    config: PermutationConfig,
    action_types: Sequence[type[Action]],
    preferred_zone_by_gate: Mapping[int, str],
) -> _PermutationCompilation:
    """Build the lookup tables and availability times that the main loop works on.

    Returns:
        The compilation progress before the first group.

    Raises:
        TypeError: If the circuit contains a gate type that the strategy does not support.
    """
    processing_zones = tuple(
        sorted(
            (
                _ProcessingZone(name, sites[0], sites[-1])
                for name, sites in (architecture.processing_zones or {}).items()
            ),
            key=operator.attrgetter("first_site"),
        )
    )
    zone_by_name = {zone.name: zone for zone in processing_zones}
    gate_kind: list[_GateKind] = []
    for gate in circuit.gates:
        if isinstance(gate, GlobalGate) or (isinstance(gate, SingleQubitGate) and architecture.is_virtual_gate(gate)):
            gate_kind.append(_GateKind.NO_PROCESSING_ZONE)
        elif isinstance(gate, SingleQubitGate):
            gate_kind.append(_GateKind.ONE_ION)
        elif isinstance(gate, TwoQubitGate):
            gate_kind.append(_GateKind.TWO_IONS)
        else:
            msg = f"the permutation strategy supports no {type(gate).__name__}"
            raise TypeError(msg)

    site_of_ion = [0] * len(state.positions)
    for ion, site in state.positions:
        site_of_ion[ion] = site
    ion_at_site = [-1] * architecture.num_sites
    for ion, site in enumerate(site_of_ion):
        ion_at_site[site] = ion
    ion_available_at_by_ion = dict(state.ions_busy_until)
    zone_available_at_by_name = dict(state.pzs_busy_until)
    return _PermutationCompilation(
        architecture=architecture,
        circuit=circuit,
        processing_zones=processing_zones,
        gate_kind=gate_kind,
        gate_duration=[architecture.action_duration(gate) for gate in circuit.gates],
        preferred_zone=[zone_by_name.get(preferred_zone_by_gate.get(gate_id, "")) for gate_id in circuit.gate_ids],
        site_of_ion=site_of_ion,
        ion_at_site=ion_at_site,
        ion_available_at=[
            max(ion_available_at_by_ion.get(ion, state.time), state.time) for ion in range(len(site_of_ion))
        ],
        site_vacated_at=[state.time] * architecture.num_sites,
        zone_available_at={
            zone: max(zone_available_at_by_name.get(zone.name, state.time), state.time) for zone in processing_zones
        },
        gate_end_time=[0] * len(circuit.gates),
        gate_is_planned=bytearray(len(circuit.gates)),
        pending_gate_ids=deque(circuit.gate_ids),
        swaps_enabled=PhysicalSwap in action_types,
        shuttles_enabled=Shuttle in action_types,
        lookahead_gates=config.lookahead_gates,
        initial_time=state.time,
        schedule_entries=[],
        committed_entry_count=0,
        planning_trace=[] if config.record_planning_trace else None,
    )


def _build_result(
    compilation: _PermutationCompilation,
    initial_state: State,
    status: CompilationStatus,
    compilation_started_at: float,
) -> LinearCompilationResult:
    """Turn the actions of all completed groups into a replayed schedule and result.

    Returns:
        The compilation result with the given status.
    """
    schedule_entries = compilation.schedule_entries[: compilation.committed_entry_count]
    schedule_entries.sort(key=operator.attrgetter("start_time", "insertion_order"))
    scheduled_actions = tuple(
        ScheduledAction(
            action_id=action_id,
            action=entry.action,
            start_time=entry.start_time,
            duration=entry.duration,
            processing_zone_id=entry.processing_zone_id,
        )
        for action_id, entry in enumerate(schedule_entries)
    )
    end_time = max([initial_state.time, *(item.end_time for item in scheduled_actions)])
    schedule = Schedule(
        scheduled_actions,
        end_time,
        LinearMachineState.from_compiler_state(initial_state),
    )
    architecture = compilation.architecture
    final_state = architecture.replay_schedule(schedule)
    return CompilationResult(
        status=status,
        schedule=schedule,
        architecture=architecture,
        final_state=final_state,
        wall_clock_s=perf_counter() - compilation_started_at,
        diagnostics=LinearDiagnostics(
            score=final_state.time,
            explored_nodes=0,
            permutation=PermutationDiagnostics(
                groups=compilation.group_count,
                groups_without_moves=compilation.transport_free_group_count,
                gates_in_preferred_zone=compilation.preferred_zone_gate_count,
                planning_trace=(None if compilation.planning_trace is None else tuple(compilation.planning_trace)),
            ),
        ),
    )


__all__ = ["compile_with_permutations", "route_to_placement"]
