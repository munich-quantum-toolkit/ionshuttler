# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded movement search for Grid compilation."""

from __future__ import annotations

from collections import deque
from itertools import combinations, pairwise
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.schedule import ScheduledAction
from mqt.ionshuttler.grid.actions import Cycle, JunctionMove
from mqt.ionshuttler.grid.model import SegmentOccupancy

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.core.gates import GateAction
    from mqt.ionshuttler.grid.architecture import GridArchitecture
    from mqt.ionshuttler.grid.state import GridMachineState


OccupancyKey = tuple[tuple[str, tuple[int, ...]], ...]
JunctionCrossings = frozenset[tuple[str, str]] | None


def route_gate_bfs(
    architecture: GridArchitecture,
    state: GridMachineState,
    gate: GateAction,
    target_segment_ids: frozenset[str],
    *,
    max_states: int,
    allowed_crossings: JunctionCrossings = None,
) -> tuple[tuple[Action, ...] | None, int]:
    """Find a shortest legal movement sequence that colocates one gate.

    Returns:
        The movement sequence, or ``None`` if the bound is exhausted, and the
        number of explored occupancy states and cycle-search steps.
    """
    queue: deque[tuple[GridMachineState, tuple[Action, ...]]] = deque([(state, ())])
    visited = {_occupancy_key(state)}
    explored = 0
    while queue and explored < max_states:
        current, path = queue.popleft()
        explored += 1
        if _gate_is_placed(current, gate, target_segment_ids):
            return path, explored
        moves = junction_moves(architecture, current, allowed_crossings=allowed_crossings)
        for move in moves:
            if move.destination.segment_id not in target_segment_ids or not any(ion in move.ions for ion in gate.ions):
                continue
            timed_move = ScheduledAction(0, move, current.time, architecture.action_duration(move))
            try:
                following = architecture.apply_layer(current, (timed_move,)).at_time(timed_move.end_time)
            except ValueError:
                continue
            if _gate_is_placed(following, gate, target_segment_ids):
                return (*path, move), explored
        cycles, cycle_work = _cycle_candidates(architecture, current, moves, max_work=max_states - explored)
        explored += cycle_work
        for action in (*moves, *cycles):
            duration = architecture.action_duration(action)
            timed_action = ScheduledAction(0, action, current.time, duration)
            try:
                following = architecture.apply_layer(current, (timed_action,)).at_time(timed_action.end_time)
            except ValueError:
                continue
            key = _occupancy_key(following)
            if key in visited:
                continue
            visited.add(key)
            queue.append((following, (*path, action)))
    return None, explored


def route_placement_bfs(
    architecture: GridArchitecture,
    state: GridMachineState,
    reached: Callable[[GridMachineState], bool],
    *,
    max_states: int,
    allowed_crossings: JunctionCrossings = None,
) -> tuple[tuple[Action, ...] | None, int]:
    """Find a shortest legal movement sequence to a placement goal.

    Returns:
        The movement sequence, or ``None`` if the bound is exhausted, and the
        number of explored occupancy states and cycle-search steps.
    """
    queue: deque[tuple[GridMachineState, tuple[Action, ...]]] = deque([(state, ())])
    visited = {_occupancy_key(state)}
    explored = 0
    while queue and explored < max_states:
        current, path = queue.popleft()
        explored += 1
        if reached(current):
            return path, explored
        moves = junction_moves(architecture, current, allowed_crossings=allowed_crossings)
        cycles, cycle_work = _cycle_candidates(architecture, current, moves, max_work=max_states - explored)
        explored += cycle_work
        for action in (*moves, *cycles):
            duration = architecture.action_duration(action)
            timed_action = ScheduledAction(0, action, current.time, duration)
            try:
                following = architecture.apply_layer(current, (timed_action,)).at_time(timed_action.end_time)
            except ValueError:
                continue
            if reached(following):
                return (*path, action), explored
            key = _occupancy_key(following)
            if key in visited:
                continue
            visited.add(key)
            queue.append((following, (*path, action)))
    return None, explored


def route_junction_crossing_greedy_cycle(
    architecture: GridArchitecture,
    state: GridMachineState,
    ion: int,
    destination_segment_id: str,
    *,
    max_states: int,
    moves: Sequence[JunctionMove],
    allowed_crossings: JunctionCrossings = None,
    ion_priorities: Mapping[int, int],
    excluded_occupancies: Collection[OccupancyKey] = (),
    allow_larger_cycles: bool = False,
) -> tuple[tuple[Action, ...] | None, int]:
    """Realize one requested junction crossing directly or with a cycle.

    Returns:
        The first legal transport actions and number of candidates examined.
    """
    source_segment_id = state.ion_segment(ion)
    explored = 0
    direct_moves = sorted(
        (
            move
            for move in moves
            if ion in move.ions
            and move.source.segment_id == source_segment_id
            and move.destination.segment_id == destination_segment_id
        ),
        key=lambda move: (len(move.ions), move.ions),
    )
    for move in direct_moves:
        if explored >= max_states:
            return None, explored
        explored += 1
        timed_move = ScheduledAction(0, move, state.time, architecture.action_duration(move))
        try:
            following = architecture.apply_layer(state, (timed_move,))
        except ValueError:
            continue
        if following.occupancy in excluded_occupancies:
            continue
        return (move,), explored

    adjacency = _segment_adjacency(architecture, allowed_crossings)
    candidate_actions, cycle_paths_explored = _find_cycle_transport(
        architecture,
        state,
        adjacency,
        ion,
        source_segment_id,
        destination_segment_id,
        max_paths=max_states - explored,
        moves=moves,
        allowed_crossings=allowed_crossings,
        ion_priorities=ion_priorities,
        excluded_occupancies=excluded_occupancies,
        allow_larger_cycles=allow_larger_cycles,
    )
    explored += cycle_paths_explored
    if candidate_actions is not None:
        return candidate_actions, explored
    path_clearing_moves, clearing_paths_explored = clear_blocked_path(
        architecture,
        state,
        ion,
        destination_segment_id,
        max_states=max_states - explored,
        moves=moves,
        allowed_crossings=allowed_crossings,
        ion_priorities=ion_priorities,
    )
    return path_clearing_moves, explored + clearing_paths_explored


def clear_blocked_path(
    architecture: GridArchitecture,
    state: GridMachineState,
    ion: int,
    destination_segment_id: str,
    *,
    max_states: int,
    moves: Sequence[JunctionMove],
    allowed_crossings: JunctionCrossings = None,
    ion_priorities: Mapping[int, int],
) -> tuple[tuple[JunctionMove, ...] | None, int]:
    """Clear a blocked path from its free-capacity end.

    If no path from the destination can be cleared, ions in front of the ion
    can still block its exit from the source segment. The function then clears
    a path for those ions into another segment at the same junction. The ion
    crosses the junction in a later layer.

    Returns:
        The legal path-clearing moves and number of segment paths examined.
    """
    source_segment_id = state.ion_segment(ion)
    adjacency = _segment_adjacency(architecture, allowed_crossings)
    queue: deque[tuple[str, ...]] = deque([(destination_segment_id,)])
    visited_segments = {source_segment_id, destination_segment_id}
    explored = 0
    while queue and explored < max_states:
        clearing_path = queue.popleft()
        explored += 1
        final_segment_id = clearing_path[-1]
        final_segment = architecture.segment(final_segment_id)
        if len(state.occupants(final_segment_id)) < final_segment.capacity:
            path_clearing_moves = _moves_to_clear_path(
                state,
                ion,
                (source_segment_id, *clearing_path),
                moves,
                ion_priorities=ion_priorities,
            )
            if path_clearing_moves is not None:
                legal_moves = _legal_path_clearing_moves(architecture, state, path_clearing_moves)
                if legal_moves:
                    return legal_moves, explored
        for neighbor in adjacency[final_segment_id]:
            if neighbor in visited_segments:
                continue
            visited_segments.add(neighbor)
            queue.append((*clearing_path, neighbor))
    exit_clearing_moves, exit_paths_explored = _clear_exit(
        architecture,
        state,
        ion,
        destination_segment_id,
        max_states=max_states - explored,
        moves=moves,
        allowed_crossings=allowed_crossings,
        ion_priorities=ion_priorities,
    )
    return exit_clearing_moves, explored + exit_paths_explored


def _clear_exit(
    architecture: GridArchitecture,
    state: GridMachineState,
    ion: int,
    destination_segment_id: str,
    *,
    max_states: int,
    moves: Sequence[JunctionMove],
    allowed_crossings: JunctionCrossings,
    ion_priorities: Mapping[int, int],
) -> tuple[tuple[JunctionMove, ...] | None, int]:
    """Clear a path for the ions in front of an ion into another segment at its exit junction.

    The ions in front must not have a higher priority than the ion. The cleared
    path does not enter the destination, so the ion keeps the free capacity
    there.

    Returns:
        The legal path-clearing moves and number of segment paths examined.
    """
    source_segment_id = state.ion_segment(ion)
    crossings = [
        move
        for move in moves
        if ion in move.ions
        and move.source.segment_id == source_segment_id
        and move.destination.segment_id == destination_segment_id
    ]
    if not crossings:
        return None, 0
    # The shortest crossing chain holds the ion and the ions in front of it.
    crossing = min(crossings, key=lambda move: len(move.ions))
    front_ions = tuple(other for other in crossing.ions if other != ion)
    missing_priority = len(ion_priorities)
    rank = ion_priorities.get(ion, missing_priority)
    if not front_ions or any(ion_priorities.get(other, missing_priority) < rank for other in front_ions):
        return None, 0
    # A chain that leaves with the nearest front ion but without the ion holds all front ions.
    nearest_front_ion = crossing.ions[1] if crossing.ions[0] == ion else crossing.ions[-2]
    exit_moves = tuple(
        move for move in moves if ion not in move.ions and move.destination.segment_id != destination_segment_id
    )
    side_segment_ids = sorted({
        move.destination.segment_id
        for move in exit_moves
        if move.source == crossing.source and nearest_front_ion in move.ions
    })
    explored = 0
    for side_segment_id in side_segment_ids:
        side_clearing_moves, side_paths_explored = clear_blocked_path(
            architecture,
            state,
            nearest_front_ion,
            side_segment_id,
            max_states=max_states - explored,
            moves=exit_moves,
            allowed_crossings=allowed_crossings,
            ion_priorities=ion_priorities,
        )
        explored += side_paths_explored
        if side_clearing_moves:
            return side_clearing_moves, explored
    return None, explored


def next_segments_toward(
    architecture: GridArchitecture,
    source_segment_id: str,
    target_segment_id: str,
    allowed_crossings: JunctionCrossings = None,
) -> tuple[str, ...]:
    """Return all next segments on shortest directed routes to a target."""
    distances = _segment_distances(architecture, target_segment_id, allowed_crossings, reverse=True)
    distance = distances.get(source_segment_id)
    if distance is None or distance == 0:
        return ()
    adjacency = _segment_adjacency(architecture, allowed_crossings)
    return tuple(neighbor for neighbor in adjacency[source_segment_id] if distances.get(neighbor) == distance - 1)


def routing_distance(
    architecture: GridArchitecture,
    source_segment_id: str,
    target_segment_id: str,
    allowed_crossings: JunctionCrossings = None,
) -> int | None:
    """Return directed segment distance to a target, or ``None`` if unreachable."""
    distances = _segment_distances(architecture, target_segment_id, allowed_crossings, reverse=True)
    return distances.get(source_segment_id)


def _find_cycle_transport(
    architecture: GridArchitecture,
    state: GridMachineState,
    adjacency: dict[str, tuple[str, ...]],
    requested_ion: int,
    source_segment_id: str,
    destination_segment_id: str,
    *,
    max_paths: int,
    moves: Sequence[JunctionMove],
    allowed_crossings: JunctionCrossings,
    ion_priorities: Mapping[int, int],
    excluded_occupancies: Collection[OccupancyKey],
    allow_larger_cycles: bool,
) -> tuple[tuple[Action, ...] | None, int]:
    """Find the first usable junction-simple cycle containing one crossing.

    Returns:
        The cycle transport and number of paths examined.
    """
    # Map each directed segment pair to the physical junction that connects it.
    junction_by_segments = {
        (source.segment_id, destination.segment_id): junction.junction_id
        for junction in architecture.junctions
        for source in junction.endpoints
        for destination in junction.endpoints
        if source.segment_id != destination.segment_id
        and (allowed_crossings is None or (source.segment_id, destination.segment_id) in allowed_crossings)
    }
    first_junction = junction_by_segments[source_segment_id, destination_segment_id]
    if allow_larger_cycles and not _can_supply_cycle_move(
        state,
        source_segment_id,
        destination_segment_id,
        moves,
        requested_ion=requested_ion,
    ):
        return None, 0
    # Track used junctions with each BFS path so that one cycle never reuses a junction.
    queue: deque[tuple[tuple[str, ...], frozenset[str]]] = deque([
        ((destination_segment_id,), frozenset({first_junction}))
    ])
    explored = 0
    while queue and explored < max_paths:
        path, used_junctions = queue.popleft()
        explored += 1
        for neighbor in adjacency[path[-1]]:
            junction_id = junction_by_segments[path[-1], neighbor]
            if junction_id in used_junctions:
                continue
            # Larger-cycle search discards a partial path as soon as one edge cannot move an ion.
            if allow_larger_cycles and not _can_supply_cycle_move(state, path[-1], neighbor, moves):
                continue
            if neighbor == source_segment_id:
                if len(path) >= 2:
                    # Lower each closed segment path immediately and return the first legal new placement.
                    candidate_actions = _build_cycle_transport(
                        state,
                        requested_ion,
                        (source_segment_id, *path),
                        moves,
                        ion_priorities=ion_priorities,
                    )
                    if candidate_actions is not None:
                        timed_actions = tuple(
                            ScheduledAction(index, action, state.time, architecture.action_duration(action))
                            for index, action in enumerate(candidate_actions)
                        )
                        try:
                            following = architecture.apply_layer(state, timed_actions)
                        except ValueError:
                            pass
                        else:
                            if following.occupancy not in excluded_occupancies:
                                return candidate_actions, explored
                    if not allow_larger_cycles:
                        return None, explored
                continue
            if neighbor in path or neighbor == destination_segment_id:
                continue
            queue.append(((*path, neighbor), used_junctions | {junction_id}))
    return None, explored


def _can_supply_cycle_move(
    state: GridMachineState,
    source_segment_id: str,
    destination_segment_id: str,
    moves: Sequence[JunctionMove],
    *,
    requested_ion: int | None = None,
) -> bool:
    """Return whether one ion can leave an occupied cycle segment along an edge."""
    if not state.occupants(source_segment_id):
        return True
    return any(
        len(move.ions) == 1
        and move.source.segment_id == source_segment_id
        and move.destination.segment_id == destination_segment_id
        and (requested_ion is None or move.ions[0] == requested_ion)
        for move in moves
    )


def _build_cycle_transport(
    state: GridMachineState,
    requested_ion: int,
    cycle_segments: tuple[str, ...],
    moves: Sequence[JunctionMove],
    *,
    ion_priorities: Mapping[int, int],
) -> tuple[Action, ...] | None:
    """Build simultaneous transport actions along a segment cycle.

    Returns:
        A full cycle or sparse move layer, or ``None`` if lowering fails.
    """
    cycle_moves: list[JunctionMove] = []
    crossings = pairwise((*cycle_segments, cycle_segments[0]))
    for index, (source_segment_id, destination_segment_id) in enumerate(crossings):
        if not state.occupants(source_segment_id):
            continue
        candidates = tuple(
            move
            for move in moves
            if len(move.ions) == 1
            and move.source.segment_id == source_segment_id
            and move.destination.segment_id == destination_segment_id
        )
        if index == 0:
            candidates = tuple(move for move in candidates if move.ions[0] == requested_ion)
        if not candidates:
            return None
        cycle_moves.append(_least_important_move(candidates, ion_priorities))
    if len(cycle_moves) < 2:
        return None
    if len(cycle_moves) == len(cycle_segments):
        try:
            return (Cycle(tuple(cycle_moves)),)
        except ValueError:
            return None
    return tuple(cycle_moves)


def _moves_to_clear_path(
    state: GridMachineState,
    requested_ion: int,
    segment_path: Sequence[str],
    moves: Sequence[JunctionMove],
    *,
    ion_priorities: Mapping[int, int],
) -> tuple[JunctionMove, ...] | None:
    """Select one current-state move for each step of a path to free capacity.

    Returns:
        The selected moves, or ``None`` when the path cannot move as one layer.
    """
    path_clearing_moves: list[JunctionMove] = []
    for index, (source_segment_id, destination_segment_id) in enumerate(pairwise(segment_path)):
        candidates = tuple(
            move
            for move in moves
            if move.source.segment_id == source_segment_id and move.destination.segment_id == destination_segment_id
        )
        if index == 0:
            candidates = tuple(move for move in candidates if requested_ion in move.ions)
        if not candidates:
            return None
        path_clearing_moves.append(_least_important_move(candidates, ion_priorities))
    if any(not set(move.ions).issubset(state.occupants(move.source.segment_id)) for move in path_clearing_moves):
        return None
    return tuple(path_clearing_moves)


def _least_important_move(
    candidates: Sequence[JunctionMove],
    ion_priorities: Mapping[int, int],
) -> JunctionMove:
    """Choose the shortest chain containing the least important ions.

    Returns:
        The selected junction move.
    """
    missing_priority = len(ion_priorities)
    return max(
        candidates,
        key=lambda move: (
            -len(move.ions),
            min(ion_priorities.get(ion, missing_priority) for ion in move.ions),
            tuple(-ion for ion in move.ions),
        ),
    )


def _legal_path_clearing_moves(
    architecture: GridArchitecture,
    state: GridMachineState,
    moves: Sequence[JunctionMove],
) -> tuple[JunctionMove, ...]:
    """Return the longest legal move suffix starting at the free end.

    Returns:
        A jointly executable suffix in source-to-destination order.
    """
    legal_moves: list[JunctionMove] = []
    for move in reversed(moves):
        candidate_moves = (move, *legal_moves)
        timed_moves = tuple(
            ScheduledAction(index, action, state.time, architecture.action_duration(action))
            for index, action in enumerate(candidate_moves)
        )
        try:
            architecture.apply_layer(state, timed_moves)
        except ValueError:
            break
        legal_moves.insert(0, move)
    return tuple(legal_moves)


def select_greedy_target(
    architecture: GridArchitecture,
    state: GridMachineState,
    gate: GateAction,
    target_segment_ids: Sequence[str],
    allowed_crossings: JunctionCrossings = None,
) -> str:
    """Return the reachable processing-zone segment nearest to all gate ions.

    Raises:
        ValueError: If no candidate processing-zone segment is reachable.
    """
    ranked: list[tuple[int, int, str]] = []
    for order, target_segment_id in enumerate(target_segment_ids):
        distances = _segment_distances(architecture, target_segment_id, allowed_crossings, reverse=True)
        if all(state.ion_segment(ion) in distances for ion in gate.ions):
            ranked.append((sum(distances[state.ion_segment(ion)] for ion in gate.ions), order, target_segment_id))
    if not ranked:
        msg = "no processing-zone segment is reachable for the selected gate"
        raise ValueError(msg)
    return min(ranked)[2]


def movement_candidates(
    architecture: GridArchitecture,
    state: GridMachineState,
    *,
    max_cycle_work: int,
    allowed_crossings: JunctionCrossings = None,
) -> tuple[tuple[Action, ...], int]:
    """Return deterministic moves and the work used to find simultaneous rotations."""
    moves = junction_moves(architecture, state, allowed_crossings=allowed_crossings)
    cycles, explored = _cycle_candidates(architecture, state, moves, max_work=max_cycle_work)
    return (*moves, *cycles), explored


def junction_moves(
    architecture: GridArchitecture,
    state: GridMachineState,
    *,
    allowed_crossings: JunctionCrossings = None,
) -> tuple[JunctionMove, ...]:
    """Return deterministic single-junction moves from the current occupancy."""
    moves: list[JunctionMove] = []
    for junction in architecture.junctions:
        for source in junction.endpoints:
            chains = _departure_chains(architecture, state, source.segment_id, source.orientation)
            for destination in junction.endpoints:
                crossing = (source.segment_id, destination.segment_id)
                if source.segment_id == destination.segment_id or (
                    allowed_crossings is not None and crossing not in allowed_crossings
                ):
                    continue
                moves.extend(JunctionMove(source, destination, chain) for chain in chains)
    return tuple(moves)


def _departure_chains(
    architecture: GridArchitecture,
    state: GridMachineState,
    segment_id: str,
    orientation: str,
) -> tuple[tuple[int, ...], ...]:
    """Return all ion chains that can leave a segment through one endpoint."""
    occupants = state.occupants(segment_id)
    if not occupants:
        return ()
    segment = architecture.segment(segment_id)
    if segment.occupancy is SegmentOccupancy.UNORDERED:
        return tuple(chain for length in range(1, len(occupants) + 1) for chain in combinations(occupants, length))
    if orientation == "start":
        return tuple(occupants[:length] for length in range(1, len(occupants) + 1))
    return tuple(occupants[len(occupants) - length :] for length in range(1, len(occupants) + 1))


def _cycle_candidates(
    architecture: GridArchitecture,
    state: GridMachineState,
    moves: Sequence[JunctionMove],
    *,
    max_work: int,
) -> tuple[tuple[Cycle, ...], int]:
    """Return legal simultaneous rotations and the number of examined moves."""
    # Restrict discovery to one-ion moves to bound the candidate combinations.
    # Cycle validation itself supports mixed chain sizes.
    one_ion_moves = tuple(move for move in moves if len(move.ions) == 1)
    by_source: dict[str, list[JunctionMove]] = {}
    for move in one_ion_moves:
        by_source.setdefault(move.source.segment_id, []).append(move)
    cycles: dict[tuple[tuple[str, str, str, str], ...], Cycle] = {}
    explored = 0
    for start in sorted(by_source):
        if explored >= max_work:
            break
        explored += _extend_cycles(
            architecture,
            state,
            start,
            start,
            by_source,
            (),
            frozenset(),
            frozenset(),
            cycles,
            max_work=max_work - explored,
        )
    return tuple(cycles[key] for key in sorted(cycles)), explored


def _extend_cycles(
    architecture: GridArchitecture,
    state: GridMachineState,
    start: str,
    current: str,
    by_source: dict[str, list[JunctionMove]],
    path: tuple[JunctionMove, ...],
    used_segments: frozenset[str],
    used_junctions: frozenset[str],
    cycles: dict[tuple[tuple[str, str, str, str], ...], Cycle],
    *,
    max_work: int,
) -> int:
    """Explore bounded simple move paths and record cycles.

    Returns:
        The number of examined moves.
    """
    if len(path) >= len(architecture.segments) or max_work <= 0:
        return 0
    explored = 0
    for move in by_source.get(current, []):
        if explored >= max_work:
            break
        explored += 1
        junction_id = architecture.junction_for(move.source).junction_id
        destination = move.destination.segment_id
        if junction_id in used_junctions or move.ions[0] in {prior_move.ions[0] for prior_move in path}:
            continue
        extended = (*path, move)
        if destination == start:
            if len(extended) < 2:
                continue
            try:
                cycle = Cycle(extended)
                timed_cycle = ScheduledAction(0, cycle, state.time, architecture.action_duration(cycle))
                architecture.apply_layer(state, (timed_cycle,))
            except ValueError:
                continue
            key = _canonical_cycle_key(extended)
            cycles.setdefault(key, cycle)
            continue
        if destination in used_segments or destination == current:
            continue
        explored += _extend_cycles(
            architecture,
            state,
            start,
            destination,
            by_source,
            extended,
            used_segments | {current},
            used_junctions | {junction_id},
            cycles,
            max_work=max_work - explored,
        )
    return explored


def _canonical_cycle_key(moves: Sequence[JunctionMove]) -> tuple[tuple[str, str, str, str], ...]:
    """Return a rotation-independent identity for a cycle's endpoint sequence."""
    parts = tuple(
        (
            move.source.segment_id,
            move.source.orientation,
            move.destination.segment_id,
            move.destination.orientation,
        )
        for move in moves
    )
    rotations = tuple((*parts[index:], *parts[:index]) for index in range(len(parts)))
    return min(rotations)


def _gate_is_placed(state: GridMachineState, gate: GateAction, target_segment_ids: frozenset[str]) -> bool:
    """Return whether all gate ions share a target processing-zone segment."""
    return (
        bool(gate.ions)
        and all(state.ion_segment(ion) in target_segment_ids for ion in gate.ions)
        and len({state.ion_segment(ion) for ion in gate.ions}) == 1
    )


def _segment_distances(
    architecture: GridArchitecture,
    start_segment_id: str,
    allowed_crossings: JunctionCrossings = None,
    *,
    reverse: bool = False,
) -> dict[str, int]:
    """Return unweighted topology distances from one segment."""
    cache_key = (start_segment_id, allowed_crossings, reverse)
    # Routing alone manages this architecture-local cache.
    distance_cache = architecture._routing_distances  # ruff: ignore[private-member-access]
    if cached := distance_cache.get(cache_key):
        return cached
    adjacency = _segment_adjacency(architecture, allowed_crossings, reverse=reverse)
    distances = {start_segment_id: 0}
    queue = deque((start_segment_id,))
    while queue:
        segment_id = queue.popleft()
        for neighbor in adjacency[segment_id]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[segment_id] + 1
            queue.append(neighbor)
    distance_cache[cache_key] = distances
    return distances


def _shortest_segment_path(
    adjacency: dict[str, tuple[str, ...]],
    source_segment_id: str,
    target_segment_id: str,
) -> tuple[str, ...] | None:
    """Return one deterministic shortest segment path."""
    queue: deque[tuple[str, ...]] = deque([(source_segment_id,)])
    visited = {source_segment_id}
    while queue:
        path = queue.popleft()
        if path[-1] == target_segment_id:
            return path
        for neighbor in adjacency[path[-1]]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((*path, neighbor))
    return None


def _segment_adjacency(
    architecture: GridArchitecture,
    allowed_crossings: JunctionCrossings = None,
    *,
    reverse: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Return deterministic segment adjacency induced by junction membership."""
    cache_key = (allowed_crossings, reverse)
    # Routing alone manages this architecture-local cache.
    adjacency_cache = architecture._routing_adjacencies  # ruff: ignore[private-member-access]
    if cached := adjacency_cache.get(cache_key):
        return cached
    adjacency: dict[str, set[str]] = {segment.segment_id: set() for segment in architecture.segments}
    for junction in architecture.junctions:
        segment_ids = {endpoint.segment_id for endpoint in junction.endpoints}
        for segment_id in segment_ids:
            for neighbor in segment_ids - {segment_id}:
                crossing = (neighbor, segment_id) if reverse else (segment_id, neighbor)
                if allowed_crossings is None or crossing in allowed_crossings:
                    adjacency[segment_id].add(neighbor)
    result = {segment_id: tuple(sorted(neighbors)) for segment_id, neighbors in adjacency.items()}
    adjacency_cache[cache_key] = result
    return result


def _occupancy_key(state: GridMachineState) -> OccupancyKey:
    """Return the position-only identity used to detect repeated search states."""
    return state.occupancy


__all__ = [
    "clear_blocked_path",
    "junction_moves",
    "movement_candidates",
    "next_segments_toward",
    "route_gate_bfs",
    "route_junction_crossing_greedy_cycle",
    "route_placement_bfs",
    "routing_distance",
    "select_greedy_target",
]
