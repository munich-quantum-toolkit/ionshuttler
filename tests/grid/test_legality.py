# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare Grid movement validation with independent rules on every small layer."""

from __future__ import annotations

from itertools import combinations, permutations, product

import pytest

from mqt.ionshuttler.core.schedule import ScheduledAction
from mqt.ionshuttler.grid import (
    Cycle,
    GridArchitecture,
    GridMachineState,
    Junction,
    JunctionMove,
    ProcessingZone,
    Segment,
    SegmentOccupancy,
)

from .legality_oracle import is_single_rotation, layer_outcome

ION_COUNT = 3
MAX_LAYER_SIZE = 3


def _ring() -> GridArchitecture:
    """Join three segments end to start, end to end, and start to start."""
    a = Segment("a", capacity=2)
    b = Segment("b", capacity=2)
    c = Segment("c")
    return GridArchitecture(
        (a, b, c),
        (
            Junction("ab", (a.end, b.start)),
            Junction("bc", (b.end, c.end)),
            Junction("ca", (c.start, a.start)),
        ),
    )


def _hub() -> GridArchitecture:
    """Join three dead-end segments at one junction and attach an unordered bucket."""
    x = Segment("x")
    y = Segment("y")
    z = Segment("z", capacity=2)
    p = Segment("p", capacity=2, occupancy=SegmentOccupancy.UNORDERED)
    return GridArchitecture(
        (x, y, z, p),
        (
            Junction("hub", (x.end, y.end, z.start)),
            Junction("dock", (z.end, p.start)),
        ),
    )


def _attached_zone() -> GridArchitecture:
    """Route from memory through exit, parking, and entry back to memory."""
    memory = Segment("memory")
    exit_path = Segment("exit")
    parking = Segment(
        "parking",
        capacity=2,
        occupancy=SegmentOccupancy.UNORDERED,
        processing_zones=(ProcessingZone("pz"),),
    )
    entry_path = Segment("entry")
    return GridArchitecture(
        (memory, exit_path, parking, entry_path),
        (
            Junction("exit-node", (memory.end, exit_path.start)),
            Junction("zone-node", (exit_path.end, parking.start, entry_path.start)),
            Junction("entry-node", (entry_path.end, memory.start)),
        ),
    )


ARCHITECTURES = {"ring": _ring(), "hub": _hub(), "attached-zone": _attached_zone()}


def _placements(architecture: GridArchitecture) -> list[dict[str, tuple[int, ...]]]:
    """Return every placement of the test ions within capacity and in every ordered arrangement."""
    segment_ids = [segment.segment_id for segment in architecture.segments]
    placements: list[dict[str, tuple[int, ...]]] = []
    for assignment in product(segment_ids, repeat=ION_COUNT):
        groups = [
            tuple(ion for ion, target in enumerate(assignment) if target == segment.segment_id)
            for segment in architecture.segments
        ]
        if any(len(ions) > segment.capacity for ions, segment in zip(groups, architecture.segments, strict=True)):
            continue
        arrangements = [
            list(permutations(ions)) if segment.occupancy is SegmentOccupancy.ORDERED else [ions]
            for ions, segment in zip(groups, architecture.segments, strict=True)
        ]
        placements.extend(dict(zip(segment_ids, arrangement, strict=True)) for arrangement in product(*arrangements))
    return placements


def _candidate_moves(architecture: GridArchitecture, occupancy: dict[str, tuple[int, ...]]) -> list[JunctionMove]:
    """Return every chain order from each traversal direction plus one absent ion."""
    all_ions = sorted(ion for ions in occupancy.values() for ion in ions)
    moves: list[JunctionMove] = []
    for junction in architecture.junctions:
        for source in junction.endpoints:
            for destination in junction.endpoints:
                if source == destination:
                    continue
                occupants = occupancy[source.segment_id]
                chains = [chain for size in range(1, len(occupants) + 1) for chain in permutations(occupants, size)]
                absent = [ion for ion in all_ions if ion not in occupants]
                if absent:
                    chains.append((absent[0],))
                moves.extend(JunctionMove(source, destination, chain) for chain in chains)
    return moves


def _applied_occupancy(
    architecture: GridArchitecture,
    state: GridMachineState,
    actions: tuple[JunctionMove | Cycle, ...],
) -> dict[str, tuple[int, ...]] | None:
    scheduled = tuple(ScheduledAction(index, action, 0, 1) for index, action in enumerate(actions))
    try:
        final = architecture.apply_layer(state, scheduled)
    except ValueError:
        return None
    return dict(final.occupancy)


CASES = [
    pytest.param(name, placement, id=f"{name}-{index}")
    for name, architecture in ARCHITECTURES.items()
    for index, placement in enumerate(_placements(architecture))
]


@pytest.mark.parametrize(("name", "placement"), CASES)
def test_layer_validation_matches_independent_rules(name: str, placement: dict[str, tuple[int, ...]]) -> None:
    """Accept exactly the layers the independent rules accept, with the same final occupancy."""
    architecture = ARCHITECTURES[name]
    state = architecture.initial_state(placement)
    occupancy = dict(state.occupancy)
    candidates = _candidate_moves(architecture, occupancy)
    mismatches: list[tuple[JunctionMove, ...]] = []
    for size in range(1, MAX_LAYER_SIZE + 1):
        for layer in combinations(candidates, size):
            expected = layer_outcome(architecture, occupancy, layer)
            if _applied_occupancy(architecture, state, layer) != expected:
                mismatches.append(layer)
            if expected is not None and size > 1:
                as_cycle = expected if is_single_rotation(architecture, layer) else None
                if _applied_occupancy(architecture, state, (Cycle(layer),)) != as_cycle:
                    mismatches.append(layer)

    assert not mismatches, mismatches[:5]


def test_enumeration_covers_legal_and_illegal_rotations() -> None:
    """Include a capacity-full rotation and its forbidden reverse direction."""
    architecture = ARCHITECTURES["ring"]
    occupancy = {"a": (0, 1), "b": (2,), "c": (3,)}
    forward = (
        JunctionMove(Segment("a").end, Segment("b").start, (1,)),
        JunctionMove(Segment("b").end, Segment("c").end, (2,)),
        JunctionMove(Segment("c").start, Segment("a").start, (3,)),
    )
    backward = (
        JunctionMove(Segment("b").start, Segment("a").end, (2,)),
        JunctionMove(Segment("c").end, Segment("b").end, (3,)),
        JunctionMove(Segment("a").start, Segment("c").start, (0,)),
    )

    assert is_single_rotation(architecture, forward)
    assert layer_outcome(architecture, occupancy, forward) == {"a": (3, 0), "b": (1,), "c": (2,)}
    assert layer_outcome(architecture, occupancy, backward) == {"a": (1, 2), "b": (3,), "c": (0,)}


def test_attached_zone_admits_one_parking_transfer_per_tick() -> None:
    """Allow parking or unparking in one tick, but not both through the shared zone junction."""
    architecture = ARCHITECTURES["attached-zone"]
    state = architecture.initial_state({"exit": (0,), "parking": (1,)})
    full = architecture.initial_state({"exit": (0,), "parking": (1, 2)})
    park = JunctionMove(Segment("exit").end, Segment("parking").start, (0,))
    unpark = JunctionMove(Segment("parking").start, Segment("entry").start, (1,))

    parked = _applied_occupancy(architecture, state, (park,))
    unparked = _applied_occupancy(architecture, state, (unpark,))

    assert parked == {"memory": (), "exit": (), "parking": (0, 1), "entry": ()}
    assert unparked == {"memory": (), "exit": (0,), "parking": (), "entry": (1,)}
    assert _applied_occupancy(architecture, state, (park, unpark)) is None
    assert _applied_occupancy(architecture, full, (park,)) is None
    reverse_state = architecture.initial_state({"parking": (1,)})
    assert _applied_occupancy(
        architecture,
        reverse_state,
        (JunctionMove(Segment("parking").start, Segment("exit").end, (1,)),),
    ) == {"memory": (), "exit": (1,), "parking": (), "entry": ()}
