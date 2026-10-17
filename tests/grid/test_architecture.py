# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Grid architecture validation and simultaneous state changes."""

from __future__ import annotations

import json
import pickle  # ruff: ignore[suspicious-pickle-import] - The tests unpickle only data they create.
from typing import TYPE_CHECKING, Literal

import pytest

from mqt.ionshuttler.core.gates import Rx, Rz, Rzz
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.grid import (
    Cycle,
    GridArchitecture,
    Junction,
    JunctionMove,
    ProcessingZone,
    Segment,
    SegmentEndpoint,
    SegmentOccupancy,
)
from mqt.ionshuttler.grid.schedule import schedule_from_json

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.grid import GridMachineState


def _cycle_architecture() -> GridArchitecture:
    a = Segment("a")
    b = Segment("b")
    c = Segment("c")
    return GridArchitecture(
        segments=(a, b, c),
        junctions=(
            Junction("ab", (a.end, b.start)),
            Junction("bc", (b.end, c.start)),
            Junction("ca", (c.end, a.start)),
        ),
    )


def test_initial_placement_validates_capacity_identity_and_order() -> None:
    """Create one authoritative occupancy map from user placement."""
    architecture = GridArchitecture(
        segments=(Segment("ordered", capacity=2), Segment("bucket", 2, SegmentOccupancy.UNORDERED)),
        junctions=(),
    )

    state = architecture.initial_state({"ordered": (2, 1), "bucket": (4, 3)})

    assert state.occupants("ordered") == (2, 1)
    assert state.occupants("bucket") == (3, 4)
    with pytest.raises(ValueError, match="exceeds capacity"):
        architecture.initial_state({"ordered": (0, 1, 2)})
    with pytest.raises(ValueError, match="exactly once"):
        architecture.initial_state({"ordered": (0,), "bucket": (0,)})
    with pytest.raises(ValueError, match="unknown segments"):
        architecture.initial_state({"missing": (0,)})


def test_simultaneous_cycle_can_rotate_full_segments() -> None:
    """Accept a legal rotation whose destinations are occupied before the layer."""
    architecture = _cycle_architecture()
    state = architecture.initial_state({"a": (0,), "b": (1,), "c": (2,)})
    cycle = Cycle((
        JunctionMove(Segment("a").end, Segment("b").start, (0,)),
        JunctionMove(Segment("b").end, Segment("c").start, (1,)),
        JunctionMove(Segment("c").end, Segment("a").start, (2,)),
    ))

    final = architecture.apply_layer(state, (ScheduledAction(0, cycle, 0, 1),))

    assert dict(final.occupancy) == {"a": (2,), "b": (0,), "c": (1,)}


def test_ordered_move_requires_boundary_chain_and_maps_orientation() -> None:
    """Use segment endpoints to validate and transform ordered ion chains."""
    source = Segment("source", capacity=3)
    destination = Segment("destination", capacity=3)
    architecture = GridArchitecture(
        (source, destination),
        (Junction("turn", (source.start, destination.start)),),
    )
    state = architecture.initial_state({"source": (0, 1), "destination": ()})

    with pytest.raises(ValueError, match="ordered chain"):
        architecture.apply_layer(
            state, (ScheduledAction(0, JunctionMove(source.start, destination.start, (1,)), 0, 1),)
        )

    final = architecture.apply_layer(
        state, (ScheduledAction(0, JunctionMove(source.start, destination.start, (0, 1)), 0, 1),)
    )
    assert final.occupants("destination") == (1, 0)


@pytest.mark.parametrize(
    ("departure", "arrival", "expected"),
    [
        ("end", "start", (0, 1, 2)),
        ("end", "end", (2, 1, 0)),
        ("start", "start", (1, 0, 2)),
        ("start", "end", (2, 0, 1)),
    ],
)
def test_moved_chain_keeps_its_travel_order(
    departure: Literal["start", "end"],
    arrival: Literal["start", "end"],
    expected: tuple[int, ...],
) -> None:
    """Place the leading ion of a moved chain farthest from the arrival end."""
    source = Segment("source", capacity=2)
    destination = Segment("destination", capacity=3)
    architecture = GridArchitecture(
        (source, destination),
        (Junction("j", (SegmentEndpoint("source", departure), SegmentEndpoint("destination", arrival))),),
    )
    state = architecture.initial_state({"source": (0, 1), "destination": (2,)})

    final = architecture.apply_layer(
        state,
        (
            ScheduledAction(
                0,
                JunctionMove(SegmentEndpoint("source", departure), SegmentEndpoint("destination", arrival), (0, 1)),
                0,
                1,
            ),
        ),
    )

    assert final.occupants("destination") == expected


def test_layer_rejects_unknown_endpoint_and_zone_references() -> None:
    """Report schedule references to absent hardware as invalid schedules."""
    architecture = GridArchitecture(
        (Segment("control", capacity=2, processing_zones=(ProcessingZone("pz"),)),),
        (),
    )
    state = architecture.initial_state({"control": (0, 1)})
    unknown_endpoint: Schedule[Action, GridMachineState] = Schedule(
        (ScheduledAction(0, JunctionMove(Segment("missing").end, Segment("other").start, (0,)), 0, 1),), 1, state
    )
    unknown_zone: Schedule[Action, GridMachineState] = Schedule(
        (ScheduledAction(0, Rzz(0, 1, 0.5), 0, 2, "missing"),), 2, state
    )

    for schedule, message in (
        (unknown_endpoint, "endpoints must belong to the same junction"),
        (unknown_zone, "unknown processing zone"),
    ):
        assert not architecture.is_schedule_valid(schedule)
        with pytest.raises(ValueError, match=message):
            architecture.replay_schedule(schedule)


def test_cycle_must_form_one_closed_rotation() -> None:
    """Reject open chains and two separate loops grouped as one cycle."""
    a, b, c = (Segment(segment_id) for segment_id in "abc")
    open_architecture = GridArchitecture(
        (a, b, c),
        (
            Junction("ab", (a.end, b.start)),
            Junction("bc", (b.end, c.start)),
        ),
    )
    open_state = open_architecture.initial_state({"a": (0,), "b": (1,), "c": (2,)})
    open_chain = Cycle((JunctionMove(a.end, b.start, (0,)), JunctionMove(b.end, c.start, (1,))))

    with pytest.raises(ValueError, match="closed segment rotation"):
        open_architecture.apply_layer(open_state, (ScheduledAction(0, open_chain, 0, 1),))

    a, b, c, d = (Segment(segment_id) for segment_id in "abcd")
    architecture = GridArchitecture(
        (a, b, c, d),
        (
            Junction("a-b", (a.end, b.start)),
            Junction("b-a", (b.end, a.start)),
            Junction("c-d", (c.end, d.start)),
            Junction("d-c", (d.end, c.start)),
        ),
    )
    state = architecture.initial_state({"a": (0,), "b": (1,), "c": (2,), "d": (3,)})
    two_loops = Cycle((
        JunctionMove(a.end, b.start, (0,)),
        JunctionMove(b.end, a.start, (1,)),
        JunctionMove(c.end, d.start, (2,)),
        JunctionMove(d.end, c.start, (3,)),
    ))

    with pytest.raises(ValueError, match="one closed segment rotation"):
        architecture.apply_layer(state, (ScheduledAction(0, two_loops, 0, 1),))
    swapped = architecture.apply_layer(state, (ScheduledAction(0, Cycle(two_loops.moves[:2]), 0, 1),))
    assert dict(swapped.occupancy) == {"a": (1,), "b": (0,), "c": (2,), "d": (3,)}


def test_architecture_pickles_with_its_lookup_tables() -> None:
    """Keep identity lookups usable after a process boundary."""
    architecture = _cycle_architecture()

    restored = pickle.loads(pickle.dumps(architecture))  # ruff: ignore[suspicious-pickle-usage] - Test-created data.

    assert restored == architecture
    assert restored.junction_for(Segment("a").end) == architecture.junction_for(Segment("a").end)


def test_action_layer_rejects_shared_junction_and_capacity_conflicts() -> None:
    """Reject independently valid moves that claim one junction or overfill a segment."""
    a = Segment("a")
    b = Segment("b")
    c = Segment("c", capacity=2)
    architecture = GridArchitecture(
        (a, b, c),
        (
            Junction("shared", (a.end, b.end, c.start)),
            Junction("other", (b.start, c.end)),
        ),
    )
    state = architecture.initial_state({"a": (0,), "b": (1,), "c": (2,)})

    with pytest.raises(ValueError, match="busy or claimed"):
        architecture.apply_layer(
            state,
            (
                ScheduledAction(0, JunctionMove(a.end, c.start, (0,)), 0, 1),
                ScheduledAction(1, JunctionMove(b.end, c.start, (1,)), 0, 1),
            ),
        )
    with pytest.raises(ValueError, match="exceeds capacity"):
        architecture.apply_layer(
            state,
            (
                ScheduledAction(0, JunctionMove(a.end, c.start, (0,)), 0, 1),
                ScheduledAction(1, JunctionMove(b.start, c.end, (1,)), 0, 1),
            ),
        )


def test_gate_execution_requires_selected_local_processing_zone() -> None:
    """Validate gate colocation, zone support, and half-open reservations."""
    pz_segment = Segment("control", capacity=3, processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        segments=(pz_segment, Segment("storage")),
        junctions=(),
    )
    state = architecture.initial_state({"control": (0, 1), "storage": (2,)})
    gate = Rzz(ion_a=0, ion_b=1, theta=0.5, gate_id=0)

    with pytest.raises(ValueError, match="must select"):
        architecture.apply_layer(state, (ScheduledAction(0, gate, 0, 2),))
    occupied = architecture.apply_layer(state, (ScheduledAction(0, gate, 0, 2, "pz"),))
    with pytest.raises(ValueError, match="busy or claimed"):
        architecture.apply_layer(
            occupied,
            (ScheduledAction(1, Rzz(ion_a=0, ion_b=1, theta=0.2, gate_id=1), 1, 2, "pz"),),
        )
    released = architecture.apply_layer(
        occupied,
        (ScheduledAction(1, Rzz(ion_a=0, ion_b=1, theta=0.2, gate_id=1), 2, 2, "pz"),),
    )
    assert dict(released.pzs_busy_until)["pz"] == 4


def test_ordered_processing_zones_on_one_segment_are_independent_resources() -> None:
    """Preserve zone order and permit disjoint gates on the same segment."""
    control = Segment(
        "control",
        capacity=2,
        processing_zones=(ProcessingZone("near-start"), ProcessingZone("near-end")),
    )
    architecture = GridArchitecture((control,), ())
    state = architecture.initial_state({"control": (0, 1)})

    final = architecture.apply_layer(
        state,
        (
            ScheduledAction(0, Rx(0, 0.5), 0, 1, "near-start"),
            ScheduledAction(1, Rx(1, 0.5), 0, 1, "near-end"),
        ),
    )
    restored = GridArchitecture.from_json(architecture.to_json())
    serialized = json.loads(architecture.to_json())

    assert architecture.processing_zones == control.processing_zones
    assert restored.segment("control").processing_zones == control.processing_zones
    assert "processing_zones" not in serialized
    assert [zone["zone_id"] for zone in serialized["segments"][0]["processing_zones"]] == [
        "near-start",
        "near-end",
    ]
    assert dict(final.pzs_busy_until) == {"near-end": 1, "near-start": 1}


def test_virtual_gate_needs_no_processing_zone() -> None:
    """Allow an instantaneous virtual rotation outside processing zones."""
    architecture = GridArchitecture((Segment("storage"),), ())
    state = architecture.initial_state({"storage": (0,)})

    final = architecture.apply_layer(state, (ScheduledAction(0, Rz(ion=0, theta=0.5), 0, 0),))

    assert final == state


def test_architecture_and_schedule_round_trip_and_replay() -> None:
    """Preserve stable identities, actions, and final state through JSON."""
    architecture = _cycle_architecture()
    restored_architecture = GridArchitecture.from_dict(json.loads(architecture.to_json()))
    state = architecture.initial_state({"a": (0,), "b": (), "c": ()})
    schedule = Schedule(
        (ScheduledAction(0, JunctionMove(Segment("a").end, Segment("b").start, (0,)), 0, 1),),
        end_time=1,
        initial_state=state,
    )
    restored_schedule = schedule_from_json(schedule.to_json())

    assert restored_architecture == architecture
    assert restored_schedule == schedule
    assert restored_architecture.is_schedule_valid(restored_schedule)
    assert restored_architecture.replay_schedule(restored_schedule).occupants("b") == (0,)


def test_architecture_save_appends_json_to_existing_suffix(tmp_path: Path) -> None:
    """Keep a supplied filename suffix when adding the JSON extension."""
    output = _cycle_architecture().save(tmp_path / "architecture.backup")

    assert output == tmp_path / "architecture.backup.json"
    assert output.is_file()
