# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for immutable Grid hardware and action values."""

from __future__ import annotations

import json

import pytest

from mqt.ionshuttler.grid import (
    Cycle,
    Junction,
    JunctionMove,
    ProcessingZone,
    Segment,
    SegmentEndpoint,
    SegmentOccupancy,
)
from mqt.ionshuttler.grid.actions import decode_grid_action


def test_segment_endpoints_have_stable_serialized_identity() -> None:
    """Keep compiler identity independent of future drawing geometry."""
    segment = Segment("storage-a", capacity=2, occupancy=SegmentOccupancy.UNORDERED)

    assert segment.start == SegmentEndpoint("storage-a", "start")
    assert segment.end == SegmentEndpoint("storage-a", "end")
    assert segment.end.to_dict() == {"segment_id": "storage-a", "orientation": "end"}
    assert Segment.from_dict(segment.to_dict()) == segment
    assert SegmentEndpoint.from_dict(segment.end.to_dict()) == segment.end


def test_junction_owns_canonical_segment_endpoints() -> None:
    """Describe junction connectivity without pairwise transport edges."""
    junction = Junction("junction", (Segment("b").start, Segment("a").end))

    assert junction.endpoints == (Segment("a").end, Segment("b").start)
    assert Junction.from_dict(junction.to_dict()) == junction


def test_grid_actions_round_trip_as_json() -> None:
    """Serialize nested cycle moves without leaking dataclass objects."""
    a = Segment("a")
    b = Segment("b")
    cycle = Cycle((JunctionMove(a.end, b.start, (0, 1)), JunctionMove(b.start, a.end, (2,))))

    serialized = json.loads(json.dumps(cycle.to_dict()))

    assert decode_grid_action(serialized) == cycle


def test_grid_actions_have_concise_human_readable_text() -> None:
    """Show ion movement without replacing the complete debugging representation."""
    a = Segment("a")
    b = Segment("b")
    c = Segment("c")
    move = JunctionMove(a.end, b.start, (0, 1))
    cycle = Cycle((JunctionMove(a.end, b.start, (0,)), JunctionMove(b.end, c.start, (1,))))

    assert str(move) == "move [q0,q1] a:end → b:start"
    assert str(cycle) == "cycle q0:a→b, q1:b→c"
    assert repr(move).startswith("JunctionMove(")


def test_invalid_model_values_are_rejected() -> None:
    """Reject ambiguous identities, capacities, and cycle operands."""
    with pytest.raises(ValueError, match="non-empty"):
        Segment("")
    with pytest.raises(ValueError, match=">= 1"):
        Segment("a", capacity=0)
    with pytest.raises(ValueError, match="duplicates"):
        Junction("j", (Segment("a").start, Segment("a").start))
    a = Segment("a")
    b = Segment("b")
    with pytest.raises(ValueError, match="at least two"):
        Cycle((JunctionMove(a.end, b.start, (0,)),))
    with pytest.raises(ValueError, match="more than once"):
        Cycle((JunctionMove(a.end, b.start, (0,)), JunctionMove(b.start, a.end, (0,))))


def test_processing_zone_declares_supported_gate_types() -> None:
    """Keep processing zones in spatial order with an explicit gate catalog."""
    first = ProcessingZone("control-a")
    second = ProcessingZone("control-b")
    segment = Segment("storage-a", processing_zones=(first, second))

    assert segment.processing_zones == (first, second)
    assert first.supported_gate_types
    assert Segment.from_dict(segment.to_dict()) == segment
