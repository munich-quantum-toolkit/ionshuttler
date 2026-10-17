# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Grid layout adapters."""

from __future__ import annotations

import networkx as nx

from mqt.ionshuttler.core.gates import Rzz
from mqt.ionshuttler.core.schedule import ScheduledAction
from mqt.ionshuttler.grid import SegmentOccupancy
from mqt.ionshuttler.grid.layouts import from_networkx, hexagonal_grid, rectangular_grid, square_grid


def test_square_grid_uses_physical_edges_as_compiler_segments() -> None:
    """Construct deterministic locations and junction turns for a square layout."""
    architecture = square_grid(3, processing_zone_segments=("h:0:0",))

    assert len(architecture.junctions) == 9
    assert len(architecture.segments) == 12
    assert architecture.segment("h:0:0").start.segment_id == "h:0:0"
    assert architecture.processing_zone_segment("pz:h:0:0").segment_id == "h:0:0"
    assert sum(len(junction.endpoints) for junction in architecture.junctions) == 24


def test_rectangular_grid_supports_relaxed_capacity_buckets() -> None:
    """Represent legacy parking capacity without changing the default ordered model."""
    architecture = rectangular_grid(
        2,
        3,
        segment_capacity=2,
        unordered_segments=("h:0:0",),
    )

    assert architecture.segment("h:0:0").capacity == 2
    assert architecture.segment("h:0:0").occupancy is SegmentOccupancy.UNORDERED
    assert architecture.segment("h:0:1").occupancy is SegmentOccupancy.ORDERED


def test_square_grid_reconstructs_the_inside_four_zone_fixture() -> None:
    """Represent the embedded four-zone device, whose segments hold two ions."""
    zone_segments = ("h:0:1", "v:1:0", "h:3:1", "v:1:3")
    architecture = square_grid(4, processing_zone_segments=zone_segments, segment_capacity=2)
    state = architecture.initial_state({
        "v:0:0": (0,),
        "h:0:0": (1,),
        "v:0:1": (2,),
        "v:0:2": (3,),
        "h:0:2": (4,),
        "v:0:3": (5,),
    })
    gate_state = architecture.initial_state({"h:0:1": (0, 1)})

    after_gate = architecture.apply_layer(gate_state, (ScheduledAction(0, Rzz(0, 1, 0.5), 0, 2, "pz:h:0:1"),))

    assert {
        architecture.processing_zone_segment(zone.zone_id).segment_id for zone in architecture.processing_zones
    } == set(zone_segments)
    assert all(segment.capacity == 2 for segment in architecture.segments)
    assert state.ions == (0, 1, 2, 3, 4, 5)
    assert len(architecture.segments) == 24
    assert dict(after_gate.pzs_busy_until)["pz:h:0:1"] == 2


def test_networkx_adapter_uses_stable_ids_and_ignores_editor_metadata() -> None:
    """Separate compiler topology from coordinates, colors, and labels."""
    graph = nx.Graph()
    graph.add_node("left", position=(10.0, 20.0), label="Left")
    graph.add_node("center", position=(30.0, 20.0), label="Center")
    graph.add_node("right", position=(50.0, 20.0), label="Right")
    graph.add_edge("left", "center", segment_id="a", capacity=2, occupancy="unordered")
    graph.add_edge("center", "right", segment_id="b")

    architecture = from_networkx(graph, processing_zone_segments=("b",))

    assert architecture.segment("a").capacity == 2
    assert architecture.segment("a").occupancy is SegmentOccupancy.UNORDERED
    assert architecture.processing_zone_segment("pz:b").segment_id == "b"
    assert any(
        endpoint.segment_id == "a"
        for endpoint in next(j for j in architecture.junctions if j.junction_id == "center").endpoints
    )


def test_hexagonal_grid_constructs_one_honeycomb_cell() -> None:
    """Construct a six-segment cell without cardinal segment identities."""
    zone_segment = "s:j:0:0--j:0:1"
    architecture = hexagonal_grid(2, 3, processing_zone_segments=(zone_segment,))

    assert len(architecture.junctions) == 6
    assert len(architecture.segments) == 6
    assert sum(len(junction.endpoints) for junction in architecture.junctions) == 12
    assert all(segment.segment_id.startswith("s:j:") for segment in architecture.segments)
    assert architecture.processing_zone_segment(f"pz:{zone_segment}").segment_id == zone_segment
