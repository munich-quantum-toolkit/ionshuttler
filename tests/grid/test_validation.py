# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Boundary and malformed-input tests for the Grid model."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import networkx as nx
import pytest

from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.gates import Rx, Rz, Rzz
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.grid import (
    Cycle,
    GridArchitecture,
    GridMachineState,
    Junction,
    JunctionMove,
    ProcessingZone,
    Segment,
    SegmentEndpoint,
    SegmentOccupancy,
    TransportTiming,
)
from mqt.ionshuttler.grid.layouts import from_networkx, hexagonal_grid, rectangular_grid
from mqt.ionshuttler.grid.schedule import load_schedule

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class _UnsupportedAction(Action):
    """Action outside the Grid implementation catalog."""

    serialized_type: ClassVar[str] = "test.unsupported"


def _line_architecture() -> GridArchitecture:
    a = Segment("a", capacity=2)
    b = Segment("b", capacity=2, processing_zones=(ProcessingZone("pz"),))
    return GridArchitecture(
        (a, b),
        (Junction("j", (a.end, b.start)),),
    )


def _machine_state(**arguments: object) -> GridMachineState:
    return cast("Any", GridMachineState)(**arguments)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        (("a", Segment("b").start, (0,)), TypeError, "SegmentEndpoint"),
        ((Segment("a").end, Segment("a").end, (0,)), ValueError, "different endpoints"),
        ((Segment("a").end, Segment("b").start, ()), ValueError, "ions must be non-empty"),
        ((Segment("a").end, Segment("b").start, (False,)), TypeError, "ions must contain integers"),
        ((Segment("a").end, Segment("b").start, (-1,)), ValueError, "ions must be non-negative"),
        ((Segment("a").end, Segment("b").start, (0, 0)), ValueError, "ions must not contain duplicates"),
    ],
)
def test_junction_move_rejects_malformed_fields(
    arguments: tuple[object, ...], error: type[Exception], message: str
) -> None:
    """Reject transport values that cannot have clear physical meaning."""
    with pytest.raises(error, match=message):
        cast("Any", JunctionMove)(*arguments)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: SegmentEndpoint("a", cast("Any", "middle")), ValueError, "orientation"),
        (lambda: Segment("a", occupancy=cast("Any", "ordered")), TypeError, "SegmentOccupancy"),
        (lambda: Junction(cast("Any", 1), ()), TypeError, "must be a string"),
        (
            lambda: Junction("j", cast("Any", ("a",))),
            TypeError,
            "SegmentEndpoint",
        ),
        (lambda: ProcessingZone("pz", cast("Any", (Segment,))), TypeError, "GateAction"),
        (lambda: ProcessingZone("pz", (Rx, Rx)), ValueError, "duplicates"),
        (
            lambda: Segment("a", processing_zones=cast("Any", ("pz",))),
            TypeError,
            "ProcessingZone values",
        ),
        (lambda: TransportTiming(junction_move=cast("Any", 1.5)), TypeError, "integer"),
    ],
)
def test_hardware_values_reject_invalid_fields(factory: Any, error: type[Exception], message: str) -> None:
    """Reject hardware values that cannot round trip through the typed model."""
    with pytest.raises(error, match=message):
        factory()


def test_architecture_rejects_invalid_references_and_catalogs() -> None:
    """Reject incomplete topology references and inconsistent gate support."""
    a = Segment("a")
    b = Segment("b")
    with pytest.raises(ValueError, match="at least one segment"):
        GridArchitecture((), ())
    with pytest.raises(ValueError, match="unknown segment"):
        GridArchitecture((a,), (Junction("j", (a.end, b.start)),))
    with pytest.raises(TypeError, match="GateTiming"):
        GridArchitecture((a,), (), gate_timing=cast("Any", {}))
    with pytest.raises(TypeError, match="TransportTiming"):
        GridArchitecture((a,), (), transport_timing=cast("Any", {}))
    with pytest.raises(TypeError, match="Action subclasses"):
        GridArchitecture((a,), (), supported_action_types=cast("Any", (object,)))
    with pytest.raises(ValueError, match="duplicates"):
        GridArchitecture((a,), (), supported_action_types=(JunctionMove, JunctionMove))
    with pytest.raises(ValueError, match="implements no action"):
        GridArchitecture((a,), (), supported_action_types=(_UnsupportedAction,))
    with pytest.raises(ValueError, match="declares unsupported gates"):
        GridArchitecture(
            (Segment("a", processing_zones=(ProcessingZone("pz", (Rzz,)),)),),
            (),
            supported_action_types=(JunctionMove,),
        )


def test_architecture_rejects_duplicate_or_wrong_model_values() -> None:
    """Require one value of the right type for every stable identity."""
    a = Segment("a")
    with pytest.raises(ValueError, match="unique identifiers"):
        GridArchitecture((a, a), ())
    with pytest.raises(TypeError, match="Segment values"):
        GridArchitecture(cast("Any", ("a",)), ())
    b = Segment("b")
    with pytest.raises(ValueError, match="belongs to junctions"):
        GridArchitecture(
            (a, b),
            (
                Junction("first", (a.end, b.start)),
                Junction("second", (a.end, b.end)),
            ),
        )
    duplicate_zone = ProcessingZone("pz")
    with pytest.raises(ValueError, match="unique identifiers within a segment"):
        Segment("a", processing_zones=(duplicate_zone, duplicate_zone))
    with pytest.raises(ValueError, match="globally unique"):
        GridArchitecture(
            (
                Segment("a", processing_zones=(duplicate_zone,)),
                Segment("b", processing_zones=(duplicate_zone,)),
            ),
            (),
        )


def test_initial_placement_rejects_malformed_values() -> None:
    """Reject placement values that cannot define a canonical state."""
    architecture = _line_architecture()
    assert architecture.initial_state().ions == ()
    with pytest.raises(TypeError, match="must be a mapping"):
        architecture.initial_state(cast("Any", ()))
    with pytest.raises(TypeError, match="must be a sequence"):
        architecture.initial_state({"a": cast("Any", 1)})
    with pytest.raises(TypeError, match="must contain integers"):
        architecture.initial_state({"a": (False,)})
    with pytest.raises(ValueError, match="non-negative"):
        architecture.initial_state({"a": (-1,)})


def test_layer_rejects_bad_timing_selection_and_topology() -> None:
    """Reject actions whose schedule metadata or endpoint topology is invalid."""
    architecture = _line_architecture()
    state = architecture.initial_state({"a": (0,), "b": (1,)})
    move = JunctionMove(Segment("a").end, Segment("b").start, (0,))
    assert architecture.apply_layer(state, ()) == state
    with pytest.raises(ValueError, match="one start time"):
        architecture.apply_layer(
            state,
            (ScheduledAction(0, move, 0, 1), ScheduledAction(1, Rzz(0, 1, 0.5), 1, 2, "pz")),
        )
    with pytest.raises(ValueError, match="before the machine state"):
        architecture.apply_layer(state.at_time(1), (ScheduledAction(0, move, 0, 1),))
    with pytest.raises(ValueError, match="duration"):
        architecture.apply_layer(state, (ScheduledAction(0, move, 0, 2),))
    with pytest.raises(ValueError, match="must not select"):
        architecture.apply_layer(state, (ScheduledAction(0, move, 0, 1, "pz"),))
    with pytest.raises(ValueError, match="same junction"):
        architecture.apply_layer(
            state,
            (ScheduledAction(0, JunctionMove(Segment("a").start, Segment("b").start, (0,)), 0, 1),),
        )


def test_layer_rejects_invalid_gate_resources() -> None:
    """Reject absent ions, wrong locations, and conflicting gate claims."""
    architecture = _line_architecture()
    state = architecture.initial_state({"a": (0,), "b": (1,)})
    with pytest.raises(ValueError, match="outside the machine state"):
        architecture.apply_layer(state, (ScheduledAction(0, Rzz(1, 9, 0.5), 0, 2, "pz"),))
    with pytest.raises(ValueError, match="selected processing-zone segment"):
        architecture.apply_layer(state, (ScheduledAction(0, Rzz(0, 1, 0.5), 0, 2, "pz"),))
    control_state = architecture.initial_state({"b": (0, 1)})
    with pytest.raises(ValueError, match="busy or claimed"):
        architecture.apply_layer(
            control_state,
            (
                ScheduledAction(0, Rzz(0, 1, 0.2), 0, 2, "pz"),
                ScheduledAction(1, Rzz(0, 1, 0.3), 0, 2, "pz"),
            ),
        )
    with pytest.raises(ValueError, match="must not select"):
        architecture.apply_layer(state, (ScheduledAction(0, Rz(0, 0.2), 0, 0, "pz"),))


def test_virtual_gates_keep_same_tick_sequence_order_without_claiming_ions() -> None:
    """Allow ordered zero-duration gates while a physical operation is active."""
    architecture = _line_architecture()
    state = architecture.initial_state({"b": (0, 1)})
    busy = architecture.apply_layer(state, (ScheduledAction(0, Rzz(0, 1, 0.2), 0, 2, "pz"),))

    updated = architecture.apply_layer(
        busy,
        (
            ScheduledAction(1, Rz(0, 0.2), 1, 0),
            ScheduledAction(2, Rz(0, 0.3), 1, 0),
        ),
    )

    assert dict(updated.ions_busy_until)[0] == 2


def test_cycle_requires_a_closed_one_in_one_out_rotation() -> None:
    """Reject move groups that do not define one simultaneous cycle."""
    a = Segment("a")
    b = Segment("b")
    c = Segment("c")
    architecture = GridArchitecture(
        (a, b, c),
        (
            Junction("ab", (a.end, b.start)),
            Junction("ac", (a.start, c.start)),
        ),
    )
    state = architecture.initial_state({"a": (0,), "b": (1,), "c": (2,)})
    with pytest.raises(ValueError, match="closed segment rotation"):
        architecture.apply_layer(
            state,
            (
                ScheduledAction(
                    0,
                    Cycle((JunctionMove(a.end, b.start, (0,)), JunctionMove(c.start, a.start, (2,)))),
                    0,
                    1,
                ),
            ),
        )


def test_state_compatibility_and_schedule_validity_are_checked() -> None:
    """Reject states that do not describe exactly one architecture."""
    architecture = _line_architecture()
    state = architecture.initial_state({"a": (0,)})
    missing_segment = replace(state, occupancy=(("a", (0,)),))
    schedule = Schedule((), 0, missing_segment)

    assert not architecture.is_schedule_valid(schedule)
    with pytest.raises(ValueError, match="exactly the architecture segments"):
        architecture.replay_schedule(schedule)


def test_serialized_architecture_rejects_unknown_schema_and_action() -> None:
    """Fail clearly when persisted data does not match the Grid contract."""
    architecture = _line_architecture()
    data = architecture.to_dict()
    with pytest.raises(ValueError, match="schema or version"):
        GridArchitecture.from_dict({**data, "version": 99})
    with pytest.raises(ValueError, match="unknown Grid action"):
        GridArchitecture.from_dict({**data, "supported_action_types": ["missing"]})


def test_layout_adapter_rejects_ambiguous_inputs() -> None:
    """Reject topology inputs that cannot produce stable compiler identities."""
    with pytest.raises(ValueError, match=">= 2"):
        rectangular_grid(1, 2)
    with pytest.raises(TypeError, match="integer"):
        rectangular_grid(rows=cast("Any", 1.5), columns=2)
    with pytest.raises(ValueError, match="unknown segments"):
        rectangular_grid(2, 2, unordered_segments=("missing",))
    directed = nx.DiGraph()
    directed.add_edge("a", "b")
    with pytest.raises(TypeError, match="undirected"):
        from_networkx(directed)
    graph = nx.Graph()
    graph.add_edge(("rich", "node"), "plain")
    with pytest.raises(TypeError, match="explicit junction_id identities"):
        from_networkx(graph)


def test_machine_state_rejects_duplicate_ions_and_backwards_time() -> None:
    """Keep state identities and the machine clock internally consistent."""
    with pytest.raises(ValueError, match="exactly one segment"):
        GridMachineState(
            occupancy=(("a", (0,)), ("b", (0,))),
            ions_busy_until=((0, 0),),
            junctions_busy_until=(),
            pzs_busy_until=(),
        )
    state = _line_architecture().initial_state({"a": (0,)})
    with pytest.raises(ValueError, match="must not move backwards"):
        state.at_time(-1)
    with pytest.raises(KeyError):
        state.ion_segment(9)


def test_machine_state_rejects_inconsistent_keys_and_times() -> None:
    """Reject duplicate resources and availability data that disagrees with occupancy."""
    base = {
        "occupancy": (("a", (0,)),),
        "ions_busy_until": ((0, 0),),
        "junctions_busy_until": (("j", 0),),
        "pzs_busy_until": (("pz", 0),),
    }
    with pytest.raises(ValueError, match="duplicate keys"):
        _machine_state(**{**base, "occupancy": (("a", (0,)), ("a", (1,)))})
    with pytest.raises(ValueError, match="non-empty strings"):
        _machine_state(**{**base, "occupancy": (("", (0,)),)})
    with pytest.raises(TypeError, match="integer ion"):
        _machine_state(**{**base, "occupancy": (("a", cast("Any", (False,))),)})
    with pytest.raises(ValueError, match="non-negative"):
        _machine_state(**{**base, "occupancy": (("a", (-1,)),), "ions_busy_until": ((-1, 0),)})
    with pytest.raises(TypeError, match="time must be an integer"):
        _machine_state(**base, time=cast("Any", 0.5))
    with pytest.raises(ValueError, match="time must be non-negative"):
        _machine_state(**base, time=-1)
    with pytest.raises(ValueError, match="exactly the placed ions"):
        _machine_state(**{**base, "ions_busy_until": ((1, 0),)})
    with pytest.raises(ValueError, match="must not precede"):
        _machine_state(**{**base, "ions_busy_until": ((0, 0),)}, time=1)
    with pytest.raises(TypeError, match="availability times must be integers"):
        _machine_state(**{**base, "junctions_busy_until": (("j", cast("Any", 0.5)),)})


def test_machine_state_serialization_rejects_malformed_occupancy() -> None:
    """Round-trip a state and reject malformed serialized occupancy entries."""
    state = _line_architecture().initial_state({"a": (0,)})
    assert GridMachineState.from_dict(state.to_dict()) == state
    bad_pair = {**state.to_dict(), "occupancy": [["a"]]}
    bad_ions = {**state.to_dict(), "occupancy": [["a", [False]]]}
    with pytest.raises(ValueError, match="segment and ion-list pairs"):
        GridMachineState.from_dict(bad_pair)
    with pytest.raises(ValueError, match="ion lists must contain integers"):
        GridMachineState.from_dict(bad_ions)
    with pytest.raises(TypeError, match="time must be an integer"):
        state.at_time(cast("Any", 0.5))


def test_architecture_checks_all_state_resources_and_capacity() -> None:
    """Reject otherwise well-formed states with incompatible architecture resources."""
    architecture = _line_architecture()
    state = architecture.initial_state({"a": (0,)})
    no_junction = replace(state, junctions_busy_until=())
    no_zone = replace(state, pzs_busy_until=())
    over_capacity = replace(state, occupancy=(("a", (0, 1, 2)), ("b", ())), ions_busy_until=((0, 0), (1, 0), (2, 0)))
    for malformed, message in (
        (no_junction, "exactly the architecture junctions"),
        (no_zone, "exactly the architecture processing zones"),
        (over_capacity, "exceeds capacity"),
    ):
        with pytest.raises(ValueError, match=message):
            architecture.apply_layer(malformed, ())


def test_layer_rejects_actions_the_device_does_not_offer() -> None:
    """Reject action types outside the device catalog and actions without Grid timing."""
    a = Segment("a")
    b = Segment("b")
    c = Segment("c")
    architecture = GridArchitecture(
        (a, b, c),
        (
            Junction("ab", (a.end, b.start)),
            Junction("bc", (b.end, c.start)),
            Junction("ca", (c.end, a.start)),
        ),
        supported_action_types=(JunctionMove,),
    )
    state = architecture.initial_state({"a": (0,), "b": (1,), "c": (2,)})
    rotation = Cycle((
        JunctionMove(a.end, b.start, (0,)),
        JunctionMove(b.end, c.start, (1,)),
        JunctionMove(c.end, a.start, (2,)),
    ))

    with pytest.raises(ValueError, match="unsupported Grid action type"):
        architecture.apply_layer(state, (ScheduledAction(0, rotation, 0, 1),))
    with pytest.raises(TypeError, match="no duration"):
        architecture.action_duration(_UnsupportedAction())


def test_layer_rejects_unsupported_or_shared_processing_zones() -> None:
    """Require zone support and one gate per zone at a time."""
    architecture = GridArchitecture(
        (Segment("control", capacity=4, processing_zones=(ProcessingZone("pz", (Rx, Rz)),)),),
        (),
    )
    state = architecture.initial_state({"control": (0, 1, 2, 3)})
    with pytest.raises(ValueError, match="does not support Rzz"):
        architecture.apply_layer(state, (ScheduledAction(0, Rzz(0, 1, 0.5), 0, 2, "pz"),))
    with pytest.raises(ValueError, match="processing zone 'pz' is busy or claimed"):
        architecture.apply_layer(
            state,
            (ScheduledAction(0, Rx(0, 0.5), 0, 1, "pz"), ScheduledAction(1, Rx(1, 0.5), 0, 1, "pz")),
        )


def test_unordered_state_must_use_canonical_order() -> None:
    """Keep one stored order for each unordered occupancy."""
    architecture = GridArchitecture((Segment("bucket", 2, SegmentOccupancy.UNORDERED),), ())
    state = architecture.initial_state({"bucket": (1, 0)})
    reordered = replace(state, occupancy=(("bucket", (1, 0)),))

    assert state.occupants("bucket") == (0, 1)
    with pytest.raises(ValueError, match="canonical ion order"):
        architecture.apply_layer(reordered, ())


def test_serialized_values_reject_invalid_enumerations_and_gate_names() -> None:
    """Fail clearly when persisted hardware values name unknown kinds."""
    with pytest.raises(ValueError, match="'start' or 'end'"):
        SegmentEndpoint.from_dict({"segment_id": "a", "orientation": "middle"})
    with pytest.raises(ValueError, match="'ordered' or 'unordered'"):
        Segment.from_dict({"segment_id": "a", "capacity": 1, "occupancy": "stacked"})
    zone = {"zone_id": "pz"}
    with pytest.raises(ValueError, match="must contain strings"):
        ProcessingZone.from_dict({**zone, "supported_gate_types": [1]}, gate_types={})
    with pytest.raises(ValueError, match="unknown processing-zone gate type"):
        ProcessingZone.from_dict({**zone, "supported_gate_types": ["gate.missing"]}, gate_types={})
    with pytest.raises(TypeError, match="JunctionMove values"):
        Cycle(cast("Any", (JunctionMove(Segment("a").end, Segment("b").start, (0,)), "move")))
    state = _line_architecture().initial_state({"a": (0,)})
    with pytest.raises(ValueError, match="integer pairs"):
        GridMachineState.from_dict({**state.to_dict(), "ions_busy_until": [[0]]})


def test_layout_constructors_reject_invalid_capacities_and_zone_segments() -> None:
    """Reject layout options that cannot produce a valid segment graph."""
    with pytest.raises(TypeError, match="segment_capacity must be an integer"):
        rectangular_grid(2, 2, segment_capacity=cast("Any", 1.5))
    with pytest.raises(ValueError, match="segment_capacity must be >= 1"):
        rectangular_grid(2, 2, segment_capacity=0)
    with pytest.raises(ValueError, match="processing_zone_segments contains unknown"):
        rectangular_grid(2, 2, processing_zone_segments=("missing",))
    with pytest.raises(ValueError, match="columns must be >= 3"):
        hexagonal_grid(2, 2)
    with pytest.raises(TypeError, match="segment_capacity must be an integer"):
        hexagonal_grid(2, 3, segment_capacity=cast("Any", 1.5))
    with pytest.raises(ValueError, match="segment_capacity must be >= 1"):
        hexagonal_grid(2, 3, segment_capacity=0)
    with pytest.raises(ValueError, match="unordered_segments contains unknown"):
        hexagonal_grid(2, 3, unordered_segments=("missing",))
    with pytest.raises(ValueError, match="processing_zone_segments contains unknown"):
        hexagonal_grid(2, 3, processing_zone_segments=("missing",))


@pytest.mark.parametrize(
    ("node_data", "edge_data", "error", "message"),
    [
        ({"junction_id": 1}, {}, TypeError, "junction_id attributes must be strings"),
        ({"junction_id": "same"}, {}, ValueError, "junction identities must be unique"),
        ({}, {"segment_id": 1}, TypeError, "segment_id attributes must be strings"),
        ({}, {"capacity": "2"}, TypeError, "capacity attributes must be integers"),
        ({}, {"occupancy": "stacked"}, ValueError, "'ordered' or 'unordered'"),
    ],
)
def test_networkx_adapter_rejects_invalid_attributes(
    node_data: dict[str, object], edge_data: dict[str, object], error: type[Exception], message: str
) -> None:
    """Reject physical graph attributes that cannot define stable segment data."""
    graph = nx.Graph()
    graph.add_node(0, **node_data)
    graph.add_node(1, **node_data)
    graph.add_edge(0, 1, **edge_data)

    with pytest.raises(error, match=message):
        from_networkx(graph)


def test_networkx_adapter_names_integer_and_coordinate_nodes() -> None:
    """Derive stable junction identities from integer and coordinate-pair nodes."""
    graph = nx.Graph()
    graph.add_edge(0, (0, 1))

    architecture = from_networkx(graph)

    assert {junction.junction_id for junction in architecture.junctions} == {"0", "0,1"}
    assert architecture.segment("segment:0--0,1").capacity == 1

    with pytest.raises(ValueError, match="processing_zone_segments contains unknown"):
        from_networkx(graph, processing_zone_segments=("missing",))


def test_architecture_and_schedule_save_to_explicit_json_paths(tmp_path: Path) -> None:
    """Persist Grid values without implicit working-directory output."""
    architecture = _line_architecture()
    architecture_path = architecture.save(tmp_path / "device")
    state = architecture.initial_state({"a": (0,)})
    schedule = Schedule((), 0, state)
    schedule_path = schedule.save(tmp_path / "schedule")

    assert architecture_path.suffix == ".json"
    restored_architecture = GridArchitecture.from_dict(json.loads(architecture_path.read_text(encoding="utf-8")))
    assert restored_architecture == architecture
    assert GridArchitecture.load(architecture_path) == architecture
    assert load_schedule(schedule_path) == schedule
