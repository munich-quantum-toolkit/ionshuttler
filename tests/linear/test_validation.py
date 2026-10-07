# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear action and transport-layer validity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from mqt.ionshuttler.linear import GateTiming
from mqt.ionshuttler.linear.actions import (
    GlobalGate,
    PhysicalSwap,
    Rx,
    Rz,
    Rzz,
    Shuttle,
    TransportAction,
)
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.state import State, has_pending_timed_work
from mqt.ionshuttler.linear.validation import is_adjacent, is_transport_layer_valid


@dataclass(frozen=True)
class _ParkingTransfer(TransportAction):
    """Transport without Linear rules."""

    ion: int
    destination: int
    serialized_type: ClassVar[str] = "test.parking_transfer"


def _state(
    positions: tuple[tuple[int, int], ...],
    *,
    ions_busy_until: tuple[tuple[int, int], ...] | None = None,
    pzs_busy_until: tuple[tuple[str, int], ...] = (("pz", 0),),
    in_progress_gates: tuple[tuple[int, int], ...] = (),
    time: int = 0,
) -> State:
    """Build a state with free ions and a free processing zone by default."""
    return State(
        positions=positions,
        completed_gates=frozenset(),
        in_progress_gates=in_progress_gates,
        ions_busy_until=(ions_busy_until if ions_busy_until is not None else tuple((ion, 0) for ion, _ in positions)),
        pzs_busy_until=pzs_busy_until,
        time=time,
    )


def test_adjacency_uses_linear_neighbor_distance() -> None:
    """Treat only sites one position apart as adjacent."""
    assert is_adjacent(2, 3)
    assert is_adjacent(3, 2)
    assert not is_adjacent(2, 2)
    assert not is_adjacent(2, 4)


def test_action_validation_enforces_transport_occupancy_and_busy_times() -> None:
    """Move or swap ions only when their sites and ions are available."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [1, 2, 3]})
    free_state = _state(((0, 0), (1, 3)))
    busy_state = _state(((0, 0), (1, 1)), ions_busy_until=((0, 2), (1, 0)), time=1)
    swap_state = _state(((0, 1), (1, 2)))

    assert architecture.is_action_valid(free_state, Shuttle(ion=0, src=0, dst=1))
    assert not architecture.is_action_valid(free_state, Shuttle(ion=0, src=1, dst=2))
    assert not architecture.is_action_valid(free_state, Shuttle(ion=0, src=0, dst=2))
    assert not architecture.is_action_valid(busy_state, Shuttle(ion=0, src=0, dst=1))
    assert architecture.is_action_valid(swap_state, PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2))
    assert not architecture.is_action_valid(free_state, PhysicalSwap(ion_a=0, ion_b=1, pos_a=0, pos_b=3))


def test_action_validation_enforces_gate_processing_zone_resources() -> None:
    """Start physical gates only on free ions in a free processing zone."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [1, 2, 3]})
    gate_state = _state(((0, 1), (1, 2)))
    outside_state = _state(((0, 0), (1, 4)))
    busy_pz_state = _state(((0, 1), (1, 2)), pzs_busy_until=(("pz", 2),))

    assert architecture.is_action_valid(gate_state, Rx(ion=0, theta=1.0))
    assert architecture.is_action_valid(gate_state, Rzz(ion_a=0, ion_b=1, theta=1.0))
    assert not architecture.is_action_valid(outside_state, Rx(ion=0, theta=1.0))
    assert not architecture.is_action_valid(outside_state, Rzz(ion_a=0, ion_b=1, theta=1.0))
    assert not architecture.is_action_valid(busy_pz_state, Rx(ion=0, theta=1.0))


def test_two_qubit_gate_allows_nonadjacent_ions_in_same_processing_zone() -> None:
    """Allow an interaction across an empty site within one processing zone."""
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [1, 2, 3]})
    gate = Rzz(ion_a=0, ion_b=1, theta=1.0)

    assert architecture.is_action_valid(_state(((0, 1), (1, 3))), gate)


def test_virtual_single_qubit_gate_requires_only_an_existing_ion() -> None:
    """Allow virtual rotations without waiting for physical hardware."""
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [1]},
        gate_timing=GateTiming(rx=0, virtual_single_qubit_gates=frozenset({"rx", "rz"})),
    )
    busy_state = _state(
        ((0, 0),),
        ions_busy_until=((0, 5),),
        pzs_busy_until=(("pz", 5),),
        time=1,
    )

    assert architecture.is_action_valid(busy_state, Rz(ion=0, theta=0.25))
    assert architecture.is_action_valid(busy_state, Rx(ion=0, theta=0.25))
    assert not architecture.is_action_valid(busy_state, Rz(ion=1, theta=0.25))


def test_physical_rz_uses_ordinary_single_qubit_resources() -> None:
    """Apply physical scheduling checks to Rz regardless of its duration."""
    architecture = LinearArchitecture(
        num_sites=3,
        processing_zones={"pz": [1]},
        gate_timing=GateTiming(rz=1, virtual_single_qubit_gates=frozenset()),
    )
    gate_state = _state(((0, 1),))
    busy_state = _state(((0, 1),), ions_busy_until=((0, 2),))
    outside_state = _state(((0, 0),))

    physical_rz = Rz(ion=0, theta=0.25)

    assert architecture.is_action_valid(gate_state, physical_rz)
    assert not architecture.is_action_valid(busy_state, physical_rz)
    assert not architecture.is_action_valid(outside_state, physical_rz)


def test_global_gate_needs_no_free_ion_or_processing_zone() -> None:
    """Let global control overlap local gates and transport in the Linear model."""
    architecture = LinearArchitecture(num_sites=3, processing_zones={"pz": [1]})
    busy_state = _state(
        ((0, 0), (1, 1)),
        ions_busy_until=((0, 3), (1, 3)),
        pzs_busy_until=(("pz", 3),),
        time=1,
    )

    assert architecture.is_action_valid(busy_state, GlobalGate(gate_name="rx", theta=np.pi, ions=(0, 1)))
    assert architecture.is_action_valid(busy_state, GlobalGate(gate_name="rx", theta=np.pi, ions=(1,)))


def test_targeted_global_gate_requires_present_ions() -> None:
    """Reject a global gate that targets an ion absent from the machine state."""
    architecture = LinearArchitecture(num_sites=3)
    state = _state(((0, 0), (1, 1)), pzs_busy_until=(("all_sites", 0),))

    assert not architecture.is_action_valid(state, GlobalGate(gate_name="rx", theta=np.pi, ions=(0, 2)))


def test_waiting_is_meaningful_only_while_timed_work_is_pending() -> None:
    """Let the compiler avoid idle time while busy hardware still has work to finish."""
    idle_state = _state(((0, 0),), pzs_busy_until=(("all_sites", 0),))
    waiting_state = _state(
        ((0, 0),),
        ions_busy_until=((0, 2),),
        pzs_busy_until=(("all_sites", 0),),
    )

    assert not has_pending_timed_work(idle_state)
    assert has_pending_timed_work(waiting_state)


def test_transport_layer_allows_simultaneous_conveyor_shift() -> None:
    """Allow occupied destinations when every occupant vacates in the layer."""
    architecture = LinearArchitecture(num_sites=4)
    state = _state(
        ((0, 0), (1, 1), (2, 2)),
        pzs_busy_until=(("all_sites", 0),),
    )
    conveyor = (
        Shuttle(ion=0, src=0, dst=1),
        Shuttle(ion=1, src=1, dst=2),
        Shuttle(ion=2, src=2, dst=3),
    )

    assert not architecture.is_action_valid(state, conveyor[0])
    assert is_transport_layer_valid(state, conveyor, architecture)


def test_transport_layer_rejects_conflicting_or_repeated_actions() -> None:
    """Reject final collisions and multiple actions for the same ion."""
    architecture = LinearArchitecture(num_sites=4)
    state = _state(((0, 0), (1, 2)), pzs_busy_until=(("all_sites", 0),))

    assert not is_transport_layer_valid(
        state,
        (Shuttle(ion=0, src=0, dst=1), Shuttle(ion=1, src=2, dst=1)),
        architecture,
    )
    assert not is_transport_layer_valid(
        state,
        (Shuttle(ion=0, src=0, dst=1), Shuttle(ion=0, src=0, dst=1)),
        architecture,
    )
    assert not is_transport_layer_valid(
        _state(((0, 0), (1, 1))),
        (Shuttle(ion=0, src=0, dst=1), Shuttle(ion=1, src=1, dst=0)),
        architecture,
    )


def test_transport_layer_combines_shuttles_and_swaps() -> None:
    """Check shuttles and swaps that start together against one pre-state."""
    architecture = LinearArchitecture(num_sites=4)
    state = _state(((0, 0), (1, 1), (2, 2)), pzs_busy_until=(("all_sites", 0),))

    assert is_transport_layer_valid(
        state,
        (PhysicalSwap(ion_a=0, ion_b=1, pos_a=0, pos_b=1), Shuttle(ion=2, src=2, dst=3)),
        architecture,
    )
    assert not is_transport_layer_valid(
        state,
        (PhysicalSwap(ion_a=1, ion_b=2, pos_a=1, pos_b=2), Shuttle(ion=2, src=2, dst=3)),
        architecture,
    )


def test_transport_layer_rejects_transport_without_linear_rules() -> None:
    """Accept only the Linear transport operations defined by the architecture."""
    architecture = LinearArchitecture(num_sites=4)
    state = _state(((0, 0), (1, 2)), pzs_busy_until=(("all_sites", 0),))

    with pytest.raises(TypeError, match="only shuttles and physical swaps"):
        is_transport_layer_valid(state, (_ParkingTransfer(ion=0, destination=1),), architecture)
