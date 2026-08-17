# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear action and transport-layer validity."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from mqt.ionshuttler.linear.actions import (
    AdvanceTime,
    GateSpec,
    GlobalPulse,
    PhysicalSwap,
    Rx,
    Rz,
    Rzz,
    Shuttle,
    TransportAction,
)
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.state import State, has_pending_timed_work
from mqt.ionshuttler.linear.validation import is_action_valid, is_adjacent, is_transport_layer_valid


@dataclass(frozen=True)
class _ParkingTransfer(TransportAction):
    """Move an ion to a hardware-defined parking site."""

    ion: int
    destination: int
    enabled: bool = True
    duration: int = 1

    def is_valid(self, state: State, architecture: Architecture) -> bool:
        """Return whether the parking transfer is enabled and stays on the device."""
        return self.enabled and self.ion in dict(state.positions) and 0 <= self.destination < architecture.num_sites

    def apply(self, state: State, architecture: Architecture) -> State:
        """Move the ion to its parking site."""
        del architecture
        positions = dict(state.positions)
        positions[self.ion] = self.destination
        return replace(state, positions=tuple(sorted(positions.items())))


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
    architecture = Architecture(num_sites=5, processing_zones={"pz": [1, 2, 3]})
    free_state = _state(((0, 0), (1, 3)))
    busy_state = _state(((0, 0), (1, 1)), ions_busy_until=((0, 2), (1, 0)), time=1)
    swap_state = _state(((0, 1), (1, 2)))

    assert is_action_valid(free_state, Shuttle(ion=0, src=0, dst=1), architecture)
    assert not is_action_valid(free_state, Shuttle(ion=0, src=1, dst=2), architecture)
    assert not is_action_valid(busy_state, Shuttle(ion=0, src=0, dst=1), architecture)
    assert is_action_valid(
        swap_state,
        PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2),
        architecture,
    )


def test_action_validation_enforces_gate_processing_zone_resources() -> None:
    """Start physical gates only on free ions in a free processing zone."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [1, 2, 3]})
    gate_state = _state(((0, 1), (1, 2)))
    outside_state = _state(((0, 0), (1, 4)))
    busy_pz_state = _state(((0, 1), (1, 2)), pzs_busy_until=(("pz", 2),))

    assert is_action_valid(gate_state, Rx(ion=0, theta=1.0), architecture)
    assert is_action_valid(gate_state, Rzz(ion_a=0, ion_b=1, theta=1.0), architecture)
    assert not is_action_valid(outside_state, Rx(ion=0, theta=1.0), architecture)
    assert not is_action_valid(outside_state, Rzz(ion_a=0, ion_b=1, theta=1.0), architecture)
    assert not is_action_valid(busy_pz_state, Rx(ion=0, theta=1.0), architecture)


def test_two_qubit_gate_allows_nonadjacent_ions_in_same_processing_zone() -> None:
    """Allow an interaction across an empty site within one processing zone."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [1, 2, 3]})
    gate = Rzz(ion_a=0, ion_b=1, theta=1.0)

    assert is_action_valid(_state(((0, 1), (1, 3))), gate, architecture)


def test_virtual_single_qubit_gate_requires_only_an_existing_ion() -> None:
    """Allow virtual rotations without waiting for physical hardware."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    busy_state = _state(
        ((0, 0),),
        ions_busy_until=((0, 5),),
        pzs_busy_until=(("pz", 5),),
        time=1,
    )

    virtual_rz = Rz(ion=0, theta=0.25)
    virtual_rx = Rx(ion=0, theta=0.25, duration=0, virtual=True)
    assert is_action_valid(busy_state, virtual_rz, architecture)
    assert is_action_valid(busy_state, virtual_rx, architecture)
    assert not is_action_valid(busy_state, Rz(ion=1, theta=0.25), architecture)


def test_physical_rz_uses_ordinary_single_qubit_resources() -> None:
    """Apply physical scheduling checks to Rz regardless of its duration."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    gate_state = _state(((0, 1),))
    busy_state = _state(((0, 1),), ions_busy_until=((0, 2),))
    outside_state = _state(((0, 0),))

    physical_rz = Rz(ion=0, theta=0.25, duration=1, virtual=False)
    with pytest.warns(UserWarning, match="may share a compiler timestep"):
        zero_duration_physical_rz = Rz(ion=0, theta=0.25, duration=0, virtual=False)

    assert is_action_valid(gate_state, physical_rz, architecture)
    assert is_action_valid(gate_state, zero_duration_physical_rz, architecture)
    assert not is_action_valid(busy_state, zero_duration_physical_rz, architecture)
    assert not is_action_valid(outside_state, zero_duration_physical_rz, architecture)


def test_advance_time_validity_is_independent_of_generation_policy() -> None:
    """Allow deliberate waiting while letting the compiler avoid pointless idling."""
    architecture = Architecture(num_sites=3)
    idle_state = _state(((0, 0),), pzs_busy_until=(("all_sites", 0),))
    waiting_state = _state(
        ((0, 0),),
        ions_busy_until=((0, 2),),
        pzs_busy_until=(("all_sites", 0),),
    )

    assert is_action_valid(idle_state, AdvanceTime(), architecture)
    assert is_action_valid(waiting_state, AdvanceTime(), architecture)
    assert not has_pending_timed_work(idle_state)
    assert has_pending_timed_work(waiting_state)
    assert is_action_valid(idle_state, GlobalPulse(gate=GateSpec("Rx", theta=np.pi)), architecture)


def test_transport_layer_allows_simultaneous_conveyor_shift() -> None:
    """Allow occupied destinations when every occupant vacates in the layer."""
    architecture = Architecture(num_sites=4)
    state = _state(
        ((0, 0), (1, 1), (2, 2)),
        pzs_busy_until=(("all_sites", 0),),
    )
    conveyor = (
        Shuttle(ion=0, src=0, dst=1),
        Shuttle(ion=1, src=1, dst=2),
        Shuttle(ion=2, src=2, dst=3),
    )

    assert not is_action_valid(state, conveyor[0], architecture)
    assert is_transport_layer_valid(state, conveyor, architecture)


def test_transport_layer_rejects_conflicting_or_repeated_actions() -> None:
    """Reject final collisions and multiple actions for the same ion."""
    architecture = Architecture(num_sites=4)
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


def test_transport_layer_uses_custom_action_validity_and_transition() -> None:
    """Honor custom transport rules and include their final positions in conflicts."""
    architecture = Architecture(num_sites=4)
    state = _state(((0, 0), (1, 2)), pzs_busy_until=(("all_sites", 0),))

    assert is_transport_layer_valid(state, (_ParkingTransfer(ion=0, destination=1),), architecture)
    assert not is_transport_layer_valid(
        state,
        (_ParkingTransfer(ion=0, destination=1, enabled=False),),
        architecture,
    )
    assert not is_transport_layer_valid(
        state,
        (_ParkingTransfer(ion=0, destination=2),),
        architecture,
    )

    mixed_state = _state(((0, 0), (1, 1), (2, 2)), pzs_busy_until=(("all_sites", 0),))
    assert is_transport_layer_valid(
        mixed_state,
        (_ParkingTransfer(ion=0, destination=3), Shuttle(ion=1, src=1, dst=0)),
        architecture,
    )
    assert is_transport_layer_valid(
        mixed_state,
        (
            _ParkingTransfer(ion=0, destination=3),
            PhysicalSwap(ion_a=1, ion_b=2, pos_a=1, pos_b=2),
        ),
        architecture,
    )
    assert not is_transport_layer_valid(
        mixed_state,
        (_ParkingTransfer(ion=0, destination=3), Shuttle(ion=0, src=0, dst=1)),
        architecture,
    )
    assert not is_transport_layer_valid(
        mixed_state,
        (_ParkingTransfer(ion=0, destination=3), Shuttle(ion=2, src=2, dst=3)),
        architecture,
    )
