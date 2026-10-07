# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear action models and their architecture-owned meaning."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import ClassVar

import pytest

from mqt.ionshuttler.linear import GateTiming, TransportTiming
from mqt.ionshuttler.linear.actions import (
    DEFAULT_ACTION_TYPES,
    LINEAR_ACTION_TYPES,
    Action,
    GateAction,
    GlobalGate,
    PhysicalSwap,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    Shuttle,
    SingleQubitGate,
    TransportAction,
    TwoQubitGate,
    decode_linear_action,
)
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.expand import apply
from mqt.ionshuttler.linear.state import AdvanceTime, State, advance_time


@dataclass(frozen=True)
class _CustomAction(Action):
    """Hardware-specific action without Linear rules."""

    ion: int
    serialized_type: ClassVar[str] = "test.custom"


@dataclass(frozen=True)
class _CustomRx(Rx):
    """Gate subclass that Linear architectures do not implement."""

    serialized_type: ClassVar[str] = "test.custom_rx"


def _state(
    *,
    positions: tuple[tuple[int, int], ...] = ((0, 0),),
    completed_gates: frozenset[int] = frozenset(),
    in_progress_gates: tuple[tuple[int, int], ...] = (),
    ions_busy_until: tuple[tuple[int, int], ...] = ((0, 0),),
    pzs_busy_until: tuple[tuple[str, int], ...] = (("all_sites", 0),),
    time: int = 0,
) -> State:
    return State(
        positions=positions,
        completed_gates=completed_gates,
        in_progress_gates=in_progress_gates,
        ions_busy_until=ions_busy_until,
        pzs_busy_until=pzs_busy_until,
        time=time,
    )


def test_action_values_are_immutable_and_hashable() -> None:
    """Keep actions stable so the compiler can compare and store them reliably."""
    actions: tuple[Action, ...] = (
        Shuttle(ion=0, src=1, dst=2),
        PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2),
        Rx(ion=0, theta=0.1),
        Ry(ion=0, theta=0.2),
        Rz(ion=0, theta=0.3),
        Rxx(ion_a=0, ion_b=1, theta=0.4),
        Ryy(ion_a=0, ion_b=1, theta=0.5),
        Rzz(ion_a=0, ion_b=1, theta=0.6),
        GlobalGate(gate_name="rx", theta=0.7, ions=(0, 1)),
        GlobalGate(gate_name="rx", theta=0.7, ions=(0,)),
    )

    assert len(set(actions)) == len(actions)
    assert Shuttle(ion=0, src=1, dst=2) == actions[0]


def test_actions_expose_ordered_ion_operands() -> None:
    """Expose one operand view without changing action-specific fields."""
    actions = (
        Shuttle(ion=2, src=0, dst=1),
        PhysicalSwap(ion_a=2, ion_b=1, pos_a=0, pos_b=1),
        Rx(ion=2, theta=0.1),
        Rzz(ion_a=2, ion_b=1, theta=0.2),
        GlobalGate(gate_name="rx", theta=0.3, ions=(2, 0, 1)),
    )

    assert tuple(action.ions for action in actions) == ((2,), (2, 1), (2,), (2, 1), (0, 1, 2))


def test_supported_action_types_switch_implemented_capabilities() -> None:
    """Enable or disable operations that the Linear architecture already implements."""
    without_swaps = LinearArchitecture(num_sites=3, supported_action_types=(Shuttle, Rx, Ry, Rz, Rzz))
    with_global_control = LinearArchitecture(
        num_sites=3, supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate, Rxx)
    )

    assert not without_swaps.supports(PhysicalSwap)
    assert with_global_control.supports(GlobalGate)
    assert with_global_control.supports(Rxx)
    assert set(LINEAR_ACTION_TYPES.values()) == {Rx, Ry, Rz, Rxx, Ryy, Rzz, GlobalGate, Shuttle, PhysicalSwap}


def test_architecture_owns_gate_timing_and_implementation() -> None:
    """Keep hardware timing out of shared circuit gate occurrences."""
    rx = Rx(ion=2, theta=0.25)
    ry = Ry(ion=2, theta=0.25)
    rz = Rz(ion=2, theta=0.25)
    architecture = LinearArchitecture(
        num_sites=3,
        gate_timing=GateTiming(rx=0, ry=2, virtual_single_qubit_gates=frozenset({"rx", "rz"})),
    )

    assert architecture.action_duration(rx) == 0
    assert architecture.action_duration(ry) == 2
    assert architecture.action_duration(rz) == 0
    assert architecture.is_virtual_gate(rx)
    assert not architecture.is_virtual_gate(ry)
    assert architecture.is_virtual_gate(rz)


def test_architecture_owns_transport_timing() -> None:
    """Take every transport duration from the architecture rather than the action value."""
    architecture = LinearArchitecture(num_sites=3, transport_timing=TransportTiming(shuttle=2, swap=5))

    assert architecture.action_duration(Shuttle(ion=0, src=0, dst=1)) == 2
    assert architecture.action_duration(PhysicalSwap(ion_a=0, ion_b=1, pos_a=0, pos_b=1)) == 5


def test_global_gate_uses_the_ordinary_gate_duration() -> None:
    """Give a global rotation the duration of the same local rotation."""
    architecture = LinearArchitecture(num_sites=1, gate_timing=GateTiming(rx=4, ry=3))

    assert architecture.action_duration(GlobalGate(gate_name="rx", theta=pi, ions=(0,))) == 4
    assert architecture.action_duration(GlobalGate(gate_name="ry", theta=pi, ions=(0,))) == 3
    assert architecture.action_processing_zone(GlobalGate(gate_name="rx", theta=pi, ions=(0,)), _state()) is None


def test_action_hierarchy_distinguishes_hardware_actions_and_search_transitions() -> None:
    """Distinguish hardware controls from the compiler's passage of time."""
    assert isinstance(Rx(ion=0, theta=0.1), SingleQubitGate)
    assert isinstance(Rzz(ion_a=0, ion_b=1, theta=0.1), TwoQubitGate)
    assert isinstance(GlobalGate(gate_name="rx", theta=pi, ions=(0,)), GateAction)
    assert isinstance(Shuttle(ion=0, src=0, dst=1), TransportAction)
    assert not isinstance(AdvanceTime(), Action)


def test_architecture_rejects_actions_it_does_not_implement() -> None:
    """Treat the catalog as a capability switch rather than a plugin mechanism."""
    with pytest.raises(ValueError, match="implements no action types named: _CustomAction"):
        LinearArchitecture(num_sites=3, supported_action_types=(*DEFAULT_ACTION_TYPES, _CustomAction))
    with pytest.raises(ValueError, match="implements no action types named: _CustomRx"):
        LinearArchitecture(num_sites=3, supported_action_types=(*DEFAULT_ACTION_TYPES, _CustomRx))
    with pytest.raises(TypeError, match="defines no rules for _CustomAction"):
        LinearArchitecture(num_sites=3).is_action_valid(_state(), _CustomAction(ion=0))
    with pytest.raises(TypeError, match="defines no duration for _CustomAction"):
        LinearArchitecture(num_sites=3).action_duration(_CustomAction(ion=0))


def test_physical_actions_change_only_machine_fields() -> None:
    """Let hardware controls change the machine without completing circuit gates."""
    architecture = LinearArchitecture(
        num_sites=4,
        processing_zones={"pz": [0, 1, 2, 3]},
        gate_timing=GateTiming(rzz=3),
        transport_timing=TransportTiming(shuttle=2),
    )
    state = _state(
        positions=((0, 0), (1, 2)),
        completed_gates=frozenset({7}),
        in_progress_gates=((8, 5),),
        ions_busy_until=((0, 0), (1, 0)),
        pzs_busy_until=(("pz", 0),),
        time=2,
    )

    shuttled = architecture.apply_action(state, Shuttle(ion=0, src=0, dst=1))
    gated = architecture.apply_action(state, Rzz(ion_a=0, ion_b=1, theta=0.5))
    global_gated = architecture.apply_action(state, GlobalGate(gate_name="rx", theta=pi, ions=(0, 1)))

    assert shuttled.positions == ((0, 1), (1, 2))
    assert shuttled.ions_busy_until == ((0, 4), (1, 0))
    assert gated.ions_busy_until == ((0, 5), (1, 5))
    assert gated.pzs_busy_until == (("pz", 5),)
    assert global_gated == state
    for updated in (shuttled, gated):
        assert updated.completed_gates == state.completed_gates
        assert updated.in_progress_gates == state.in_progress_gates
        assert updated.time == state.time


def test_advance_time_owns_scheduler_clock_and_completion_transition() -> None:
    """Advance the clock and complete only operations due by the next tick."""
    architecture = LinearArchitecture(num_sites=2)
    state = _state(
        completed_gates=frozenset({2}),
        in_progress_gates=((3, 2), (4, 3)),
        ions_busy_until=((0, 3),),
        pzs_busy_until=(("all_sites", 3),),
        time=1,
    )

    updated = advance_time(state)

    assert updated == apply(state, architecture, AdvanceTime())
    assert updated.time == 2
    assert updated.completed_gates == frozenset({2, 3})
    assert updated.in_progress_gates == ((4, 3),)
    assert updated.ions_busy_until == state.ions_busy_until
    assert updated.pzs_busy_until == state.pzs_busy_until


def test_action_serialization_uses_each_actions_intrinsic_data() -> None:
    """Serialize actions without architecture-owned timing or implementation data."""
    assert Shuttle(ion=0, src=1, dst=2).to_dict() == {"type": "linear.shuttle", "ion": 0, "src": 1, "dst": 2}
    assert PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2).to_dict() == {
        "type": "linear.physical_swap",
        "ion_a": 0,
        "ion_b": 1,
        "pos_a": 1,
        "pos_b": 2,
    }
    assert Rz(ion=0, theta=0.2).to_dict() == {"type": "gate.rz", "ion": 0, "theta": 0.2}
    assert GlobalGate(gate_name="rx", theta=0.3, ions=(1, 0)).to_dict() == {
        "type": "gate.global",
        "gate_name": "rx",
        "theta": 0.3,
        "ions": [0, 1],
    }


def test_linear_actions_decode_only_implemented_action_types() -> None:
    """Restore every built-in Linear action and refuse other serialized types."""
    for action in (
        Shuttle(ion=0, src=1, dst=2),
        PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2),
        Rxx(ion_a=0, ion_b=1, theta=0.4),
        GlobalGate(gate_name="ry", theta=pi, ions=(0, 1)),
    ):
        assert decode_linear_action(action.to_dict()) == action
    with pytest.raises(ValueError, match=r"unknown action type: test\.custom"):
        decode_linear_action(_CustomAction(ion=0).to_dict())
