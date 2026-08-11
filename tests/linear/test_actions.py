# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear compiler action models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import pytest

from mqt.ionshuttler.linear.actions import (
    Action,
    AdvanceTime,
    GateAction,
    GateSpec,
    GlobalPulse,
    PhysicalAction,
    PhysicalSwap,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    SchedulableAction,
    Shuttle,
    SingleQubitGate,
    TwoQubitGate,
)
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.expand import apply
from mqt.ionshuttler.linear.state import State
from mqt.ionshuttler.linear.validation import is_action_valid


@dataclass(frozen=True)
class _CustomAction(PhysicalAction):
    """Example hardware-specific action defined outside the built-in model."""

    ion: int
    duration: int = 2

    def is_valid(self, state: State, architecture: Architecture) -> bool:
        """Require an existing ion and a hardware array with spare sites."""
        return self.ion in dict(state.positions) and architecture.num_sites > len(state.positions)

    def apply(self, state: State, architecture: Architecture) -> State:
        """Reserve the selected ion while preserving compiler progress."""
        del architecture
        ions_busy = dict(state.ions_busy_until)
        ions_busy[self.ion] = state.time + self.duration
        return replace(state, ions_busy_until=tuple(sorted(ions_busy.items())))


@dataclass(frozen=True, slots=True)
class _CalibrationGate(GateAction):
    """Example custom gate with a fixed two-timestep duration."""

    duration: int = 2

    def apply(self, state: State, architecture: Architecture) -> State:
        """Return an unchanged state after reading the validated duration."""
        del architecture
        assert self._duration() == self.duration
        return state


def test_action_values_are_immutable_and_hashable() -> None:
    """Keep actions stable so the compiler can compare and store them reliably."""
    actions: tuple[SchedulableAction, ...] = (
        Shuttle(ion=0, src=1, dst=2),
        PhysicalSwap(ion_a=0, ion_b=1, pos_a=1, pos_b=2),
        Rx(ion=0, theta=0.1),
        Ry(ion=0, theta=0.2),
        Rz(ion=0, theta=0.3),
        Rxx(ion_a=0, ion_b=1, theta=0.4),
        Ryy(ion_a=0, ion_b=1, theta=0.5),
        Rzz(ion_a=0, ion_b=1, theta=0.6),
        GlobalPulse(gate=GateSpec("Rx", theta=0.7)),
        AdvanceTime(),
    )

    assert len(set(actions)) == len(actions)
    assert Shuttle(ion=0, src=1, dst=2) == actions[0]


def test_custom_gate_duration_is_available_to_the_compiler() -> None:
    """Read the duration exposed by a custom hardware gate."""
    architecture = Architecture(num_sites=1)
    state = State(
        positions=((0, 0),),
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=((0, 0),),
        pzs_busy_until=(("all_sites", 0),),
        time=0,
    )

    updated = apply(state, architecture, _CalibrationGate(), gate_id=4)

    assert updated.in_progress_gates == ((4, 2),)


def test_scheduled_actions_reject_invalid_durations() -> None:
    """Reject noninteger and nonpositive scheduled durations."""
    invalid_durations: tuple[object, ...] = (True, 0, -1, 1.5)
    for duration in invalid_durations:
        with pytest.raises(ValueError, match="action duration"):
            Shuttle(ion=0, src=0, dst=1, duration=cast("int", duration))


def test_single_qubit_gate_defaults_match_hardware_conventions() -> None:
    """Default Rx and Ry to physical while defaulting Rz to virtual."""
    rx = Rx(ion=2, theta=0.25)
    ry = Ry(ion=2, theta=0.25)
    rz = Rz(ion=2, theta=0.25)

    assert not rx.virtual
    assert not ry.virtual
    assert rz.virtual
    assert rz.duration == 0


def test_virtual_single_qubit_gate_requires_zero_duration() -> None:
    """Allow virtual rotations of any axis only when their duration is zero."""
    assert Rx(ion=0, theta=0.25, duration=0, virtual=True).virtual

    with pytest.raises(ValueError, match="virtual single-qubit gate duration"):
        Rz(ion=0, theta=0.25, duration=1)
    with pytest.raises(ValueError, match="virtual single-qubit gate duration"):
        Ry(ion=0, theta=0.25, duration=1, virtual=True)


def test_zero_duration_physical_single_qubit_gate_warns() -> None:
    """Allow a physical zero-duration abstraction while making it conspicuous."""
    with pytest.warns(UserWarning, match="may share a compiler timestep"):
        gate = Rz(ion=0, theta=0.25, duration=0, virtual=False)

    assert not gate.virtual
    assert gate.duration == 0


def test_single_qubit_gate_requires_boolean_virtual_flag() -> None:
    """Reject ambiguous truthy values at the action boundary."""
    with pytest.raises(TypeError, match="virtual must be a boolean"):
        Rx(ion=0, theta=0.25, virtual=cast("bool", 1))


def test_action_hierarchy_distinguishes_physical_and_scheduler_transitions() -> None:
    """Distinguish hardware controls from the compiler's passage of time."""
    rx = Rx(ion=0, theta=0.1)
    rzz = Rzz(ion_a=0, ion_b=1, theta=0.1)
    assert isinstance(rx, Action)
    assert isinstance(rx, PhysicalAction)
    assert isinstance(rx, SingleQubitGate)
    assert isinstance(rzz, Action)
    assert isinstance(rzz, PhysicalAction)
    assert isinstance(rzz, TwoQubitGate)
    assert isinstance(AdvanceTime(), Action)
    assert not isinstance(AdvanceTime(), PhysicalAction)


def test_custom_action_owns_validity_transition_and_serialization() -> None:
    """Add a hardware-specific action without changing the shared validator."""
    architecture = Architecture(num_sites=3)
    state = State(
        positions=((0, 1),),
        completed_gates=frozenset({4}),
        in_progress_gates=((5, 3),),
        ions_busy_until=((0, 0),),
        pzs_busy_until=(("all_sites", 0),),
        time=1,
    )
    action = _CustomAction(ion=0)

    assert is_action_valid(state, action, architecture)
    updated = action.apply(state, architecture)
    assert updated.ions_busy_until == ((0, 3),)
    assert updated.completed_gates == state.completed_gates
    assert updated.in_progress_gates == state.in_progress_gates
    assert action.to_dict() == {"type": "_CustomAction", "ion": 0, "duration": 2}


def test_physical_actions_change_only_machine_fields() -> None:
    """Let hardware controls change the machine without completing circuit gates."""
    architecture = Architecture(num_sites=4, processing_zones={"pz": [0, 1, 2, 3]})
    state = State(
        positions=((0, 0), (1, 2)),
        completed_gates=frozenset({7}),
        in_progress_gates=((8, 5),),
        ions_busy_until=((0, 0), (1, 0)),
        pzs_busy_until=(("pz", 0),),
        time=2,
    )

    shuttled = Shuttle(ion=0, src=0, dst=1, duration=2).apply(state, architecture)
    gated = Rzz(ion_a=0, ion_b=1, theta=0.5, duration=3).apply(state, architecture)

    assert shuttled.positions == ((0, 1), (1, 2))
    assert shuttled.ions_busy_until == ((0, 4), (1, 0))
    assert gated.ions_busy_until == ((0, 5), (1, 5))
    assert gated.pzs_busy_until == (("pz", 5),)
    for updated in (shuttled, gated):
        assert updated.completed_gates == state.completed_gates
        assert updated.in_progress_gates == state.in_progress_gates
        assert updated.time == state.time


def test_advance_time_owns_scheduler_clock_and_completion_transition() -> None:
    """Advance the clock and complete only operations due by the next tick."""
    architecture = Architecture(num_sites=2)
    state = State(
        positions=((0, 0),),
        completed_gates=frozenset({2}),
        in_progress_gates=((3, 2), (4, 3)),
        ions_busy_until=((0, 3),),
        pzs_busy_until=(("all_sites", 3),),
        time=1,
    )

    updated = AdvanceTime().apply(state, architecture)

    assert updated.time == 2
    assert updated.completed_gates == frozenset({2, 3})
    assert updated.in_progress_gates == ((4, 3),)
    assert updated.ions_busy_until == state.ions_busy_until
    assert updated.pzs_busy_until == state.pzs_busy_until


def test_action_serialization_uses_each_actions_intrinsic_data() -> None:
    """Let each action describe itself while omitting an unchanged virtuality default."""
    assert Shuttle(ion=0, src=1, dst=2).to_dict() == {
        "type": "Shuttle",
        "ion": 0,
        "src": 1,
        "dst": 2,
        "duration": 1,
    }
    assert Rz(ion=0, theta=0.2).to_dict() == {
        "type": "Rz",
        "ion": 0,
        "theta": 0.2,
        "duration": 0,
    }
    assert Rx(ion=0, theta=0.2, duration=0, virtual=True).to_dict()["virtual"] is True
    assert GlobalPulse(gate=GateSpec("Rx", theta=0.3)).to_dict() == {
        "type": "GlobalPulse",
        "gate_name": "Rx",
        "theta": 0.3,
        "duration": 1,
    }
