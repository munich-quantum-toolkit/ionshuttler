# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Pauli-frame replay across dynamical-decoupling pulses."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import AdvanceTime, GateSpec, GlobalPulse, Rx, Ry, Rz, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd.frame_replay import (
    PauliFrame,
    PauliFrameOperation,
    accumulated_frame_phase,
    build_frame_history,
    effective_action,
    frame_operation_for_gate_spec,
    framed_action_events,
    global_pulse_timesteps,
)
from mqt.ionshuttler.linear.dd.timeline import build_timeline
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus, DDInsertionRecord
from mqt.ionshuttler.linear.state import create_initial_state


def _result(
    path: list[AdvanceTime | GlobalPulse | Rx | Ry | Rz | Shuttle],
    num_timesteps: int,
    *,
    dd_insertions: tuple[DDInsertionRecord, ...] = (),
) -> CompilationResult:
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    return CompilationResult(
        status=CompilationStatus.SUCCESS,
        path=list(path),
        num_timesteps=num_timesteps,
        architecture=architecture,
        initial_state=create_initial_state(1, architecture, initial_positions=[0]),
        dd_insertions=dd_insertions,
    )


def test_pauli_frames_compose_and_transform_single_qubit_axes() -> None:
    """Track Pauli products and preserve target gate timing metadata."""
    frame = PauliFrame("X").compose(PauliFrameOperation("Y"))

    assert frame == PauliFrame("Z")
    assert frame.phase_sign("Z") == 1
    assert frame.phase_sign("X") == -1
    assert effective_action(Rz(ion=0, theta=0.25, duration=3, virtual=False), PauliFrame("X")) == Rz(
        ion=0,
        theta=-0.25,
        duration=3,
        virtual=False,
    )
    assert effective_action(Rx(ion=0, theta=0.5, duration=2), PauliFrame("Z")) == Rx(
        ion=0,
        theta=-0.5,
        duration=2,
    )


def test_frame_history_includes_same_boundary_and_terminal_global_pulses() -> None:
    """Apply pulses before the following interval and retain terminal frames."""
    result = _result(
        [
            GlobalPulse(GateSpec("Rx", pi)),
            AdvanceTime(),
            GlobalPulse(GateSpec("Ry", pi)),
        ],
        1,
    )
    timeline = build_timeline(result)
    history = build_frame_history(timeline, result)

    assert history.global_frames_by_time == (PauliFrame("X"), PauliFrame("Z"))
    assert global_pulse_timesteps(timeline) == (0, 1)
    assert accumulated_frame_phase(
        timeline,
        ion=0,
        t_start=0,
        t_end=1,
        field_profile=FieldProfile(num_sites=2, site_field=((0, 0.5),)),
        frame_history=history,
    ) == pytest.approx(-0.5)


def test_local_record_identity_and_same_timestep_event_order() -> None:
    """Identify only the recorded local pulse when gates share a boundary."""
    record = DDInsertionRecord(
        ion=0,
        window=(0, 1),
        scheme_name="Hahn",
        gate_timesteps=(0,),
    )
    result = _result([Rx(ion=0, theta=pi), Rz(ion=0, theta=0.3), AdvanceTime()], 1, dd_insertions=(record,))

    events = framed_action_events(result)
    history = build_frame_history(build_timeline(result), result)

    assert [event.kind for event in events] == ["local_dd_pulse", "algorithmic_gate", "advance_time"]
    assert [event.action for event in events] == result.path
    assert history.frame_for_ion(0, 0) == PauliFrame("X")
    assert events[0].ion_frames == ((0, PauliFrame("X")),)
    assert events[1].ion_frames == ((0, PauliFrame("X")),)


@pytest.mark.parametrize(
    "spec",
    [GateSpec("Rx", pi / 2), GateSpec("Rxx", pi)],
)
def test_frame_replay_rejects_unsupported_pulses(spec: GateSpec) -> None:
    """Reject pulses that cannot be represented as a Pauli frame."""
    with pytest.raises(ValueError, match=r"odd pi|does not support"):
        frame_operation_for_gate_spec(spec)


def test_frame_history_rejects_missing_recorded_local_action() -> None:
    """Fail clearly when legacy DD metadata cannot identify its pulse."""
    record = DDInsertionRecord(ion=0, window=(0, 1), scheme_name="Hahn", gate_timesteps=(0,))
    result = _result([AdvanceTime()], 1, dd_insertions=(record,))

    with pytest.raises(ValueError, match="local DD gate"):
        build_frame_history(build_timeline(result), result)
