# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Pauli-frame replay across dynamical-decoupling pulses."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import GlobalGate, Rx, Ry, Rz, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.dd import GateSpec
from mqt.ionshuttler.linear.dd.frame_replay import (
    FrameHistory,
    PauliFrame,
    PauliFrameOperation,
    accumulated_frame_phase,
    build_frame_history,
    effective_action,
    frame_operation_for_action,
    frame_operation_for_gate_spec,
    framed_action_events,
    global_pulse_timesteps,
)
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.schedule import Schedule, schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state
from mqt.ionshuttler.linear.timeline import build_timeline

_ARCHITECTURE = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
_GLOBAL_X = GlobalGate(gate_name="rx", theta=pi, ions=(0,))
_GLOBAL_Y = GlobalGate(gate_name="ry", theta=pi, ions=(0,))


def _result(
    path: list[AdvanceTime | GlobalGate | Rx | Ry | Rz | Shuttle],
    end_time: int,
    *,
    num_ions: int = 1,
) -> Schedule:
    program = schedule_from_path(
        list(path),
        create_initial_state(num_ions, _ARCHITECTURE, initial_positions=list(range(num_ions))),
        _ARCHITECTURE,
    )
    assert program.end_time == end_time
    return program


def test_pauli_frames_compose_and_transform_single_qubit_axes() -> None:
    """Track Pauli products and preserve target gate timing metadata."""
    frame = PauliFrame("X").compose(PauliFrameOperation("Y"))

    assert frame == PauliFrame("Z")
    assert frame.phase_sign("Z") == 1
    assert frame.phase_sign("X") == -1
    assert effective_action(Rz(ion=0, theta=0.25), PauliFrame("X")) == Rz(ion=0, theta=-0.25)
    assert effective_action(Rx(ion=0, theta=0.5), PauliFrame("Z")) == Rx(ion=0, theta=-0.5)


def test_frame_history_includes_same_boundary_and_terminal_global_pulses() -> None:
    """Apply pulses before the following interval and retain terminal frames."""
    program = _result(
        [
            _GLOBAL_X,
            AdvanceTime(),
            _GLOBAL_Y,
        ],
        2,
    )
    timeline = build_timeline(program, _ARCHITECTURE)
    history = build_frame_history(timeline)

    assert history.global_frames_by_time == (PauliFrame("X"), PauliFrame("Z"), PauliFrame("Z"))
    assert global_pulse_timesteps(timeline) == (0, 1)
    assert accumulated_frame_phase(
        timeline,
        ion=0,
        t_start=0,
        t_end=1,
        field_profile=FieldProfile(num_sites=2, site_field=((0, 0.5),)),
        frame_history=history,
    ) == pytest.approx(-0.5)


def test_global_pulse_on_some_ions_changes_only_their_frames() -> None:
    """Apply a pulse that targets some ions to their frames instead of the global frame."""
    program = _result(
        [GlobalGate(gate_name="rx", theta=pi, ions=(1,)), AdvanceTime()],
        1,
        num_ions=2,
    )
    timeline = build_timeline(program, _ARCHITECTURE)
    history = build_frame_history(timeline)
    events = framed_action_events(program, _ARCHITECTURE, timeline)

    assert history.global_frames_by_time == (PauliFrame(), PauliFrame())
    assert history.frame_for_ion(0, 0) == PauliFrame()
    assert history.frame_for_ion(1, 0) == PauliFrame("X")
    assert global_pulse_timesteps(timeline) == (0,)
    assert [(event.kind, event.ion_frames) for event in events] == [("global_dd_pulse", ((1, PauliFrame("X")),))]


def test_global_pulse_on_every_ion_changes_the_global_frame() -> None:
    """Treat a pulse that targets every ion of the schedule as a global frame change."""
    pulse = GlobalGate(gate_name="rx", theta=pi, ions=(0, 1))
    program = _result([pulse, AdvanceTime()], 1, num_ions=2)
    timeline = build_timeline(program, _ARCHITECTURE)
    history = build_frame_history(timeline)
    events = framed_action_events(program, _ARCHITECTURE, timeline)

    assert history.global_frames_by_time == (PauliFrame("X"), PauliFrame("X"))
    assert history.local_frame_overrides == {}
    assert frame_operation_for_action(pulse) == PauliFrameOperation("X")
    assert frame_operation_for_action(Rx(ion=0, theta=pi)) is None
    assert events[0].ion_frames == ((0, PauliFrame("X")), (1, PauliFrame("X")))


def test_accumulated_frame_phase_rejects_interval_outside_schedule() -> None:
    """Reject a requested phase interval that lies outside the schedule."""
    program = _result(
        [
            _GLOBAL_X,
            AdvanceTime(),
            _GLOBAL_Y,
        ],
        2,
    )
    timeline = build_timeline(program, _ARCHITECTURE)
    history = build_frame_history(timeline)

    with pytest.raises(ValueError, match="t_start"):
        accumulated_frame_phase(
            timeline,
            ion=0,
            t_start=0,
            t_end=3,
            field_profile=FieldProfile(num_sites=2, site_field=((0, 0.5),)),
            frame_history=history,
        )


def test_frame_for_ion_rejects_out_of_range_timestep() -> None:
    """Reject a timestep outside the tracked schedule boundaries."""
    program = _result(
        [
            _GLOBAL_X,
            AdvanceTime(),
            _GLOBAL_Y,
        ],
        2,
    )
    timeline = build_timeline(program, _ARCHITECTURE)
    history = build_frame_history(timeline)

    with pytest.raises(ValueError, match="timestep"):
        history.frame_for_ion(0, 3)


def test_build_frame_history_rejects_unsupported_local_pulse_action() -> None:
    """Raise instead of silently ignoring a non-Rx/Ry/Rz local pulse record."""
    program = _result([Shuttle(ion=0, src=0, dst=1), AdvanceTime()], 1)
    timeline = build_timeline(program, _ARCHITECTURE)
    local_pulse_action_ids = frozenset({program.scheduled_actions[0].action_id})

    with pytest.raises(ValueError, match="unsupported local DD pulse"):
        build_frame_history(timeline, local_pulse_action_ids)


def test_frame_history_rejects_mismatched_local_override_length() -> None:
    """Reject a local override that does not span every schedule boundary."""
    with pytest.raises(ValueError, match="local_frame_overrides"):
        FrameHistory(
            start_time=0,
            global_frames_by_time=(PauliFrame(), PauliFrame()),
            local_frame_overrides={0: (PauliFrame(),)},
        )


def test_local_record_identity_and_same_timestep_event_order() -> None:
    """Identify only the recorded local pulse when gates share a boundary."""
    program = _result(
        [Rx(ion=0, theta=pi), Rz(ion=0, theta=0.3), AdvanceTime()],
        1,
    )

    timeline = build_timeline(program, _ARCHITECTURE)
    local_pulse_action_ids = frozenset({program.scheduled_actions[0].action_id})
    events = framed_action_events(program, _ARCHITECTURE, timeline, local_pulse_action_ids)
    history = build_frame_history(timeline, local_pulse_action_ids)

    assert [event.kind for event in events] == ["local_dd_pulse", "algorithmic_gate"]
    assert [event.action for event in events] == list(program.path)
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


def test_equal_same_boundary_pulses_retain_distinct_provenance_identity() -> None:
    """Classify equal actions by DD-owned identity instead of tuple equality."""
    program = _result(
        [Rx(ion=0, theta=pi), Rx(ion=0, theta=pi), AdvanceTime()],
        1,
    )

    events = framed_action_events(
        program,
        _ARCHITECTURE,
        build_timeline(program, _ARCHITECTURE),
        frozenset({program.scheduled_actions[0].action_id}),
    )

    assert [event.kind for event in events] == ["local_dd_pulse", "algorithmic_gate"]
    assert program.scheduled_actions[0].action == program.scheduled_actions[1].action
    assert program.scheduled_actions[0].action_id != program.scheduled_actions[1].action_id
