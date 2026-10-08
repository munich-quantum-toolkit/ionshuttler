# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for periodic global dynamical decoupling."""

from __future__ import annotations

from dataclasses import replace
from math import pi
from typing import TYPE_CHECKING

import pytest
from qiskit import QuantumCircuit

from mqt.ionshuttler.core.gates import GlobalGate, Rx
from mqt.ionshuttler.linear import GateTiming, LinearCompiler
from mqt.ionshuttler.linear.actions import (
    DEFAULT_ACTION_TYPES,
    Shuttle,
)
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.dd import (
    GateSpec,
    GlobalDDConfig,
    GlobalDDReport,
    apply_periodic_global_dd,
    compute_critical_segments,
)
from mqt.ionshuttler.linear.dd.frame_replay import PauliFrame, build_frame_history
from mqt.ionshuttler.linear.dd.result import DDPassResult
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.replay import replay_schedule
from mqt.ionshuttler.linear.schedule import schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state
from mqt.ionshuttler.linear.timeline import build_timeline

if TYPE_CHECKING:
    from mqt.ionshuttler.core.schedule import Schedule
    from mqt.ionshuttler.linear.state import SearchTransition

_GLOBAL_X = GlobalGate(gate_name="rx", theta=pi, ions=(0,))


def _result(
    architecture: LinearArchitecture,
    path: list[SearchTransition],
    *,
    end_time: int,
    positions: list[int] | None = None,
) -> Schedule:
    initial_positions = positions or [0]
    program = schedule_from_path(
        path,
        create_initial_state(len(initial_positions), architecture, initial_positions=initial_positions),
        architecture,
    )
    assert program.end_time == end_time
    return program


def test_periodic_global_dd_overlaps_existing_actions_and_records_summary() -> None:
    """Insert global pulses first at each selected boundary and serialize them."""
    architecture = LinearArchitecture(
        num_sites=2,
        processing_zones={"pz": [0, 1]},
        field_profile=FieldProfile(2, ((0, 1.0), (1, 1.0))),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    original = _result(
        architecture,
        [
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Rx(ion=0, theta=0.5),
            AdvanceTime(),
        ],
        end_time=2,
    )

    output = apply_periodic_global_dd(original, architecture, GlobalDDConfig(spacing=1))
    timeline = build_timeline(output.schedule, architecture)
    restored = DDPassResult.from_json(output.to_json(), GlobalDDReport)

    assert timeline.action_at(0) == (_GLOBAL_X, Shuttle(ion=0, src=0, dst=1))
    assert timeline.action_at(1) == (_GLOBAL_X, Rx(ion=0, theta=0.5))
    assert output.report.pulse_timesteps == (0, 1)
    assert restored == output
    assert sum(isinstance(item.action, GlobalGate) for item in output.schedule.scheduled_actions) == 2


def test_periodic_global_dd_centers_the_first_odd_spacing_window() -> None:
    """Use the source comparator's centered first pulse by default."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        field_profile=FieldProfile(1, ((0, 1.0),), default_field=0.0),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    original = _result(
        architecture,
        [AdvanceTime() for _ in range(5)],
        end_time=5,
    )

    output = apply_periodic_global_dd(original, architecture, GlobalDDConfig(spacing=5))
    history = build_frame_history(build_timeline(output.schedule, architecture))

    assert output.report.pulse_timesteps == (2,)
    assert output.report.phase_cost == pytest.approx(1.0)
    assert history.frame_for_ion(0, 1) == PauliFrame("I")
    assert history.frame_for_ion(0, 2) == PauliFrame("X")


def test_periodic_global_dd_uses_the_architecture_rotation_duration() -> None:
    """Keep a multi-timestep global X pulse within the original schedule."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
        gate_timing=GateTiming(rx=3),
    )
    original = _result(architecture, [AdvanceTime() for _ in range(5)], end_time=5)

    output = apply_periodic_global_dd(original, architecture, GlobalDDConfig(spacing=2))

    pulses = [item for item in output.schedule.scheduled_actions if isinstance(item.action, GlobalGate)]
    assert [item.action for item in pulses] == [_GLOBAL_X]
    assert [item.start_time for item in pulses] == [1]
    assert [item.duration for item in pulses] == [3]
    assert output.schedule.end_time == original.end_time
    replay_schedule(output.schedule, architecture)


def test_periodic_global_dd_shifts_pulses_to_reduce_phase_cost() -> None:
    """Use the shared critical-segment phase cost for pulse adjustment."""
    architecture = LinearArchitecture(
        num_sites=2,
        processing_zones={"pz": [0, 1]},
        field_profile=FieldProfile(2, ((0, 1.0), (1, 2.0)), default_field=0.0),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    original = _result(
        architecture,
        [AdvanceTime() for _ in range(5)],
        end_time=5,
        positions=[0, 1],
    )
    periodic_result = apply_periodic_global_dd(
        original,
        architecture,
        GlobalDDConfig(spacing=5, half_first_window=False),
    )
    shifted_result = apply_periodic_global_dd(
        original,
        architecture,
        GlobalDDConfig(
            spacing=5,
            shift_range=2,
            half_first_window=False,
        ),
    )
    periodic = periodic_result.schedule
    shifted = shifted_result.schedule

    assert {item.action for item in shifted.scheduled_actions if isinstance(item.action, GlobalGate)} == {
        GlobalGate(gate_name="rx", theta=pi, ions=(0, 1))
    }
    assert periodic_result.report.pulse_timesteps == (4,)
    assert periodic_result.report.phase_cost == pytest.approx(18.0)
    assert shifted_result.report.pulse_timesteps == (2,)
    assert shifted_result.report.phase_cost == pytest.approx(2.0)
    assert compute_critical_segments(periodic, architecture).phase_cost == periodic_result.report.phase_cost
    assert compute_critical_segments(shifted, architecture).phase_cost == shifted_result.report.phase_cost


def test_periodic_global_dd_keeps_unimproved_times_and_returns_short_noop() -> None:
    """Keep strict-improvement ties stable and avoid rebuilding empty sequences."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        field_profile=FieldProfile(1, ((0, 1.0),), default_field=0.0),
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    original = _result(architecture, [AdvanceTime() for _ in range(4)], end_time=4)
    periodic = apply_periodic_global_dd(original, architecture, GlobalDDConfig(spacing=2))
    shifted = apply_periodic_global_dd(original, architecture, GlobalDDConfig(spacing=2, shift_range=1))
    short = _result(architecture, [AdvanceTime()], end_time=1)
    unchanged = apply_periodic_global_dd(short, architecture, GlobalDDConfig(spacing=5))

    assert shifted.report == periodic.report
    assert unchanged.schedule is short
    assert unchanged.report.pulse_timesteps == ()


def test_periodic_global_dd_rejects_existing_global_pulses() -> None:
    """Reject ambiguous augmentation of a schedule with global pulses."""
    architecture = LinearArchitecture(
        num_sites=1,
        processing_zones={"pz": [0]},
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    original = _result(
        architecture,
        [_GLOBAL_X, AdvanceTime()],
        end_time=1,
    )

    with pytest.raises(ValueError, match="without global pulses"):
        apply_periodic_global_dd(original, architecture, GlobalDDConfig(spacing=1))


def test_periodic_global_dd_requires_architecture_support() -> None:
    """Reject global controls that the target architecture does not provide."""
    architecture = LinearArchitecture(num_sites=1, processing_zones={"pz": [0]})
    original = _result(architecture, [AdvanceTime()], end_time=1)

    with pytest.raises(ValueError, match="global-pulse support"):
        apply_periodic_global_dd(original, architecture, GlobalDDConfig(spacing=1))


def test_global_dd_can_target_an_extended_architecture_after_compilation() -> None:
    """Compile with a strict subset and add a capability for a later pass."""
    compilation_architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    circuit = QuantumCircuit(1)
    circuit.rx(0.25, 0)
    compilation = LinearCompiler(
        compilation_architecture,
        action_types=DEFAULT_ACTION_TYPES,
    ).compile(circuit)
    extended_architecture = replace(
        compilation_architecture,
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )

    result = apply_periodic_global_dd(
        compilation.schedule,
        extended_architecture,
        GlobalDDConfig(spacing=1),
    )

    assert compilation.architecture is compilation_architecture
    assert result.architecture is extended_architecture
    assert any(isinstance(action, GlobalGate) for action in result.schedule.path)


def test_global_dd_rejects_an_incompatible_replacement_architecture() -> None:
    """Validate the complete input schedule before applying new controls."""
    original_architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    schedule = _result(
        original_architecture,
        [Rx(ion=0, theta=0.25), AdvanceTime()],
        end_time=1,
    )
    incompatible_architecture = LinearArchitecture(
        num_sites=2,
        processing_zones={"other": [0, 1]},
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )

    with pytest.raises(ValueError, match="processing-zone resources"):
        apply_periodic_global_dd(schedule, incompatible_architecture, GlobalDDConfig(spacing=1))


def test_global_dd_config_rejects_invalid_values_and_pulses() -> None:
    """Validate placement controls and the global X-frame restriction."""
    with pytest.raises(TypeError, match="spacing"):
        GlobalDDConfig(spacing=True)
    with pytest.raises(ValueError, match="spacing"):
        GlobalDDConfig(spacing=0)
    with pytest.raises(ValueError, match="shift_range"):
        GlobalDDConfig(spacing=2, shift_range=-1)
    with pytest.raises(ValueError, match="only global X"):
        GlobalDDConfig(spacing=2, pulse=GateSpec("Ry", pi))


def test_global_dd_report_rejects_malformed_public_values() -> None:
    """Keep comparator summaries immutable and internally consistent."""
    with pytest.raises(ValueError, match="strictly increasing"):
        GlobalDDReport(
            scheme_name="periodic_x",
            pulse_timesteps=(3, 1),
            spacing=2,
            phase_cost=0.0,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        GlobalDDReport(
            scheme_name="periodic_x",
            pulse_timesteps=(1, 3),
            spacing=2,
            phase_cost=-1.0,
        )
    with pytest.raises(ValueError, match="JSON object"):
        GlobalDDReport.from_dict([])
