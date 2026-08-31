# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the nearest-feasible midpoint Hahn comparator."""

from __future__ import annotations

from math import pi

import pytest

from mqt.ionshuttler.linear.actions import Action, AdvanceTime, Rx, Ry, Rz, Rzz, Shuttle
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd import NearestHahnConfig, NearestHahnReport, run_nearest_hahn
from mqt.ionshuttler.linear.dd import nearest_hahn as nearest_hahn_module
from mqt.ionshuttler.linear.dd import schedule_transform as schedule_transform_module
from mqt.ionshuttler.linear.dd.frame_replay import PauliFrame, build_frame_history
from mqt.ionshuttler.linear.dd.result import DDPassResult
from mqt.ionshuttler.linear.dd.schedule_transform import validate_rebuilt_schedule
from mqt.ionshuttler.linear.dd.timeline import CompiledTimeline, build_timeline
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state


def _schedule(architecture: Architecture, path: list[Action], positions: list[int]) -> ActionSchedule:
    return ActionSchedule.from_actions(
        path,
        create_initial_state(len(positions), architecture, initial_positions=positions),
    )


def _pulse_frame(
    output: DDPassResult[NearestHahnReport], architecture: Architecture, ion: int, timestep: int
) -> PauliFrame:
    action_ids = frozenset(action_id for sequence in output.report.sequences for action_id in sequence.action_ids)
    history = build_frame_history(build_timeline(output.schedule, architecture), action_ids)
    return history.frame_for_ion(ion, timestep)


def test_nearest_hahn_places_the_exact_midpoint_without_extending_the_schedule() -> None:
    """Use the ion's existing processing-zone access at the window midpoint."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    original = _schedule(architecture, [AdvanceTime() for _ in range(4)], [0])

    output = run_nearest_hahn(original, architecture)

    (record,) = output.report.opportunities
    assert record.status == "exact"
    assert record.ideal_midpoint == 2
    assert record.selected_timestep == 2
    assert record.processing_zone == "pz"
    assert record.signed_displacement == 0
    assert output.schedule.num_timesteps == original.num_timesteps
    assert output.report.sequences[0].pulse_timesteps == (2,)
    assert validate_rebuilt_schedule(output.schedule, architecture)


def test_nearest_hahn_agrees_with_the_shared_midpoint_quantization() -> None:
    """Derive the ideal midpoint from the same policy the Hahn sequence uses."""
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})

    for length, expected in ((2, 1), (3, 2), (5, 2), (7, 4)):
        original = _schedule(architecture, [AdvanceTime() for _ in range(length)], [0])
        (record,) = run_nearest_hahn(original, architecture).report.opportunities
        assert record.ideal_midpoint == expected


def test_nearest_hahn_shifts_to_naturally_available_zone_access() -> None:
    """Take the nearest boundary at which the ion already sits in a zone."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [2]})
    original = _schedule(
        architecture,
        [
            AdvanceTime(),
            AdvanceTime(),
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Shuttle(ion=0, src=1, dst=2),
            *(AdvanceTime() for _ in range(3)),
        ],
        [0],
    )

    output = run_nearest_hahn(original, architecture)

    (record,) = output.report.opportunities
    assert record.status == "shifted"
    assert record.selected_timestep is not None
    assert record.selected_timestep > record.ideal_midpoint
    assert record.absolute_displacement == record.selected_timestep - record.ideal_midpoint
    assert validate_rebuilt_schedule(output.schedule, architecture)


def test_nearest_hahn_breaks_equal_distance_ties_toward_the_earlier_boundary() -> None:
    """Prefer the earlier of two equally distant feasible boundaries."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    original = _schedule(
        architecture,
        [AdvanceTime(), AdvanceTime(), Shuttle(ion=0, src=0, dst=1), AdvanceTime(), AdvanceTime(), AdvanceTime()],
        [0],
    )

    (record,) = run_nearest_hahn(original, architecture).report.opportunities

    assert record.ideal_midpoint == 2
    assert record.status == "shifted"
    assert record.selected_timestep == 1
    assert record.signed_displacement == -1


def test_nearest_hahn_serializes_pulses_competing_for_one_processing_zone() -> None:
    """Let an earlier placement occupy the zone the next opportunity wanted."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    original = _schedule(
        architecture,
        [AdvanceTime(), AdvanceTime(), Ry(ion=1, theta=0.5), AdvanceTime(), AdvanceTime(), AdvanceTime()],
        [0, 1],
    )

    output = run_nearest_hahn(original, architecture)
    by_key = {(record.ion, record.window): record for record in output.report.opportunities}

    assert by_key[1, (0, 2)].selected_timestep == 1
    assert by_key[0, (0, 5)].selected_timestep == 3
    assert by_key[0, (0, 5)].status == "shifted"
    assert validate_rebuilt_schedule(output.schedule, architecture)


def test_nearest_hahn_orders_contention_chronologically_then_by_ion() -> None:
    """Resolve competing windows in a deterministic, position-independent order."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    original = _schedule(architecture, [AdvanceTime() for _ in range(4)], [0, 1])

    output = run_nearest_hahn(original, architecture)

    keys = [(record.ideal_midpoint, record.ion, record.window) for record in output.report.opportunities]
    assert keys == sorted(keys)
    assert [record.ion for record in output.report.opportunities] == [0, 1]


def test_nearest_hahn_reports_missing_processing_zone_access() -> None:
    """Skip a window in which the ion never occupies a processing zone."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [1]})
    original = _schedule(architecture, [AdvanceTime() for _ in range(4)], [0])

    output = run_nearest_hahn(original, architecture)

    (record,) = output.report.opportunities
    assert record.status == "skipped"
    assert record.skip_reason == "no_processing_zone_access"
    assert record.selected_timestep is None
    assert output.schedule is original
    assert output.report.sequences == ()


def test_nearest_hahn_reports_a_busy_ion_and_honors_the_minimum_window() -> None:
    """Skip when transport occupies every boundary the window offers."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    original = _schedule(
        architecture,
        [Shuttle(ion=0, src=0, dst=1, duration=2), AdvanceTime(), AdvanceTime()],
        [0],
    )

    output = run_nearest_hahn(original, architecture)

    assert [record.skip_reason for record in output.report.opportunities] == ["ion_busy"]
    assert run_nearest_hahn(original, architecture, NearestHahnConfig(min_idle_timesteps=3)).report.opportunities == ()


def test_nearest_hahn_reports_processing_zone_contention() -> None:
    """Skip when another ion occupies the shared zone at every candidate."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    original = _schedule(
        architecture,
        [Ry(ion=1, theta=0.5), AdvanceTime(), Ry(ion=1, theta=0.5), AdvanceTime()],
        [0, 1],
    )

    output = run_nearest_hahn(original, architecture)

    (record,) = output.report.opportunities
    assert record.ion == 0
    assert record.status == "skipped"
    assert record.skip_reason == "processing_zone_busy"
    assert output.schedule is original


def test_nearest_hahn_persistent_frame_survives_virtual_rz_and_two_qubit_gates() -> None:
    """Carry the unrecovered frame past later logical operations."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [0, 1]})
    original = _schedule(
        architecture,
        [
            AdvanceTime(),
            AdvanceTime(),
            AdvanceTime(),
            Rz(ion=0, theta=0.3, virtual=True),
            Rzz(ion_a=0, ion_b=1, theta=0.5),
            AdvanceTime(),
        ],
        [0, 1],
    )

    output = run_nearest_hahn(original, architecture)
    ion_zero = next(record for record in output.report.opportunities if record.ion == 0)

    assert ion_zero.status != "skipped"
    assert _pulse_frame(output, architecture, 0, output.schedule.num_timesteps) == PauliFrame("X")


def test_nearest_hahn_round_trips_its_report_through_json() -> None:
    """Preserve placements and the skip audit across serialization."""
    architecture = Architecture(num_sites=2, processing_zones={"pz": [1]})
    original = _schedule(architecture, [AdvanceTime() for _ in range(4)], [0, 1])

    output = run_nearest_hahn(original, architecture)
    restored = DDPassResult.from_json(output.to_json(), NearestHahnReport)

    assert restored == output
    assert {record.status for record in output.report.opportunities} == {"exact", "skipped"}
    assert output.report.placed == 1


def test_nearest_hahn_never_inserts_transport_or_extends_the_makespan() -> None:
    """Keep the comparator schedule-preserving by construction."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [1]})
    original = _schedule(
        architecture,
        [Shuttle(ion=0, src=0, dst=1), *(AdvanceTime() for _ in range(5))],
        [0, 2],
    )

    output = run_nearest_hahn(original, architecture)

    assert output.schedule.num_timesteps == original.num_timesteps
    original_transport = [action for action in original.path if isinstance(action, Shuttle)]
    updated_transport = [action for action in output.schedule.path if isinstance(action, Shuttle)]
    assert updated_transport == original_transport
    inserted = [action for action in output.schedule.path if isinstance(action, Rx)]
    assert all(action.theta == pytest.approx(pi) for action in inserted)
    assert not [action for action in output.schedule.path if isinstance(action, Ry)]


def test_nearest_hahn_reuses_one_timeline_and_validates_the_batch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid replaying the growing schedule after every accepted pulse."""
    architecture = Architecture(num_sites=2, processing_zones={"left": [0], "right": [1]})
    original = _schedule(architecture, [AdvanceTime() for _ in range(4)], [0, 1])
    real_build_timeline = build_timeline
    real_validate = validate_rebuilt_schedule
    timeline_builds = 0
    validations = 0

    def counted_build_timeline(schedule: ActionSchedule, model: Architecture) -> CompiledTimeline:
        nonlocal timeline_builds
        timeline_builds += 1
        return real_build_timeline(schedule, model)

    def counted_validate(schedule: ActionSchedule, model: Architecture) -> bool:
        nonlocal validations
        validations += 1
        return real_validate(schedule, model)

    monkeypatch.setattr(nearest_hahn_module, "build_timeline", counted_build_timeline)
    monkeypatch.setattr(schedule_transform_module, "build_timeline", counted_build_timeline)
    monkeypatch.setattr(nearest_hahn_module, "validate_rebuilt_schedule", counted_validate)

    output = run_nearest_hahn(original, architecture)

    assert output.report.placed == 2
    assert timeline_builds == 1
    assert validations == 1


def test_nearest_hahn_falls_back_to_per_placement_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover the conservative placement result when batch validation fails."""
    architecture = Architecture(num_sites=2, processing_zones={"left": [0], "right": [1]})
    original = _schedule(architecture, [AdvanceTime() for _ in range(4)], [0, 1])
    expected = run_nearest_hahn(original, architecture)
    validations = 0

    def reject_batch_then_accept_placements(_schedule: ActionSchedule, _model: Architecture) -> bool:
        nonlocal validations
        validations += 1
        return validations > 1

    monkeypatch.setattr(
        nearest_hahn_module,
        "validate_rebuilt_schedule",
        reject_batch_then_accept_placements,
    )

    output = run_nearest_hahn(original, architecture)

    assert output == expected
    assert validations == 1 + expected.report.placed


def test_nearest_hahn_config_rejects_invalid_values() -> None:
    """Reject invalid comparator configuration eagerly."""
    with pytest.raises(TypeError, match="min_idle_timesteps"):
        NearestHahnConfig(min_idle_timesteps=True)
    with pytest.raises(ValueError, match="min_idle_timesteps"):
        NearestHahnConfig(min_idle_timesteps=1)
    with pytest.raises(ValueError, match="label"):
        NearestHahnConfig(label="")
