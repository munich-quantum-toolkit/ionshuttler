# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the shared explicit schedule and compilation result."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, cast

import pytest

from mqt.ionshuttler.core.actions import Action, decode_action, index_action_types
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction


@dataclass(frozen=True)
class _Pulse(Action):
    """Action of a minimal test architecture."""

    channel: int
    serialized_type: ClassVar[str] = "test.pulse"


@dataclass(frozen=True)
class _ClockState:
    """Machine state of a minimal test architecture."""

    time: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return the state using JSON-compatible values."""
        return {"time": self.time}

    @classmethod
    def from_dict(cls, data: object) -> _ClockState:
        """Restore a state."""
        return cls(time=cast("dict[str, int]", data)["time"])


@dataclass(frozen=True)
class _ClockArchitecture:
    """Architecture whose actions only advance the clock."""

    def to_dict(self) -> dict[str, object]:
        """Return the architecture using JSON-compatible values."""
        return {"kind": "clock"}

    @classmethod
    def from_dict(cls, data: object) -> _ClockArchitecture:
        """Restore an architecture."""
        assert data == {"kind": "clock"}
        return cls()

    def replay_schedule(self, schedule: Schedule) -> _ClockState:
        """Return the state at the end of the schedule."""
        return _ClockState(time=schedule.end_time)


@dataclass(frozen=True)
class _Counts:
    """Diagnostics of a minimal test compiler."""

    attempts: int

    def to_dict(self) -> dict[str, object]:
        """Return the diagnostics using JSON-compatible values."""
        return {"attempts": self.attempts}

    @classmethod
    def from_dict(cls, data: object) -> _Counts:
        """Restore diagnostics."""
        return cls(attempts=cast("dict[str, int]", data)["attempts"])


_DECODE_PULSE = partial(decode_action, action_types=index_action_types((_Pulse,)))


def _schedule() -> Schedule[Action, _ClockState]:
    return Schedule(
        scheduled_actions=(
            ScheduledAction(0, _Pulse(channel=1), start_time=0, duration=2),
            ScheduledAction(1, _Pulse(channel=2), start_time=0, duration=1, processing_zone_id="zone"),
            ScheduledAction(2, _Pulse(channel=1), start_time=3, duration=0),
        ),
        end_time=4,
        initial_state=_ClockState(),
    )


def test_schedule_round_trips_with_explicit_decoders() -> None:
    """Restore a schedule from caller-supplied action and state decoders."""
    schedule = _schedule()

    restored = Schedule.from_dict(schedule.to_dict(), decode_action=_DECODE_PULSE, decode_state=_ClockState.from_dict)

    assert restored == schedule
    serialized_actions = cast("list[dict[str, object]]", schedule.to_dict()["actions"])
    assert serialized_actions[1]["processing_zone_id"] == "zone"
    assert "processing_zone" not in serialized_actions[1]
    assert schedule.to_dict()["initial_state"] == {"time": 0}
    assert schedule.start_time == 0
    assert schedule.end_time == 4
    assert schedule.duration == 4
    assert schedule.next_action_id == 3


def test_schedule_rejects_unsupported_versions() -> None:
    """Refuse a schedule document from another format version."""
    data = _schedule().to_dict()
    data["version"] = 1

    with pytest.raises(ValueError, match="unsupported schedule schema or version"):
        Schedule.from_dict(data, decode_action=_DECODE_PULSE, decode_state=_ClockState.from_dict)


def test_scheduled_actions_contain_only_actions() -> None:
    """Keep compiler bookkeeping such as time advances out of public schedules."""

    @dataclass(frozen=True)
    class Wait:
        """A search transition that is not a hardware action."""

    with pytest.raises(TypeError, match="action must be an Action"):
        ScheduledAction(0, cast("Action", Wait()), start_time=0, duration=1)


def test_result_round_trips_with_explicit_decoders() -> None:
    """Restore a result without any architecture-specific loader in the shared layer."""
    schedule = _schedule()
    result = CompilationResult(
        status=CompilationStatus.TIMEOUT,
        schedule=schedule,
        architecture=_ClockArchitecture(),
        final_state=_ClockState(time=4),
        wall_clock_s=0.25,
        diagnostics=_Counts(attempts=3),
    )

    restored = CompilationResult.from_dict(
        result.to_dict(),
        decode_architecture=_ClockArchitecture.from_dict,
        decode_action=_DECODE_PULSE,
        decode_state=_ClockState.from_dict,
        decode_diagnostics=_Counts.from_dict,
    )

    assert restored == result
    assert restored.wall_clock_s == result.wall_clock_s
    assert result.to_dict()["diagnostics"] == {"attempts": 3}
    restored.validate()


def test_result_equality_ignores_wall_clock_time() -> None:
    """Treat two compilation outcomes as equal when only their wall-clock times differ."""
    fast, slow = (
        CompilationResult(
            status=CompilationStatus.SUCCESS,
            schedule=_schedule(),
            architecture=_ClockArchitecture(),
            final_state=_ClockState(time=4),
            wall_clock_s=wall_clock_s,
        )
        for wall_clock_s in (0.25, 3.0)
    )

    assert fast == slow


def test_result_validation_rejects_a_mismatched_final_state() -> None:
    """Compare the single replay result with the stored final state."""
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=_schedule(),
        architecture=_ClockArchitecture(),
        final_state=_ClockState(time=3),
    )

    with pytest.raises(ValueError, match="schedule replay does not match final_state"):
        result.validate()


def test_result_exposes_no_method_specific_views() -> None:
    """Keep method statistics in typed diagnostics and rendering in the visualization layer."""
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=_schedule(),
        architecture=_ClockArchitecture(),
        final_state=_ClockState(time=4),
    )

    for name in ("score", "explored_nodes", "visualize"):
        assert not hasattr(result, name)


def test_shared_layer_loads_no_compiler_or_rendering_code() -> None:
    """Import the shared layer and package root without a compiler or plotting backend."""
    command = (
        "import sys; "
        "import mqt.ionshuttler; "
        "import mqt.ionshuttler.core; "
        "loaded = [name for name in sys.modules if name.startswith(('mqt.ionshuttler.linear', 'matplotlib'))]; "
        "assert not loaded, loaded"
    )
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - Fixed interpreter command.
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
