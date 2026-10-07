# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Linear machine state, explicit schedules, and schedule persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.linear.actions import Action, decode_linear_action
from mqt.ionshuttler.linear.state import AdvanceTime, State, advance_time

from .._json_utils import require_int, require_int_pairs, require_mapping, require_str_int_pairs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.linear.architecture import LinearArchitecture
    from mqt.ionshuttler.linear.state import SearchTransition


@dataclass(frozen=True)
class LinearMachineState:
    """Describes a Linear hardware state independent of compiler progress."""

    positions: tuple[tuple[int, int], ...]
    ions_busy_until: tuple[tuple[int, int], ...]
    pzs_busy_until: tuple[tuple[str, int], ...]
    time: int = 0

    def __post_init__(self) -> None:
        """Normalize and validate machine-state values.

        Raises:
            TypeError: If the machine clock has the wrong type.
            ValueError: If mappings, occupancy, or timestamps are inconsistent.
        """
        positions = tuple(sorted(self.positions))
        ions_busy_until = tuple(sorted(self.ions_busy_until))
        pzs_busy_until = tuple(sorted(self.pzs_busy_until))
        _require_unique_keys(positions, "positions")
        _require_unique_keys(ions_busy_until, "ions_busy_until")
        _require_unique_keys(pzs_busy_until, "pzs_busy_until")
        if len({site for _ion, site in positions}) != len(positions):
            msg = "positions must not contain duplicate occupied sites"
            raise ValueError(msg)
        if isinstance(self.time, bool) or not isinstance(self.time, int):
            msg = "time must be an integer"
            raise TypeError(msg)
        if self.time < 0:
            msg = "time must be non-negative"
            raise ValueError(msg)
        if {ion for ion, _site in positions} != {ion for ion, _time in ions_busy_until}:
            msg = "ions_busy_until must contain exactly the positioned ions"
            raise ValueError(msg)
        if any(free_time < self.time for _ion, free_time in ions_busy_until):
            msg = "ion availability times must not precede the machine time"
            raise ValueError(msg)
        if any(free_time < self.time for _zone, free_time in pzs_busy_until):
            msg = "processing-zone availability times must not precede the machine time"
            raise ValueError(msg)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "ions_busy_until", ions_busy_until)
        object.__setattr__(self, "pzs_busy_until", pzs_busy_until)

    @classmethod
    def from_compiler_state(cls, state: State) -> LinearMachineState:
        """Copy hardware fields from a Linear compiler state.

        Returns:
            The canonical machine state.
        """
        return cls(
            positions=state.positions,
            ions_busy_until=tuple((ion, max(free_time, state.time)) for ion, free_time in state.ions_busy_until),
            pzs_busy_until=tuple((zone, max(free_time, state.time)) for zone, free_time in state.pzs_busy_until),
            time=state.time,
        )

    def to_replay_state(self) -> State:
        """Return a Linear state with empty circuit progress."""
        return State(
            positions=self.positions,
            completed_gates=frozenset(),
            in_progress_gates=(),
            ions_busy_until=self.ions_busy_until,
            pzs_busy_until=self.pzs_busy_until,
            time=self.time,
        )

    def to_dict(self) -> dict[str, object]:
        """Return this machine state using JSON-compatible values."""
        return {
            "positions": [list(item) for item in self.positions],
            "ions_busy_until": [list(item) for item in self.ions_busy_until],
            "pzs_busy_until": [list(item) for item in self.pzs_busy_until],
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, data: object) -> LinearMachineState:
        """Restore a Linear machine state.

        Returns:
            The restored state.
        """
        mapping = require_mapping(data, "linear_machine_state")
        return cls(
            positions=tuple(require_int_pairs(mapping, "positions")),
            ions_busy_until=tuple(require_int_pairs(mapping, "ions_busy_until")),
            pzs_busy_until=tuple(require_str_int_pairs(mapping, "pzs_busy_until")),
            time=require_int(mapping, "time"),
        )


def schedule_from_path(
    path: Sequence[SearchTransition],
    initial_state: State | LinearMachineState,
    architecture: LinearArchitecture,
) -> Schedule[Action, LinearMachineState]:
    """Convert a Linear search path to the shared explicit timeline.

    Each :class:`~mqt.ionshuttler.linear.state.AdvanceTime` transition moves
    the clock forward and does not appear in the schedule. The architecture
    supplies every action's duration and processing zone.

    Returns:
        A schedule with explicit action start times.
    """
    machine_state = (
        initial_state
        if isinstance(initial_state, LinearMachineState)
        else LinearMachineState.from_compiler_state(initial_state)
    )
    replay_state = machine_state.to_replay_state()
    scheduled_actions: list[ScheduledAction[Action]] = []
    for transition in path:
        if isinstance(transition, AdvanceTime):
            replay_state = advance_time(replay_state)
            continue
        scheduled_actions.append(
            ScheduledAction(
                action_id=len(scheduled_actions),
                action=transition,
                start_time=replay_state.time,
                duration=architecture.action_duration(transition),
                processing_zone_id=architecture.action_processing_zone(transition, replay_state),
            )
        )
        replay_state = architecture.apply_action(replay_state, transition)
    end_time = max([replay_state.time, *(item.end_time for item in scheduled_actions)])
    return Schedule(tuple(scheduled_actions), end_time, machine_state)


def schedule_from_dict(data: object) -> Schedule[Action, LinearMachineState]:
    """Restore a Linear schedule from its versioned JSON representation.

    Returns:
        The restored schedule.
    """
    return Schedule.from_dict(
        data,
        decode_action=decode_linear_action,
        decode_state=LinearMachineState.from_dict,
    )


def schedule_from_json(raw: str) -> Schedule[Action, LinearMachineState]:
    """Restore a Linear schedule from JSON text.

    Returns:
        The restored schedule.
    """
    return schedule_from_dict(json.loads(raw))


def load_schedule(filename: str | Path) -> Schedule[Action, LinearMachineState]:
    """Load a Linear schedule from a UTF-8 JSON file.

    Returns:
        The restored schedule.
    """
    return schedule_from_json(Path(filename).read_text(encoding="utf-8"))


def _require_unique_keys(values: Sequence[tuple[object, object]], label: str) -> None:
    if len({key for key, _value in values}) != len(values):
        msg = f"{label} must not contain duplicate keys"
        raise ValueError(msg)


__all__ = [
    "LinearMachineState",
    "Schedule",
    "ScheduledAction",
    "load_schedule",
    "schedule_from_dict",
    "schedule_from_json",
    "schedule_from_path",
]
