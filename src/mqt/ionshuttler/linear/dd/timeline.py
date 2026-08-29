# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reconstruct timed hardware state from a compiled Linear schedule."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast

from mqt.ionshuttler.linear.actions import (
    Action,
    AdvanceTime,
    PhysicalSwap,
    Shuttle,
    SingleQubitGate,
    TwoQubitGate,
)
from mqt.ionshuttler.linear.state import State, to_dict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.schedule import ActionSchedule, ScheduledAction


class _TimedGate(Protocol):
    """Describe the duration field shared by concrete physical gates."""

    duration: int


@dataclass(frozen=True)
class CompiledTimeline:
    """Provide schedule-boundary positions, resource occupancy, and actions."""

    makespan: int
    _positions_by_time: tuple[Mapping[int, int], ...]
    _ion_busy_by_time: Mapping[int, frozenset[int]]
    _ion_gate_busy_by_time: Mapping[int, frozenset[int]]
    _pz_busy_by_time: Mapping[str, frozenset[int]]
    _actions_by_time: Mapping[int, tuple[Action, ...]]
    _scheduled_actions_by_time: Mapping[int, tuple[ScheduledAction, ...]]

    def ion_position(self, ion_id: int, timestep: int) -> int:
        """Return an ion's site at one schedule boundary."""
        return self._positions_by_time[_validate_time(timestep, self.makespan)][ion_id]

    def ion_busy(self, ion_id: int, timestep: int) -> bool:
        """Return whether an ion is occupied by any physical action."""
        time = _validate_time(timestep, self.makespan)
        return time in self._ion_busy_by_time.get(ion_id, frozenset())

    def ion_gate_busy(self, ion_id: int, timestep: int) -> bool:
        """Return whether an ion is occupied specifically by a gate."""
        time = _validate_time(timestep, self.makespan)
        return time in self._ion_gate_busy_by_time.get(ion_id, frozenset())

    def pz_busy(self, zone_name: str, timestep: int) -> bool:
        """Return whether a processing zone is occupied by a gate."""
        time = _validate_time(timestep, self.makespan)
        return time in self._pz_busy_by_time.get(zone_name, frozenset())

    def position_occupied_by_other(self, ion_id: int, site: int, timestep: int) -> bool:
        """Return whether another ion occupies a site at a boundary."""
        positions = self._positions_by_time[_validate_time(timestep, self.makespan)]
        return any(position == site for ion, position in positions.items() if ion != ion_id)

    def action_at(self, timestep: int) -> tuple[Action, ...] | None:
        """Return the ordered actions at a schedule boundary, if any."""
        return self._actions_by_time.get(_validate_time(timestep, self.makespan))

    def scheduled_action_at(self, timestep: int) -> tuple[ScheduledAction, ...] | None:
        """Return actions with stable identity at one boundary."""
        return self._scheduled_actions_by_time.get(_validate_time(timestep, self.makespan))

    def state_at(self, timestep: int) -> State:
        """Reconstruct the hardware portion of state at a schedule boundary.

        Returns:
            The reconstructed hardware state.
        """
        current_time = _validate_time(timestep, self.makespan)
        positions = tuple(sorted(self._positions_by_time[current_time].items()))
        ions_busy_until = tuple(
            sorted(
                (
                    ion_id,
                    current_time + 1 if self.ion_busy(ion_id, current_time) else current_time,
                )
                for ion_id in self._positions_by_time[current_time]
            )
        )
        pzs_busy_until = tuple(
            sorted(
                (
                    zone_name,
                    current_time + 1 if self.pz_busy(zone_name, current_time) else current_time,
                )
                for zone_name in self._pz_busy_by_time
            )
        )
        return State(
            positions=positions,
            completed_gates=frozenset(),
            in_progress_gates=(),
            ions_busy_until=ions_busy_until,
            pzs_busy_until=pzs_busy_until,
            time=current_time,
        )

    def with_inserted_single_qubit_gate(
        self,
        scheduled_action: ScheduledAction,
        zone: str,
        timestep: int,
    ) -> CompiledTimeline:
        """Return a coherent timeline containing one additional physical gate.

        Returns:
            A timeline with updated resource and ordered-action indices.

        Raises:
            ValueError: If the action is not a physical single-qubit gate or
                would finish beyond the timeline.
        """
        action = scheduled_action.action
        if not isinstance(action, SingleQubitGate) or action.virtual:
            msg = "incremental timeline insertion requires a physical single-qubit gate"
            raise ValueError(msg)
        start = _validate_time(timestep, self.makespan)
        duration = _action_duration(action)
        if start + duration > self.makespan:
            msg = "inserted gate must finish within the timeline"
            raise ValueError(msg)

        occupied = frozenset(range(start, start + duration))
        ion_busy = dict(self._ion_busy_by_time)
        ion_busy[action.ion] = ion_busy.get(action.ion, frozenset()) | occupied
        ion_gate_busy = dict(self._ion_gate_busy_by_time)
        ion_gate_busy[action.ion] = ion_gate_busy.get(action.ion, frozenset()) | occupied
        pz_busy = dict(self._pz_busy_by_time)
        pz_busy[zone] = pz_busy.get(zone, frozenset()) | occupied

        actions = dict(self._actions_by_time)
        actions[start] = _insert_action_before_time_advance(actions.get(start, ()), action)
        scheduled_actions = dict(self._scheduled_actions_by_time)
        scheduled_actions[start] = _insert_scheduled_action_before_time_advance(
            scheduled_actions.get(start, ()), scheduled_action
        )
        return replace(
            self,
            _ion_busy_by_time=MappingProxyType(ion_busy),
            _ion_gate_busy_by_time=MappingProxyType(ion_gate_busy),
            _pz_busy_by_time=MappingProxyType(pz_busy),
            _actions_by_time=MappingProxyType(actions),
            _scheduled_actions_by_time=MappingProxyType(scheduled_actions),
        )


def build_timeline(
    schedule: ActionSchedule,
    architecture: Architecture,
) -> CompiledTimeline:
    """Reconstruct a timeline solely from executable schedule information.

    Args:
        schedule: Executable schedule to reconstruct.
        architecture: Hardware model against which to interpret the actions.

    Returns:
        The reconstructed immutable timeline.
    """
    initial_positions = to_dict(schedule.initial_state.to_replay_state())
    makespan = schedule.num_timesteps
    initial_time = schedule.initial_state.time
    position_checkpoints: list[tuple[int, dict[int, int]]] = []
    ion_busy_by_time: dict[int, set[int]] = {
        ion_id: set(range(max(0, free_time - initial_time)))
        for ion_id, free_time in schedule.initial_state.ions_busy_until
    }
    ion_gate_busy_by_time: dict[int, set[int]] = {ion_id: set() for ion_id in initial_positions}
    initial_pz_availability = dict(schedule.initial_state.pzs_busy_until)
    pz_busy_by_time: dict[str, set[int]] = {
        zone_name: set(range(max(0, initial_pz_availability.get(zone_name, initial_time) - initial_time)))
        for zone_name in (architecture.processing_zones or {})
    }
    actions_by_time: dict[int, list[Action]] = {}
    scheduled_actions_by_time: dict[int, list[ScheduledAction]] = {}
    current_positions = dict(initial_positions)
    current_time = 0

    for scheduled_action in schedule.scheduled_actions:
        action = scheduled_action.action
        actions_by_time.setdefault(current_time, []).append(action)
        scheduled_actions_by_time.setdefault(current_time, []).append(scheduled_action)

        if isinstance(action, Shuttle):
            current_positions[action.ion] = action.dst
            _mark_busy_range(ion_busy_by_time.setdefault(action.ion, set()), current_time, action.duration)
            _record_positions(position_checkpoints, current_positions, current_time)
        elif isinstance(action, PhysicalSwap):
            current_positions[action.ion_a], current_positions[action.ion_b] = (
                current_positions[action.ion_b],
                current_positions[action.ion_a],
            )
            _mark_busy_range(ion_busy_by_time.setdefault(action.ion_a, set()), current_time, action.duration)
            _mark_busy_range(ion_busy_by_time.setdefault(action.ion_b, set()), current_time, action.duration)
            _record_positions(position_checkpoints, current_positions, current_time)
        elif isinstance(action, SingleQubitGate):
            if not action.virtual:
                _mark_gate_resources(
                    action.ion,
                    _action_duration(action),
                    current_time,
                    current_positions,
                    architecture,
                    ion_busy_by_time,
                    ion_gate_busy_by_time,
                    pz_busy_by_time,
                )
        elif isinstance(action, TwoQubitGate):
            for ion in (action.ion_a, action.ion_b):
                duration = _action_duration(action)
                _mark_busy_range(ion_busy_by_time.setdefault(ion, set()), current_time, duration)
                _mark_busy_range(ion_gate_busy_by_time.setdefault(ion, set()), current_time, duration)
            zone = architecture.get_processing_zone(current_positions[action.ion_a])
            if zone is not None:
                _mark_busy_range(pz_busy_by_time.setdefault(zone, set()), current_time, duration)
        elif isinstance(action, AdvanceTime):
            current_time += action.timestep_increment

    return CompiledTimeline(
        makespan=makespan,
        _positions_by_time=_materialize_positions(position_checkpoints, initial_positions, makespan),
        _ion_busy_by_time=MappingProxyType({ion_id: frozenset(times) for ion_id, times in ion_busy_by_time.items()}),
        _ion_gate_busy_by_time=MappingProxyType({
            ion_id: frozenset(times) for ion_id, times in ion_gate_busy_by_time.items()
        }),
        _pz_busy_by_time=MappingProxyType({zone: frozenset(times) for zone, times in pz_busy_by_time.items()}),
        _actions_by_time=MappingProxyType({timestep: tuple(actions) for timestep, actions in actions_by_time.items()}),
        _scheduled_actions_by_time=MappingProxyType({
            timestep: tuple(actions) for timestep, actions in scheduled_actions_by_time.items()
        }),
    )


def _mark_gate_resources(
    ion: int,
    duration: int,
    current_time: int,
    current_positions: dict[int, int],
    architecture: Architecture,
    ion_busy_by_time: dict[int, set[int]],
    ion_gate_busy_by_time: dict[int, set[int]],
    pz_busy_by_time: dict[str, set[int]],
) -> None:
    _mark_busy_range(ion_busy_by_time.setdefault(ion, set()), current_time, duration)
    _mark_busy_range(ion_gate_busy_by_time.setdefault(ion, set()), current_time, duration)
    zone = architecture.get_processing_zone(current_positions[ion])
    if zone is not None:
        _mark_busy_range(pz_busy_by_time.setdefault(zone, set()), current_time, duration)


def _action_duration(action: SingleQubitGate | TwoQubitGate) -> int:
    return cast("_TimedGate", action).duration


def _validate_time(timestep: int, makespan: int) -> int:
    if isinstance(timestep, bool) or not isinstance(timestep, int):
        msg = "timestep must be an integer"
        raise TypeError(msg)
    if not 0 <= timestep <= makespan:
        msg = f"timestep must be within [0, {makespan}]"
        raise ValueError(msg)
    return timestep


def _mark_busy_range(target: set[int], start: int, duration: int) -> None:
    target.update(range(start, start + duration))


def _record_positions(
    checkpoints: list[tuple[int, dict[int, int]]],
    current_positions: dict[int, int],
    current_time: int,
) -> None:
    snapshot = dict(current_positions)
    if checkpoints and checkpoints[-1][0] == current_time:
        checkpoints[-1] = (current_time, snapshot)
    else:
        checkpoints.append((current_time, snapshot))


def _materialize_positions(
    checkpoints: list[tuple[int, dict[int, int]]],
    initial_positions: dict[int, int],
    makespan: int,
) -> tuple[Mapping[int, int], ...]:
    positions_by_time: list[Mapping[int, int]] = []
    current_positions: Mapping[int, int] = MappingProxyType(dict(initial_positions))
    checkpoint_index = 0
    for timestep in range(makespan + 1):
        while checkpoint_index < len(checkpoints) and checkpoints[checkpoint_index][0] <= timestep:
            current_positions = MappingProxyType(checkpoints[checkpoint_index][1])
            checkpoint_index += 1
        positions_by_time.append(current_positions)
    return tuple(positions_by_time)


def _insert_action_before_time_advance(items: tuple[Action, ...], action: Action) -> tuple[Action, ...]:
    insert_index = next(
        (index for index, existing in enumerate(items) if isinstance(existing, AdvanceTime)),
        len(items),
    )
    return (*items[:insert_index], action, *items[insert_index:])


def _insert_scheduled_action_before_time_advance(
    items: tuple[ScheduledAction, ...],
    scheduled_action: ScheduledAction,
) -> tuple[ScheduledAction, ...]:
    insert_index = next(
        (index for index, existing in enumerate(items) if isinstance(existing.action, AdvanceTime)),
        len(items),
    )
    return (*items[:insert_index], scheduled_action, *items[insert_index:])


__all__ = ["CompiledTimeline", "build_timeline"]
