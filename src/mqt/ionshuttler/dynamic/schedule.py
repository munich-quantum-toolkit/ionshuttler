# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Schedules with measurement-dependent branches and loops."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeAlias

from mqt.ionshuttler.core.gates import Measure
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction

from .._json_utils import require_int, require_list, require_mapping, require_str
from .program import Condition, check_repetition, enumerate_paths, outcome_depths, outcome_key, walk_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from mqt.ionshuttler.core.actions import Action

    from .program import OutcomeKey


@dataclass(frozen=True)
class ScheduledBranch:
    """Hold the precompiled schedules for both sides of one branch."""

    condition: Condition
    then_steps: tuple[ScheduleStep, ...]
    else_steps: tuple[ScheduleStep, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the step sequences."""
        object.__setattr__(self, "then_steps", _normalize_steps(self.then_steps))
        object.__setattr__(self, "else_steps", _normalize_steps(self.else_steps))


@dataclass(frozen=True)
class ScheduledLoop:
    """Hold the precompiled schedules of one loop body.

    The body ends in the machine state in which it starts, so every iteration
    runs the same schedules. ``count``, ``entry``, and ``repeat`` mean the same
    as in :class:`~mqt.ionshuttler.dynamic.program.Loop`.
    """

    steps: tuple[ScheduleStep, ...]
    count: int | None = None
    entry: Condition | None = None
    repeat: Condition | None = None

    def __post_init__(self) -> None:
        """Normalize the steps and validate how the loop repeats."""
        object.__setattr__(self, "steps", _normalize_steps(self.steps))
        check_repetition(self.count, self.entry, self.repeat)


ScheduleStep: TypeAlias = "Schedule[Any, Any] | ScheduledBranch | ScheduledLoop"


@dataclass(frozen=True)
class TimedSchedule:
    """Place one precompiled schedule on the timeline of one path."""

    schedule: Schedule[Any, Any]
    start_time: int


@dataclass(frozen=True)
class BranchingSchedule:
    """Store precompiled schedules for every measurement-dependent path.

    Each schedule starts at time 0 from an idle machine state, and steps run in
    order. A schedule starts when the previous one ends. The steps after a
    branch or loop decision start no earlier than ``feedback_delay`` after the
    end of every measurement that the decision reads. ``initial_state`` is the
    machine state before the first step; it must provide ``time``,
    ``at_time``, and ``to_dict`` like the Grid machine state.
    """

    steps: tuple[ScheduleStep, ...]
    initial_state: Any
    feedback_delay: int = 0

    def __post_init__(self) -> None:
        """Validate the steps and the feedback delay.

        Raises:
            TypeError: If a step or the delay has the wrong type.
            ValueError: If the delay is negative.
        """
        object.__setattr__(self, "steps", _normalize_steps(self.steps))
        if isinstance(self.feedback_delay, bool) or not isinstance(self.feedback_delay, int):
            msg = "feedback_delay must be an integer"
            raise TypeError(msg)
        if self.feedback_delay < 0:
            msg = "feedback_delay must be non-negative"
            raise ValueError(msg)

    def paths(self, *, max_iterations: int | None = None) -> Iterator[dict[OutcomeKey, int]]:
        """Yield one assignment of the deciding outcomes for each distinct path.

        A loop without a fixed count needs ``max_iterations``; paths with more
        iterations are left out.
        """
        yield from enumerate_paths(self.steps, max_iterations=max_iterations)

    def timeline_for(self, values: Mapping[OutcomeKey, int]) -> tuple[TimedSchedule, ...]:
        """Return the schedules that run for the given measurement outcomes, with their start times."""
        timeline: list[TimedSchedule] = []
        measurement_ends: dict[OutcomeKey, int] = {}
        depths = outcome_depths(self.steps)
        time = self.initial_state.time

        def place(step: object, iterations: tuple[int, ...]) -> None:
            nonlocal time
            assert isinstance(step, Schedule)
            offset = time - step.start_time
            timeline.append(TimedSchedule(step, time))
            for item in step.scheduled_actions:
                if isinstance(item.action, Measure) and item.action.outcome is not None:
                    key = outcome_key(item.action.outcome, depths[item.action.outcome], iterations)
                    measurement_ends[key] = item.end_time + offset
            time = step.end_time + offset

        def wait_for(keys: tuple[OutcomeKey, ...]) -> None:
            nonlocal time
            for key in keys:
                time = max(time, measurement_ends[key] + self.feedback_delay)

        walk_path(self.steps, values, depths, on_step=place, on_decision=wait_for)
        return tuple(timeline)

    def schedule_for(self, values: Mapping[OutcomeKey, int]) -> Schedule[Any, Any]:
        """Return the flat schedule that runs for the given measurement outcomes."""
        actions: list[ScheduledAction[Any]] = []
        end_time = self.initial_state.time
        for timed in self.timeline_for(values):
            offset = timed.start_time - timed.schedule.start_time
            first_id = len(actions)
            actions.extend(
                replace(item, action_id=first_id + position, start_time=item.start_time + offset)
                for position, item in enumerate(timed.schedule.scheduled_actions)
            )
            end_time = timed.schedule.end_time + offset
        return Schedule(tuple(actions), end_time, self.initial_state)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible branching-schedule data."""
        return {
            "feedback_delay": self.feedback_delay,
            "initial_state": self.initial_state.to_dict(),
            "steps": [_step_to_dict(step) for step in self.steps],
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        decode_action: Callable[[Mapping[str, object]], Action],
        decode_state: Callable[[object], Any],
    ) -> BranchingSchedule:
        """Restore a branching schedule with the architecture's decoders.

        Returns:
            The restored branching schedule.
        """
        mapping = require_mapping(data, "branching schedule")
        return cls(
            steps=tuple(_step_from_dict(item, decode_action, decode_state) for item in require_list(mapping, "steps")),
            initial_state=decode_state(mapping.get("initial_state")),
            feedback_delay=require_int(mapping, "feedback_delay"),
        )


def _normalize_steps(steps: Iterable[object]) -> tuple[ScheduleStep, ...]:
    normalized: list[ScheduleStep] = []
    for step in steps:
        if not isinstance(step, Schedule | ScheduledBranch | ScheduledLoop):
            msg = "schedule steps must be Schedule, ScheduledBranch, or ScheduledLoop values"
            raise TypeError(msg)
        normalized.append(step)
    return tuple(normalized)


def _step_to_dict(step: ScheduleStep) -> dict[str, object]:
    if isinstance(step, Schedule):
        return {"type": "schedule", "schedule": step.to_dict()}
    if isinstance(step, ScheduledBranch):
        return {
            "type": "branch",
            "condition": step.condition.to_dict(),
            "then_steps": [_step_to_dict(item) for item in step.then_steps],
            "else_steps": [_step_to_dict(item) for item in step.else_steps],
        }
    return {
        "type": "loop",
        "count": step.count,
        "entry": None if step.entry is None else step.entry.to_dict(),
        "repeat": None if step.repeat is None else step.repeat.to_dict(),
        "steps": [_step_to_dict(item) for item in step.steps],
    }


def _step_from_dict(
    data: object,
    decode_action: Callable[[Mapping[str, object]], Action],
    decode_state: Callable[[object], Any],
) -> ScheduleStep:
    mapping = require_mapping(data, "schedule step")
    kind = require_str(mapping, "type")
    if kind == "schedule":
        return Schedule.from_dict(mapping.get("schedule"), decode_action=decode_action, decode_state=decode_state)
    if kind == "branch":
        return ScheduledBranch(
            condition=Condition.from_dict(mapping.get("condition")),
            then_steps=tuple(
                _step_from_dict(item, decode_action, decode_state) for item in require_list(mapping, "then_steps")
            ),
            else_steps=tuple(
                _step_from_dict(item, decode_action, decode_state) for item in require_list(mapping, "else_steps")
            ),
        )
    if kind == "loop":
        count = mapping.get("count")
        if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
            msg = "loop count must be an integer or null"
            raise ValueError(msg)
        entry = mapping.get("entry")
        repeat = mapping.get("repeat")
        return ScheduledLoop(
            steps=tuple(_step_from_dict(item, decode_action, decode_state) for item in require_list(mapping, "steps")),
            count=count,
            entry=None if entry is None else Condition.from_dict(entry),
            repeat=None if repeat is None else Condition.from_dict(repeat),
        )
    msg = f"unknown schedule step type: {kind}"
    raise ValueError(msg)


__all__ = ["BranchingSchedule", "ScheduleStep", "ScheduledBranch", "ScheduledLoop", "TimedSchedule"]
