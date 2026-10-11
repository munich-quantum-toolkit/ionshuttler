# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Records of the decisions that the Grid compiler made while it scheduled.

A planning trace holds one :class:`PlanningSnapshot` for each scheduling
decision. A snapshot shows what the compiler intended to do next: the gates it
selected, the targets it chose, and the ions it tried to move. The schedule
remains the only record of the actions that actually happened.

The trace does not store the routing search. It stores only the result of each
transport request, so its size grows with the number of decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar

from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.grid.actions import decode_grid_action

from .._json_utils import require_int, require_int_list, require_list, require_mapping, require_str

if TYPE_CHECKING:
    from mqt.ionshuttler.core.actions import Action

EnumT = TypeVar("EnumT", bound=StrEnum)


class MoveStatus(StrEnum):
    """State why a prioritized ion moves or stays in the next layer.

    Attributes:
        SELECTED: The proposed actions are part of the next layer.
        IN_GATE: The ion takes part in a gate that starts in this decision.
        AT_TARGET: The ion is already in its target segment.
        NO_ROUTE: No permitted junction crossing leads toward the target
            segment.
        NO_CANDIDATE: The router found no legal action for the next junction
            crossing.
        ALREADY_SELECTED: The next layer already contains a proposed action,
            selected for a higher-priority ion.
        CONFLICT: The proposed actions cannot run together with the actions
            already selected.
        DISPLACES_HIGHER_PRIORITY: The proposed actions move a higher-priority
            ion away from its target.
    """

    SELECTED = "selected"
    IN_GATE = "in_gate"
    AT_TARGET = "at_target"
    NO_ROUTE = "no_route"
    NO_CANDIDATE = "no_candidate"
    ALREADY_SELECTED = "already_selected"
    CONFLICT = "conflict"
    DISPLACES_HIGHER_PRIORITY = "displaces_higher_priority"


class IdleReason(StrEnum):
    """State why a decision selected no gate and no movement.

    Attributes:
        WAITING_FOR_RESOURCE: A busy ion, junction, or processing zone becomes
            free later. The schedule waits until then.
        NO_LEGAL_ACTION: No action is legal, and no busy resource becomes free
            later.
        NO_ROUTE: The bounded route search found no route to a processing zone.
    """

    WAITING_FOR_RESOURCE = "waiting_for_resource"
    NO_LEGAL_ACTION = "no_legal_action"
    NO_ROUTE = "no_route"


@dataclass(frozen=True, slots=True)
class PlannedGate:
    """Name a gate in a snapshot and the processing-zone segment chosen for it.

    Attributes:
        gate: The circuit gate. Its ``gate_id`` identifies it.
        target_segment: Segment of the processing zone that the compiler
            selected or locked for the gate. ``None`` for a virtual gate and
            for a gate without a reachable target.
    """

    gate: GateAction
    target_segment: str | None

    def __post_init__(self) -> None:
        """Validate the gate.

        Raises:
            TypeError: If the gate is not a gate action.
            ValueError: If the gate has no gate ID.
        """
        if not isinstance(self.gate, GateAction):
            msg = "gate must be a GateAction"
            raise TypeError(msg)
        if self.gate.gate_id is None:
            msg = "planned gates must have a gate ID"
            raise ValueError(msg)

    @property
    def gate_id(self) -> int:
        """The circuit gate ID."""
        gate_id = self.gate.gate_id
        assert gate_id is not None
        return gate_id

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of this gate."""
        return {"gate": self.gate.to_dict(), "target_segment": self.target_segment}

    @classmethod
    def from_dict(cls, data: object) -> PlannedGate:
        """Restore a planned gate.

        Returns:
            The restored gate.

        Raises:
            ValueError: If the data does not describe a gate.
        """
        mapping = require_mapping(data, "planned gate")
        gate = decode_grid_action(mapping.get("gate"))
        if not isinstance(gate, GateAction):
            msg = "planned gate must contain a gate action"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
        return cls(gate, _optional_str(mapping, "target_segment"))


@dataclass(frozen=True, slots=True)
class PrioritizedIon:
    """Describe the transport request for one prioritized ion.

    Attributes:
        ion: The ion.
        target_segment: Segment that the ion should reach.
        status: Why the ion moves or stays.
        actions: Actions that the router proposed for the ion. Empty when the
            router proposed nothing.
    """

    ion: int
    target_segment: str
    status: MoveStatus
    actions: tuple[Action, ...] = ()

    def __post_init__(self) -> None:
        """Validate the status and normalize the actions.

        Raises:
            TypeError: If the status is not a :class:`MoveStatus`.
        """
        if not isinstance(self.status, MoveStatus):
            msg = "status must be a MoveStatus"
            raise TypeError(msg)
        object.__setattr__(self, "actions", tuple(self.actions))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of this request."""
        return {
            "ion": self.ion,
            "target_segment": self.target_segment,
            "status": self.status.value,
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(cls, data: object) -> PrioritizedIon:
        """Restore a transport request.

        Returns:
            The restored request.
        """
        mapping = require_mapping(data, "prioritized ion")
        return cls(
            ion=require_int(mapping, "ion"),
            target_segment=require_str(mapping, "target_segment"),
            status=_enum_value(MoveStatus, require_str(mapping, "status"), "status"),
            actions=tuple(decode_grid_action(action) for action in require_list(mapping, "actions")),
        )


@dataclass(frozen=True, slots=True)
class PlanningSnapshot:
    """Record one scheduling decision of the Grid compiler.

    Gate fields contain circuit gate IDs. :attr:`gates` describes each
    selected, lookahead, and started gate once, in gate-ID order. Ready gates
    appear only by ID, so the size of a snapshot does not grow with the number
    of ready gates.

    Attributes:
        iteration: Scheduling iteration of the decision, starting at one.
        time: Schedule timestep at which the decision was made.
        ready_gates: Gates whose predecessors are complete and that have not
            started.
        selected_gates: Gates that the compiler selected from the ready gates.
        lookahead_gates: Gates in the short dependency horizon, in the order
            that ranks their ions. This includes the selected gates.
        started_gates: Gates that start in this decision.
        gates: The selected, lookahead, and started gates and their target
            segments.
        prioritized_ions: Transport requests in priority order.
        retained_ions: Ions that must stay in their processing zone.
        moves: Transport actions selected for the next layer. The
            breadth-first strategy selects a complete route.
        idle_reason: Why the decision selected no gate and no movement.
            ``None`` when it selected an action.
    """

    iteration: int
    time: int
    ready_gates: tuple[int, ...]
    selected_gates: tuple[int, ...]
    lookahead_gates: tuple[int, ...]
    started_gates: tuple[int, ...]
    gates: tuple[PlannedGate, ...]
    prioritized_ions: tuple[PrioritizedIon, ...]
    retained_ions: tuple[int, ...]
    moves: tuple[Action, ...]
    idle_reason: IdleReason | None = None

    def __post_init__(self) -> None:
        """Normalize the fields to tuples.

        Raises:
            TypeError: If the idle reason is not an :class:`IdleReason`.
        """
        for name in (
            "ready_gates",
            "selected_gates",
            "lookahead_gates",
            "started_gates",
            "gates",
            "prioritized_ions",
            "retained_ions",
            "moves",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.idle_reason is not None and not isinstance(self.idle_reason, IdleReason):
            msg = "idle_reason must be an IdleReason or None"
            raise TypeError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of this decision."""
        return {
            "iteration": self.iteration,
            "time": self.time,
            "ready_gates": list(self.ready_gates),
            "selected_gates": list(self.selected_gates),
            "lookahead_gates": list(self.lookahead_gates),
            "started_gates": list(self.started_gates),
            "gates": [gate.to_dict() for gate in self.gates],
            "prioritized_ions": [ion.to_dict() for ion in self.prioritized_ions],
            "retained_ions": list(self.retained_ions),
            "moves": [action.to_dict() for action in self.moves],
            "idle_reason": None if self.idle_reason is None else self.idle_reason.value,
        }

    @classmethod
    def from_dict(cls, data: object) -> PlanningSnapshot:
        """Restore one decision.

        Returns:
            The restored snapshot.
        """
        mapping = require_mapping(data, "planning snapshot")
        idle_reason = _optional_str(mapping, "idle_reason")
        return cls(
            iteration=require_int(mapping, "iteration"),
            time=require_int(mapping, "time"),
            ready_gates=tuple(require_int_list(mapping, "ready_gates")),
            selected_gates=tuple(require_int_list(mapping, "selected_gates")),
            lookahead_gates=tuple(require_int_list(mapping, "lookahead_gates")),
            started_gates=tuple(require_int_list(mapping, "started_gates")),
            gates=tuple(PlannedGate.from_dict(gate) for gate in require_list(mapping, "gates")),
            prioritized_ions=tuple(PrioritizedIon.from_dict(ion) for ion in require_list(mapping, "prioritized_ions")),
            retained_ions=tuple(require_int_list(mapping, "retained_ions")),
            moves=tuple(decode_grid_action(action) for action in require_list(mapping, "moves")),
            idle_reason=None if idle_reason is None else _enum_value(IdleReason, idle_reason, "idle_reason"),
        )


def _optional_str(mapping: dict[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is not None and not isinstance(value, str):
        msg = f"{key} must be a string or null"
        raise ValueError(msg)
    return value


def _enum_value(enum_type: type[EnumT], value: str, key: str) -> EnumT:
    try:
        return enum_type(value)
    except ValueError as error:
        msg = f"{key} has unknown value {value!r}"
        raise ValueError(msg) from error


__all__ = ["IdleReason", "MoveStatus", "PlannedGate", "PlanningSnapshot", "PrioritizedIon"]
