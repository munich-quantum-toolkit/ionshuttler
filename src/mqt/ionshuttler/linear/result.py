# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Linear compiler diagnostics and result persistence."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.result import CompilationResult as _CompilationResult
from mqt.ionshuttler.linear.actions import decode_linear_action
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.schedule import LinearMachineState

from .._json_utils import require_int, require_int_list, require_list, require_mapping


@dataclass(frozen=True)
class GateGroup:
    """Record one gate group planned for a shared ion placement.

    The permutation strategy creates this record only for its diagnostics.
    The compiler chooses one target placement for the group. The schedule is
    the authoritative record of the actions and their execution order.

    Attributes:
        gate_ids: Circuit gate IDs of the group, in circuit order.
        zone_by_ion: Processing-zone assignment for each ion used by a physical
            gate in the group, ordered by ion.
        target_site_of_ion: Target site indexed by ion.
        swaps: Number of physical swaps needed for the target placement.
        shuttles: Number of shuttles needed for the target placement.
        start_time: Start of the first move or gate of the group.
        end_time: End of the last move or gate of the group.
    """

    gate_ids: tuple[int, ...]
    zone_by_ion: tuple[tuple[int, str], ...]
    target_site_of_ion: tuple[int, ...]
    swaps: int
    shuttles: int
    start_time: int
    end_time: int

    def __post_init__(self) -> None:
        """Validate the placement, counts, and times.

        Raises:
            TypeError: If an identifier or placement entry is not an integer.
            ValueError: If an identifier or site is invalid, repeated, or
                missing, or if the group ends before it starts.
        """
        gate_ids = tuple(self.gate_ids)
        if not gate_ids:
            msg = "a gate group must contain at least one gate"
            raise ValueError(msg)
        for gate_id in gate_ids:
            _require_count(gate_id, "gate IDs")
        if len(set(gate_ids)) != len(gate_ids):
            msg = "gate IDs must be unique"
            raise ValueError(msg)

        zone_by_ion = tuple(self.zone_by_ion)
        assigned_ions: set[int] = set()
        for ion, processing_zone_id in zone_by_ion:
            _require_count(ion, "assigned ions")
            if ion in assigned_ions:
                msg = "assigned ions must be unique"
                raise ValueError(msg)
            if not isinstance(processing_zone_id, str):
                msg = "processing-zone IDs must be strings"
                raise TypeError(msg)
            if not processing_zone_id:
                msg = "processing-zone IDs must be non-empty"
                raise ValueError(msg)
            assigned_ions.add(ion)

        target_site_of_ion = tuple(self.target_site_of_ion)
        if not target_site_of_ion:
            msg = "target placement must contain at least one ion"
            raise ValueError(msg)
        for site in target_site_of_ion:
            _require_count(site, "target sites")
        if len(set(target_site_of_ion)) != len(target_site_of_ion):
            msg = "target sites must be unique"
            raise ValueError(msg)

        object.__setattr__(self, "gate_ids", gate_ids)
        object.__setattr__(self, "zone_by_ion", zone_by_ion)
        object.__setattr__(self, "target_site_of_ion", target_site_of_ion)
        for name in ("swaps", "shuttles", "start_time", "end_time"):
            _require_count(getattr(self, name), name)
        if self.end_time < self.start_time:
            msg = "a gate group must not end before it starts"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of this group."""
        return {
            "gate_ids": list(self.gate_ids),
            "zone_by_ion": [list(entry) for entry in self.zone_by_ion],
            "target_site_of_ion": list(self.target_site_of_ion),
            "swaps": self.swaps,
            "shuttles": self.shuttles,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, data: object) -> GateGroup:
        """Restore one gate group.

        Returns:
            The restored group.

        Raises:
            ValueError: If an ion-zone entry is malformed.
        """
        mapping = require_mapping(data, "gate group")
        zone_by_ion: list[tuple[int, str]] = []
        for entry in require_list(mapping, "zone_by_ion"):
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or isinstance(entry[0], bool)
                or not isinstance(entry[0], int)
                or not isinstance(entry[1], str)
            ):
                msg = "zone_by_ion entries must be [ion, processing_zone_id] pairs"
                raise ValueError(msg)
            zone_by_ion.append((entry[0], entry[1]))
        return cls(
            gate_ids=tuple(require_int_list(mapping, "gate_ids")),
            zone_by_ion=tuple(zone_by_ion),
            target_site_of_ion=tuple(require_int_list(mapping, "target_site_of_ion")),
            swaps=require_int(mapping, "swaps"),
            shuttles=require_int(mapping, "shuttles"),
            start_time=require_int(mapping, "start_time"),
            end_time=require_int(mapping, "end_time"),
        )


@dataclass(frozen=True, kw_only=True)
class LinearDiagnostics(ABC):
    """Store statistics shared by all Linear compiler strategies.

    Attributes:
        score: Schedule time of the final compiler state.
        preferred_gate_zones: Processing zone that pre-partitioning preferred
            for each gate that needs one. The schedule may use another zone.
        makespan_lower_bound: No complete schedule can be shorter. The bound
            combines dependency-chain time, processing-zone load, and the load
            on zones that support two-ion gates. It ignores transport.
    """

    score: int
    preferred_gate_zones: tuple[tuple[int, str], ...] = ()
    makespan_lower_bound: int = 0

    def __post_init__(self) -> None:
        """Validate the shared statistics.

        Raises:
            TypeError: If a statistic or assignment has the wrong type.
            ValueError: If a statistic or identifier is invalid.
        """
        _require_count(self.score, "score")
        _require_count(self.makespan_lower_bound, "makespan_lower_bound")
        preferred_gate_zones = tuple(sorted(self.preferred_gate_zones))
        gate_ids: set[int] = set()
        for gate_id, processing_zone_id in preferred_gate_zones:
            if isinstance(gate_id, bool) or not isinstance(gate_id, int):
                msg = "preferred gate IDs must be integers"
                raise TypeError(msg)
            if gate_id < 0:
                msg = "preferred gate IDs must be non-negative"
                raise ValueError(msg)
            if not isinstance(processing_zone_id, str):
                msg = "preferred processing-zone IDs must be strings"
                raise TypeError(msg)
            if not processing_zone_id:
                msg = "preferred processing-zone IDs must be non-empty"
                raise ValueError(msg)
            if gate_id in gate_ids:
                msg = "preferred gate IDs must be unique"
                raise ValueError(msg)
            gate_ids.add(gate_id)
        object.__setattr__(self, "preferred_gate_zones", preferred_gate_zones)

    def _common_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "preferred_gate_zones": [list(assignment) for assignment in self.preferred_gate_zones],
            "makespan_lower_bound": self.makespan_lower_bound,
        }

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """Return these diagnostics using JSON-compatible values."""

    def _common_statistics(self) -> tuple[tuple[str, int | float | str], ...]:
        return (("Score", self.score), ("Makespan lower bound", self.makespan_lower_bound))

    @abstractmethod
    def statistics(self) -> tuple[tuple[str, int | float | str], ...]:
        """Return the run statistics as ``(label, value)`` pairs in display order."""

    @classmethod
    def from_dict(cls, data: object) -> LinearDiagnostics:
        """Restore diagnostics for the recorded Linear compiler strategy.

        Returns:
            Search or permutation diagnostics.

        Raises:
            ValueError: If a diagnostic field is malformed.
        """
        mapping = require_mapping(data, "linear diagnostics")
        common = {
            "score": require_int(mapping, "score"),
            "preferred_gate_zones": _preferred_gate_zones_from_dict(mapping),
            "makespan_lower_bound": (
                require_int(mapping, "makespan_lower_bound") if "makespan_lower_bound" in mapping else 0
            ),
        }
        raw_permutation = mapping.get("permutation")
        explored_nodes = require_int(mapping, "explored_nodes")
        if raw_permutation is None:
            return SearchDiagnostics(explored_nodes=explored_nodes, **common)
        if explored_nodes != 0:
            msg = "permutation diagnostics must have explored_nodes=0"
            raise ValueError(msg)
        permutation = require_mapping(raw_permutation, "permutation diagnostics")
        return PermutationDiagnostics(
            groups=require_int(permutation, "groups"),
            groups_without_moves=require_int(permutation, "groups_without_moves"),
            gates_in_preferred_zone=require_int(permutation, "gates_in_preferred_zone"),
            planning_trace=(
                None
                if permutation.get("planning_trace") is None
                else tuple(GateGroup.from_dict(group) for group in require_list(permutation, "planning_trace"))
            ),
            **common,
        )


@dataclass(frozen=True, kw_only=True)
class SearchDiagnostics(LinearDiagnostics):
    """Store statistics of the search-based compilation strategy.

    Attributes:
        explored_nodes: Number of states that the strategy explored.
    """

    explored_nodes: int

    def __post_init__(self) -> None:
        """Validate the search statistics."""
        super().__post_init__()
        _require_count(self.explored_nodes, "explored_nodes")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible search diagnostics."""
        return {**self._common_dict(), "explored_nodes": self.explored_nodes, "permutation": None}

    def statistics(self) -> tuple[tuple[str, int | float | str], ...]:
        """Return the score, the makespan lower bound, and the number of explored nodes."""
        return (*self._common_statistics(), ("Explored nodes", self.explored_nodes))


@dataclass(frozen=True, kw_only=True)
class PermutationDiagnostics(LinearDiagnostics):
    """Store statistics of the permutation-based compilation strategy.

    Attributes:
        groups: Number of gate groups.
        groups_without_moves: Number of groups whose target placement required
            no transport.
        gates_in_preferred_zone: Number of gates that ran in the processing
            zone that pre-partitioning preferred for them.
        planning_trace: One record per group, in planning order. Consecutive
            records can overlap in time. ``None``
            unless :attr:`~mqt.ionshuttler.linear.PermutationConfig.record_planning_trace`
            is set.
    """

    groups: int
    groups_without_moves: int
    gates_in_preferred_zone: int
    planning_trace: tuple[GateGroup, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the counters and the trace.

        Raises:
            TypeError: If the trace contains another value.
            ValueError: If the counters or trace length are inconsistent.
        """
        super().__post_init__()
        for name in ("groups", "groups_without_moves", "gates_in_preferred_zone"):
            _require_count(getattr(self, name), name)
        if self.groups_without_moves > self.groups:
            msg = "groups_without_moves must be <= groups"
            raise ValueError(msg)
        if self.planning_trace is not None:
            trace = tuple(self.planning_trace)
            if any(not isinstance(group, GateGroup) for group in trace):
                msg = "planning_trace must contain GateGroup values"
                raise TypeError(msg)
            if len(trace) != self.groups:
                msg = "planning_trace must contain one entry per group"
                raise ValueError(msg)
            object.__setattr__(self, "planning_trace", trace)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible permutation diagnostics."""
        permutation: dict[str, object] = {
            "groups": self.groups,
            "groups_without_moves": self.groups_without_moves,
            "gates_in_preferred_zone": self.gates_in_preferred_zone,
        }
        if self.planning_trace is not None:
            permutation["planning_trace"] = [group.to_dict() for group in self.planning_trace]
        return {**self._common_dict(), "explored_nodes": 0, "permutation": permutation}

    def statistics(self) -> tuple[tuple[str, int | float | str], ...]:
        """Return the score, the makespan lower bound, and the gate-group counters."""
        return (
            *self._common_statistics(),
            ("Gate groups", self.groups),
            ("Groups without moves", self.groups_without_moves),
            ("Gates in preferred zone", self.gates_in_preferred_zone),
        )


def _preferred_gate_zones_from_dict(mapping: dict[str, object]) -> tuple[tuple[int, str], ...]:
    preferred_gate_zones: list[tuple[int, str]] = []
    for raw_assignment in require_list(mapping, "preferred_gate_zones"):
        if not isinstance(raw_assignment, list) or len(raw_assignment) != 2:
            msg = "preferred_gate_zones entries must be [gate_id, processing_zone_id] pairs"
            raise ValueError(msg)
        gate_id, processing_zone_id = raw_assignment
        if isinstance(gate_id, bool) or not isinstance(gate_id, int):
            msg = "preferred gate IDs must be integers"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
        if not isinstance(processing_zone_id, str):
            msg = "preferred processing-zone IDs must be strings"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] - JSON format errors use ValueError.
        preferred_gate_zones.append((gate_id, processing_zone_id))
    return tuple(preferred_gate_zones)


def _require_count(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if value < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


LinearCompilationResult: TypeAlias = _CompilationResult[
    LinearArchitecture, Action, LinearMachineState, LinearDiagnostics
]


def result_from_dict(data: object) -> LinearCompilationResult:
    """Restore a Linear compilation result from its versioned JSON representation.

    Returns:
        The restored compilation result.
    """
    return _CompilationResult.from_dict(
        data,
        decode_architecture=LinearArchitecture.from_dict,
        decode_action=decode_linear_action,
        decode_state=LinearMachineState.from_dict,
        decode_diagnostics=LinearDiagnostics.from_dict,
    )


def result_from_json(raw: str) -> LinearCompilationResult:
    """Restore a Linear compilation result from JSON text.

    Returns:
        The restored compilation result.
    """
    return result_from_dict(json.loads(raw))


def load_result(filename: str | Path) -> LinearCompilationResult:
    """Load a Linear compilation result from a UTF-8 JSON file.

    Returns:
        The restored compilation result.
    """
    return result_from_json(Path(filename).read_text(encoding="utf-8"))


__all__ = [
    "GateGroup",
    "LinearCompilationResult",
    "LinearDiagnostics",
    "PermutationDiagnostics",
    "SearchDiagnostics",
    "load_result",
    "result_from_dict",
    "result_from_json",
]
