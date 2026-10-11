# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Settings for Grid compilation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.ionshuttler.partitioning import FineGrainedTabuConfig


class GridCompilerStrategy(StrEnum):
    """Select how the compiler chooses transport actions."""

    BREADTH_FIRST = "breadth_first"
    GREEDY = "greedy"


class GridDependencyMode(StrEnum):
    """Select which circuit dependencies constrain Grid scheduling."""

    CONSERVATIVE = "conservative"
    COMMUTATION_AWARE = "commutation_aware"


@dataclass(frozen=True)
class GridCompilerConfig:
    """Bound the deterministic Grid compiler.

    ``dependency_mode`` controls whether commuting gates can change order.
    ``max_iterations`` bounds scheduling progress. ``max_routing_states``
    bounds each breadth-first search or greedy candidate evaluation.
    ``allowed_junction_crossings`` optionally limits routing to directed segment
    pairs without changing the hardware topology. ``record_planning_trace``
    stores one :class:`~mqt.ionshuttler.grid.planning_trace.PlanningSnapshot`
    per scheduling decision in the result diagnostics. Use it to debug failed
    or slow compilations; it costs memory.
    """

    strategy: GridCompilerStrategy = GridCompilerStrategy.GREEDY
    dependency_mode: GridDependencyMode = GridDependencyMode.COMMUTATION_AWARE
    pre_partition_config: FineGrainedTabuConfig | None = None
    allowed_junction_crossings: frozenset[tuple[str, str]] | None = None
    max_iterations: int = 10_000
    max_routing_states: int = 10_000
    greedy_lookahead_rounds: int = 10
    record_planning_trace: bool = False

    def __post_init__(self) -> None:
        """Validate compiler settings.

        Raises:
            TypeError: If a mode does not use its enum type, a junction
                crossing is malformed, or ``record_planning_trace`` is not a
                boolean.
        """
        if not isinstance(self.strategy, GridCompilerStrategy):
            msg = "strategy must be a GridCompilerStrategy"
            raise TypeError(msg)
        if not isinstance(self.dependency_mode, GridDependencyMode):
            msg = "dependency_mode must be a GridDependencyMode"
            raise TypeError(msg)
        if self.allowed_junction_crossings is not None:
            crossings = frozenset(self.allowed_junction_crossings)
            if any(
                not isinstance(crossing, tuple)
                or len(crossing) != 2
                or any(not isinstance(segment_id, str) or not segment_id for segment_id in crossing)
                for crossing in crossings
            ):
                msg = "allowed_junction_crossings must contain pairs of segment identifiers"
                raise TypeError(msg)
            object.__setattr__(self, "allowed_junction_crossings", crossings)
        _require_positive_int(self.max_iterations, "max_iterations")
        _require_positive_int(self.max_routing_states, "max_routing_states")
        _require_positive_int(self.greedy_lookahead_rounds, "greedy_lookahead_rounds")
        if not isinstance(self.record_planning_trace, bool):
            msg = "record_planning_trace must be a boolean"
            raise TypeError(msg)


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be an integer >= 1"
        raise ValueError(msg)


__all__ = ["GridCompilerConfig", "GridCompilerStrategy", "GridDependencyMode"]
