# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Settings for Linear schedule search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.cost import HeuristicFn
    from mqt.ionshuttler.partitioning import FineGrainedTabuConfig


@dataclass(frozen=True)
class SearchConfig:
    """Configure how the compiler searches for a schedule.

    A finite ``horizon`` plans a few gates at a time and commits
    ``committed_gates`` before planning again. Set ``horizon`` to ``None`` to
    search the complete circuit at once. ``heuristic`` selects the
    remaining-cost estimate that guides the search: the default ``None`` uses
    a quality-oriented estimate that finds useful schedules faster but may
    overestimate the remaining time, while
    :func:`~mqt.ionshuttler.linear.cost.zero_heuristic` is admissible and
    supports exact search when all other limits and shortcuts are also
    disabled. Any other callable matching
    :class:`~mqt.ionshuttler.linear.cost.HeuristicFn` may be supplied instead.
    """

    horizon: int | None = 3
    committed_gates: int = 2
    iterative_diving_search: bool = True
    informed_action_prioritization: bool = False
    num_solutions: int = 1
    max_frontier_size: int | None = 1000
    max_compile_time: float | None = 1800.0
    use_dependencies: bool = True
    heuristic: HeuristicFn | None = None
    pre_partition_config: FineGrainedTabuConfig | None = None

    def __post_init__(self) -> None:
        """Ensure all search limits are meaningful and mutually consistent.

        Raises:
            TypeError: If a Boolean option is not Boolean.
            ValueError: If a numeric bound or horizon relationship is invalid.
        """
        if self.horizon is not None:
            _require_integer_at_least(self.horizon, "horizon", minimum=1)
        _require_integer_at_least(self.committed_gates, "committed_gates", minimum=1)
        if self.horizon is not None and self.committed_gates > self.horizon:
            msg = "committed_gates must be <= horizon"
            raise ValueError(msg)
        _require_integer_at_least(self.num_solutions, "num_solutions", minimum=1)
        if self.max_frontier_size is not None:
            _require_integer_at_least(self.max_frontier_size, "max_frontier_size", minimum=1)
        if self.max_compile_time is not None and (
            isinstance(self.max_compile_time, bool)
            or not isinstance(self.max_compile_time, int | float)
            or self.max_compile_time < 0
        ):
            msg = "max_compile_time must be None or a nonnegative number"
            raise ValueError(msg)
        for name in (
            "iterative_diving_search",
            "informed_action_prioritization",
            "use_dependencies",
        ):
            if not isinstance(getattr(self, name), bool):
                msg = f"{name} must be a boolean"
                raise TypeError(msg)
        if self.heuristic is not None and not callable(self.heuristic):
            msg = "heuristic must be callable or None"
            raise TypeError(msg)


@dataclass(frozen=True)
class LinearCompilerConfig:
    """Configure the Linear scheduling method."""

    search: SearchConfig = field(default_factory=SearchConfig)


def _require_integer_at_least(value: object, name: str, *, minimum: int) -> None:
    """Require a non-Boolean integer at or above a minimum.

    Raises:
        ValueError: If the value is Boolean, noninteger, or below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{name} must be an integer >= {minimum}"
        raise ValueError(msg)


__all__ = ["LinearCompilerConfig", "SearchConfig"]
