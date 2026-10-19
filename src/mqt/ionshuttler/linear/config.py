# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Settings for Linear compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mqt.ionshuttler.core.actions import Action

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.cost import HeuristicFn
    from mqt.ionshuttler.partitioning import FineGrainedTabuConfig


@dataclass(frozen=True, kw_only=True)
class LinearConfig(ABC):
    """Configuration settings shared by all Linear compiler strategies.

    ``action_types`` limits the compiler to a subset of the operations
    supported by the architecture. ``None`` uses every supported operation.
    ``max_compile_time`` bounds the complete compilation in seconds; ``None``
    removes the limit. ``pre_partition`` toggles a pre-partitioning step that
    assigns gates to preferred processing zones. ``pre_partition_config``
    tunes that step.

    Use :class:`SearchConfig` or :class:`PermutationConfig` to select a
    strategy and configure its remaining settings.
    """

    action_types: tuple[type[Action], ...] | None = None
    max_compile_time: float | None = 1800.0
    pre_partition: bool = False
    pre_partition_config: FineGrainedTabuConfig | None = None

    @abstractmethod
    def __post_init__(self) -> None:
        """Validate the shared settings.

        Raises:
            TypeError: If an action type is invalid.
            ValueError: If the time limit is invalid or an action type is repeated.
        """
        object.__setattr__(self, "action_types", _normalize_action_types(self.action_types))
        _validate_compile_time(self.max_compile_time)
        if not isinstance(self.pre_partition, bool):
            msg = "pre_partition must be a boolean"
            raise TypeError(msg)


@dataclass(frozen=True, kw_only=True)
class SearchConfig(LinearConfig):
    """Configuration of the search-based strategy.

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
    use_dependencies: bool = True
    heuristic: HeuristicFn | None = None

    def __post_init__(self) -> None:
        """Ensure all search limits are meaningful and mutually consistent.

        Raises:
            TypeError: If a Boolean option or action type is invalid.
            ValueError: If a bound is invalid or an action type is repeated.
        """
        super().__post_init__()
        if self.horizon is not None:
            _require_integer_at_least(self.horizon, "horizon", minimum=1)
        _require_integer_at_least(self.committed_gates, "committed_gates", minimum=1)
        if self.horizon is not None and self.committed_gates > self.horizon:
            msg = "committed_gates must be <= horizon"
            raise ValueError(msg)
        _require_integer_at_least(self.num_solutions, "num_solutions", minimum=1)
        if self.max_frontier_size is not None:
            _require_integer_at_least(self.max_frontier_size, "max_frontier_size", minimum=1)
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


@dataclass(frozen=True, kw_only=True)
class PermutationConfig(LinearConfig):
    """Configuration of the permutation-based strategy.

    This strategy requires ``pre_partition=True`` because it constructs each
    target placement from preferred gate zones.

    ``lookahead_gates`` bounds how many upcoming gates the compiler inspects
    when it fills one group of gates. A larger value can fill groups better but
    takes more planning time per group. ``record_planning_trace`` stores one
    :class:`~mqt.ionshuttler.linear.result.GateGroup` per group in the result
    diagnostics. Use it to inspect the compiler's decisions; it costs memory.
    """

    pre_partition: Literal[True] = True
    lookahead_gates: int = 32
    record_planning_trace: bool = False

    def __post_init__(self) -> None:
        """Validate the permutation settings.

        Raises:
            TypeError: If a Boolean option or action type is invalid.
            ValueError: If pre-partitioning is disabled.
        """
        super().__post_init__()
        if not self.pre_partition:
            msg = "PermutationConfig requires pre_partition=True"
            raise ValueError(msg)
        _require_integer_at_least(self.lookahead_gates, "lookahead_gates", minimum=1)
        if not isinstance(self.record_planning_trace, bool):
            msg = "record_planning_trace must be a boolean"
            raise TypeError(msg)


def _normalize_action_types(
    action_types: tuple[type[Action], ...] | None,
) -> tuple[type[Action], ...] | None:
    """Validate and normalize the enabled action types.

    Returns:
        The normalized action types, or ``None``.

    Raises:
        TypeError: If an action type is not an ``Action`` subclass.
        ValueError: If an action type is repeated.
    """
    if action_types is not None:
        action_types = tuple(action_types)
        for action_type in action_types:
            if not isinstance(action_type, type) or not issubclass(action_type, Action):
                msg = "action_types must contain Action subclasses"
                raise TypeError(msg)
        if len(set(action_types)) != len(action_types):
            msg = "action_types must not contain duplicates"
            raise ValueError(msg)
    return action_types


def _validate_compile_time(max_compile_time: float | None) -> None:
    """Validate the compilation time limit.

    Raises:
        ValueError: If the time limit is not ``None`` or a nonnegative number.
    """
    if max_compile_time is not None and (
        isinstance(max_compile_time, bool) or not isinstance(max_compile_time, int | float) or max_compile_time < 0
    ):
        msg = "max_compile_time must be None or a nonnegative number"
        raise ValueError(msg)


def _require_integer_at_least(value: object, name: str, *, minimum: int) -> None:
    """Require a non-Boolean integer at or above a minimum.

    Raises:
        ValueError: If the value is Boolean, noninteger, or below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{name} must be an integer >= {minimum}"
        raise ValueError(msg)


__all__ = ["LinearConfig", "PermutationConfig", "SearchConfig"]
