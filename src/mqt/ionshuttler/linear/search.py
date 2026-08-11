# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Find Linear schedules with quality-oriented or exact search settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES, Action, AdvanceTime, GateAction
from mqt.ionshuttler.linear.config import LinearCompilerConfig, TransportTiming
from mqt.ionshuttler.linear.cost import cost, heuristic
from mqt.ionshuttler.linear.expand import ExpansionOptions, GenerationMode, expand, replay_path
from mqt.ionshuttler.linear.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.state import State, normalize_initial_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.config import HeuristicMode, SearchConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TimeBudget:
    """Track elapsed time and an optional search deadline."""

    start_time: float
    deadline: float | None

    @classmethod
    def start(cls, max_compile_time: float | None) -> _TimeBudget:
        """Start a budget with an optional duration in seconds.

        Returns:
            A running time budget.
        """
        now = perf_counter()
        return cls(
            start_time=now,
            deadline=None if max_compile_time is None else now + max_compile_time,
        )

    def expired(self) -> bool:
        """Return whether the deadline has passed."""
        return self.deadline is not None and perf_counter() >= self.deadline

    def elapsed(self) -> float:
        """Return elapsed wall-clock time in seconds."""
        return perf_counter() - self.start_time


@dataclass(frozen=True)
class _SearchPolicy:
    """Collect the candidate modes and limits used during search."""

    generation_schedule: tuple[GenerationMode, ...]
    iterative_diving: bool
    num_solutions: int
    max_frontier_size: int | None
    heuristic_mode: HeuristicMode

    @classmethod
    def from_config(cls, config: SearchConfig) -> _SearchPolicy:
        """Create a search policy from user-facing settings.

        Returns:
            The corresponding internal search policy.
        """
        modes = (
            (GenerationMode.INFORMED, GenerationMode.UNINFORMED)
            if config.informed_action_prioritization
            else (GenerationMode.FULL,)
        )
        return cls(
            generation_schedule=modes,
            iterative_diving=config.iterative_diving_search,
            num_solutions=config.num_solutions,
            max_frontier_size=config.max_frontier_size,
            heuristic_mode=config.heuristic_mode,
        )

    @property
    def initial_mode(self) -> GenerationMode:
        """First candidate-generation mode to explore."""
        return self.generation_schedule[0]

    def next_mode(self, current: GenerationMode) -> GenerationMode | None:
        """Return the broader mode following ``current``, if any."""
        try:
            index = self.generation_schedule.index(current) + 1
        except ValueError:
            return None
        return self.generation_schedule[index] if index < len(self.generation_schedule) else None


@dataclass(frozen=True)
class _SearchNode:
    """Store one compiler state and the schedule prefix that reached it."""

    state: State
    path: tuple[Action, ...]
    cost_value: int
    heuristic_value: int
    generation_mode: GenerationMode


@dataclass(frozen=True)
class _FoundSolution:
    """Store a completed schedule and its final state and cost."""

    path: tuple[Action, ...]
    final_state: State
    cost_value: int


@dataclass(frozen=True)
class _SearchContext:
    """Collect immutable circuit, hardware, and search-policy inputs."""

    architecture: Architecture
    gate_order: Sequence[int]
    gates: Mapping[int, GateAction]
    predecessors: Mapping[int, frozenset[int]] | None
    policy: _SearchPolicy
    transport_timing: TransportTiming
    action_types: tuple[type[Action], ...]


FrontierEntry = tuple[int, int, _SearchNode]
Frontier = list[FrontierEntry]


@dataclass
class _SearchProgress:
    """Track frontier state, best-known paths, and completed solutions."""

    frontier: Frontier
    tie_breaker: count
    current_node: _SearchNode | None
    best_by_state: dict[State, tuple[int, int]]
    best_by_mode: dict[tuple[State, GenerationMode], int]
    explored_nodes: int
    best_path: tuple[Action, ...]
    best_state: State
    best_key: tuple[int, int, int]
    found_goal_states: set[State]
    best_solution: _FoundSolution | None


@dataclass
class _RollingProgress:
    """Accumulate committed schedule windows and their explored-node count."""

    state: State
    schedule: list[Action]
    explored_nodes: int = 0


def search(
    initial_state: State,
    gate_order: Sequence[int],
    gates: Mapping[int, GateAction],
    architecture: Architecture,
    predecessors: Mapping[int, frozenset[int]] | None = None,
    config: LinearCompilerConfig | None = None,
    *,
    action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES,
) -> CompilationResult:
    """Compile a circuit using the configured global or rolling search.

    Returns:
        The schedule, completion status, and state reached by the search.
    """
    compiler_config = config or LinearCompilerConfig()
    normalized_state = normalize_initial_state(initial_state, architecture)
    if compiler_config.search.horizon is None:
        return exhaustive_search(
            normalized_state,
            architecture,
            gate_order,
            gates,
            predecessors=predecessors,
            config=compiler_config,
            action_types=action_types,
        )
    return rolling_horizon_search(
        normalized_state,
        architecture,
        gate_order,
        gates,
        predecessors=predecessors,
        config=compiler_config,
        action_types=action_types,
    )


def exhaustive_search(
    initial_state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: Mapping[int, GateAction],
    *,
    predecessors: Mapping[int, frozenset[int]] | None = None,
    config: LinearCompilerConfig | None = None,
    action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES,
) -> CompilationResult:
    """Search the complete circuit at once.

    Returns:
        The best complete schedule, or the best partial schedule if search stops early.
    """
    compiler_config = config or LinearCompilerConfig()
    budget = _TimeBudget.start(compiler_config.search.max_compile_time)
    context = _context(
        architecture,
        gate_order,
        gates,
        predecessors,
        compiler_config,
        action_types,
    )
    result = _search_with_budget(initial_state, context, budget)
    return _with_public_metadata(
        result,
        budget=budget,
        architecture=architecture,
        initial_state=initial_state,
        action_types=action_types,
    )


def rolling_horizon_search(
    initial_state: State,
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: Mapping[int, GateAction],
    *,
    predecessors: Mapping[int, frozenset[int]] | None,
    config: LinearCompilerConfig,
    action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES,
) -> CompilationResult:
    """Plan a limited number of upcoming gates at a time.

    Returns:
        The combined schedule from every completed planning window.

    Raises:
        ValueError: If the configuration does not select a rolling horizon.
    """
    horizon = config.search.horizon
    if horizon is None:
        msg = "rolling-horizon search requires a finite horizon"
        raise ValueError(msg)

    budget = _TimeBudget.start(config.search.max_compile_time)
    context = _context(architecture, gate_order, gates, predecessors, config, action_types)
    progress = _RollingProgress(state=initial_state, schedule=[])

    try:
        status = _run_rolling_search(
            progress,
            context,
            horizon,
            config.search.committed_gates,
            budget,
        )
    except KeyboardInterrupt:
        status = CompilationStatus.INTERRUPTED

    return _rolling_result(
        progress.schedule,
        status,
        progress.state,
        initial_state,
        architecture,
        progress.explored_nodes,
        budget,
        action_types,
    )


def _run_rolling_search(
    progress: _RollingProgress,
    context: _SearchContext,
    horizon: int,
    committed_gates: int,
    budget: _TimeBudget,
) -> CompilationStatus:
    goal = frozenset(context.gate_order)
    while not goal.issubset(progress.state.completed_gates):
        if budget.expired():
            return CompilationStatus.TIMEOUT
        local_order = [gate_id for gate_id in context.gate_order if gate_id not in progress.state.completed_gates][
            :horizon
        ]
        local_gates = {gate_id: context.gates[gate_id] for gate_id in local_order}
        local_predecessors = _local_predecessors(
            local_order,
            context.predecessors,
            progress.state.completed_gates,
        )
        local_context = _SearchContext(
            architecture=context.architecture,
            gate_order=local_order,
            gates=local_gates,
            predecessors=local_predecessors,
            policy=context.policy,
            transport_timing=context.transport_timing,
            action_types=context.action_types,
        )
        local_result = _search_with_budget(progress.state, local_context, budget)
        progress.explored_nodes += local_result.explored_nodes or 0
        if local_result.status is not CompilationStatus.SUCCESS:
            return local_result.status
        _commit_window(progress, local_result.path, local_context, committed_gates)
    return CompilationStatus.SUCCESS


def _commit_window(
    progress: _RollingProgress,
    path: Sequence[Action],
    context: _SearchContext,
    committed_gates: int,
) -> None:
    completed_before = progress.state.completed_gates
    for action in path:
        progress.state = replay_path(
            progress.state,
            context.architecture,
            [action],
            context.gate_order,
            context.gates,
            predecessors=context.predecessors,
        )
        progress.schedule.append(action)
        completed = progress.state.completed_gates.difference(completed_before)
        if len(completed & set(context.gate_order)) >= committed_gates:
            return


def _context(
    architecture: Architecture,
    gate_order: Sequence[int],
    gates: Mapping[int, GateAction],
    predecessors: Mapping[int, frozenset[int]] | None,
    config: LinearCompilerConfig,
    action_types: tuple[type[Action], ...],
) -> _SearchContext:
    return _SearchContext(
        architecture=architecture,
        gate_order=gate_order,
        gates=gates,
        predecessors=predecessors,
        policy=_SearchPolicy.from_config(config.search),
        transport_timing=config.hardware_timing.transport,
        action_types=action_types,
    )


def _search_with_budget(
    initial_state: State,
    context: _SearchContext,
    budget: _TimeBudget,
) -> CompilationResult:
    tie_breaker = count()
    frontier: Frontier = []
    initial_heuristic = _heuristic(initial_state, context)
    initial_node = _SearchNode(
        state=initial_state,
        path=(),
        cost_value=cost(initial_state),
        heuristic_value=initial_heuristic,
        generation_mode=context.policy.initial_mode,
    )
    current_node = initial_node if context.policy.iterative_diving else None
    if current_node is None:
        _push_frontier(frontier, initial_node, tie_breaker, context.policy.max_frontier_size)

    progress = _SearchProgress(
        frontier=frontier,
        tie_breaker=tie_breaker,
        current_node=current_node,
        best_by_state={},
        best_by_mode={},
        explored_nodes=0,
        best_path=(),
        best_state=initial_state,
        best_key=_candidate_key(initial_state, initial_heuristic, cost(initial_state)),
        found_goal_states=set(),
        best_solution=None,
    )

    try:
        return _run_search(progress, context, budget)
    except KeyboardInterrupt:
        return _partial_result(
            progress.best_solution,
            progress.best_path,
            progress.best_state,
            CompilationStatus.INTERRUPTED,
            context.architecture,
            progress.explored_nodes,
        )


def _run_search(
    progress: _SearchProgress,
    context: _SearchContext,
    budget: _TimeBudget,
) -> CompilationResult:
    goal = frozenset(context.gate_order)
    while progress.current_node is not None or progress.frontier:
        if budget.expired():
            return _partial_result(
                progress.best_solution,
                progress.best_path,
                progress.best_state,
                CompilationStatus.TIMEOUT,
                context.architecture,
                progress.explored_nodes,
            )

        node, progress.current_node = _take_node(
            progress.current_node,
            progress.frontier,
        )
        if _is_dominated(node, progress.best_by_state, progress.best_by_mode):
            continue
        _record_exploration(progress, node, len(goal))

        if goal.issubset(node.state.completed_gates):
            solution = _better_solution(
                progress.best_solution,
                _FoundSolution(node.path, node.state, node.cost_value),
            )
            progress.best_solution = solution
            progress.found_goal_states.add(node.state)
            if len(progress.found_goal_states) >= context.policy.num_solutions:
                return _result(
                    solution.path,
                    CompilationStatus.SUCCESS,
                    solution.final_state,
                    context.architecture,
                    progress.explored_nodes,
                )
            continue

        progress.best_by_state[node.state] = (node.cost_value, len(node.path))
        progress.best_by_mode[node.state, node.generation_mode] = node.cost_value
        children, progress.best_path, progress.best_state, progress.best_key = _children(
            node,
            context,
            progress.best_by_state,
            progress.best_path,
            progress.best_state,
            progress.best_key,
        )
        _queue_next_mode(
            node,
            context.policy,
            progress.best_by_mode,
            progress.frontier,
            progress.tie_breaker,
        )
        _queue_children(progress, context.policy, node, children)

    if progress.best_solution is not None:
        return _result(
            progress.best_solution.path,
            CompilationStatus.SUCCESS,
            progress.best_solution.final_state,
            context.architecture,
            progress.explored_nodes,
        )
    return _result(
        progress.best_path,
        CompilationStatus.FAILED,
        progress.best_state,
        context.architecture,
        progress.explored_nodes,
    )


def _record_exploration(
    progress: _SearchProgress,
    node: _SearchNode,
    goal_size: int,
) -> None:
    progress.explored_nodes += 1
    if progress.explored_nodes == 1 or progress.explored_nodes % 1000 == 0:
        logger.debug(
            "search progress: explored=%s, completed=%s/%s, frontier=%s",
            progress.explored_nodes,
            len(node.state.completed_gates),
            goal_size,
            len(progress.frontier),
        )
    node_key = _candidate_key(node.state, node.heuristic_value, node.cost_value)
    if node_key > progress.best_key:
        progress.best_path = node.path
        progress.best_state = node.state
        progress.best_key = node_key


def _queue_children(
    progress: _SearchProgress,
    policy: _SearchPolicy,
    parent: _SearchNode,
    children: Sequence[_SearchNode],
) -> None:
    if policy.iterative_diving:
        progress.current_node = _choose_dive(
            children,
            parent.heuristic_value,
            progress.best_by_state,
        )
    for child in children:
        if child is not progress.current_node:
            _push_frontier(
                progress.frontier,
                child,
                progress.tie_breaker,
                policy.max_frontier_size,
            )


def _children(
    node: _SearchNode,
    context: _SearchContext,
    best_by_state: Mapping[State, tuple[int, int]],
    best_path: tuple[Action, ...],
    best_state: State,
    best_key: tuple[int, int, int],
) -> tuple[list[_SearchNode], tuple[Action, ...], State, tuple[int, int, int]]:
    children: list[_SearchNode] = []
    options = ExpansionOptions(
        mode=node.generation_mode,
        transport_timing=context.transport_timing,
        action_types=context.action_types,
    )
    for action, _, new_state in expand(
        node.state,
        context.architecture,
        context.gate_order,
        context.gates,
        predecessors=context.predecessors,
        options=options,
    ):
        new_path = (*node.path, action)
        new_cost = cost(new_state)
        new_heuristic = _heuristic(new_state, context)
        candidate_key = _candidate_key(new_state, new_heuristic, new_cost)
        if candidate_key > best_key:
            best_path, best_state, best_key = new_path, new_state, candidate_key
        if (new_cost, len(new_path)) >= best_by_state.get(
            new_state,
            (float("inf"), float("inf")),
        ):
            continue
        children.append(
            _SearchNode(
                state=new_state,
                path=new_path,
                cost_value=new_cost,
                heuristic_value=new_heuristic,
                generation_mode=context.policy.initial_mode,
            )
        )
    return children, best_path, best_state, best_key


def _choose_dive(
    children: Sequence[_SearchNode],
    parent_heuristic: int,
    best_by_state: Mapping[State, tuple[int, int]],
) -> _SearchNode | None:
    improving = [child for child in children if child.heuristic_value < parent_heuristic]
    if improving:
        candidate = min(
            improving,
            key=lambda child: (
                _priority(child),
                -len(child.state.completed_gates),
                child.cost_value,
            ),
        )
        existing = best_by_state.get(candidate.state)
        if existing is None or (candidate.cost_value, len(candidate.path)) < existing:
            return candidate
        return None
    return next(
        (child for child in children if child.path and isinstance(child.path[-1], AdvanceTime)),
        None,
    )


def _queue_next_mode(
    node: _SearchNode,
    policy: _SearchPolicy,
    best_by_mode: Mapping[tuple[State, GenerationMode], int],
    frontier: Frontier,
    tie_breaker: count,
) -> None:
    next_mode = policy.next_mode(node.generation_mode)
    if next_mode is None:
        return
    if node.cost_value >= best_by_mode.get((node.state, next_mode), float("inf")):
        return
    _push_frontier(
        frontier,
        _SearchNode(
            state=node.state,
            path=node.path,
            cost_value=node.cost_value,
            heuristic_value=node.heuristic_value,
            generation_mode=next_mode,
        ),
        tie_breaker,
        policy.max_frontier_size,
    )


def _take_node(
    current_node: _SearchNode | None,
    frontier: Frontier,
) -> tuple[_SearchNode, None]:
    if current_node is not None:
        return current_node, None
    return heappop(frontier)[2], None


def _is_dominated(
    node: _SearchNode,
    best_by_state: Mapping[State, tuple[int, int]],
    best_by_mode: Mapping[tuple[State, GenerationMode], int],
) -> bool:
    state_best = best_by_state.get(node.state)
    if state_best is not None and (node.cost_value, len(node.path)) > state_best:
        return True
    mode_best = best_by_mode.get((node.state, node.generation_mode))
    return mode_best is not None and node.cost_value > mode_best


def _push_frontier(
    frontier: Frontier,
    node: _SearchNode,
    tie_breaker: count,
    max_size: int | None,
) -> None:
    heappush(frontier, (_priority(node), next(tie_breaker), node))
    if max_size is None or len(frontier) <= max_size:
        return
    worst_index = max(
        range(len(frontier)),
        key=lambda index: (frontier[index][0], frontier[index][1]),
    )
    last_entry = frontier.pop()
    if worst_index < len(frontier):
        frontier[worst_index] = last_entry
        _sift_up(frontier, worst_index)


def _sift_up(frontier: Frontier, position: int) -> None:
    """Repair a heap after replacing one entry with its previous last entry."""
    end = len(frontier)
    start = position
    new_entry = frontier[position]
    child = 2 * position + 1
    while child < end:
        right = child + 1
        if right < end and not _frontier_entry_precedes(frontier[child], frontier[right]):
            child = right
        frontier[position] = frontier[child]
        position = child
        child = 2 * position + 1
    frontier[position] = new_entry
    _sift_down(frontier, start, position)


def _sift_down(frontier: Frontier, start: int, position: int) -> None:
    new_entry = frontier[position]
    while position > start:
        parent = (position - 1) >> 1
        parent_entry = frontier[parent]
        if not _frontier_entry_precedes(new_entry, parent_entry):
            break
        frontier[position] = parent_entry
        position = parent
    frontier[position] = new_entry


def _frontier_entry_precedes(left: FrontierEntry, right: FrontierEntry) -> bool:
    return (left[0], left[1]) < (right[0], right[1])


def _heuristic(state: State, context: _SearchContext) -> int:
    if context.policy.heuristic_mode == "zero":
        return 0
    return heuristic(
        state,
        context.architecture,
        context.gate_order,
        context.gates,
        context.predecessors,
    )


def _priority(node: _SearchNode) -> int:
    return node.cost_value + node.heuristic_value


def _candidate_key(state: State, heuristic_value: int, cost_value: int) -> tuple[int, int, int]:
    return len(state.completed_gates), -heuristic_value, -cost_value


def _better_solution(
    current: _FoundSolution | None,
    candidate: _FoundSolution,
) -> _FoundSolution:
    if current is None:
        return candidate
    return min(
        current,
        candidate,
        key=lambda solution: (solution.cost_value, len(solution.path)),
    )


def _partial_result(
    solution: _FoundSolution | None,
    best_path: tuple[Action, ...],
    best_state: State,
    status: CompilationStatus,
    architecture: Architecture,
    explored_nodes: int,
) -> CompilationResult:
    if solution is not None:
        return _result(
            solution.path,
            status,
            solution.final_state,
            architecture,
            explored_nodes,
        )
    return _result(best_path, status, best_state, architecture, explored_nodes)


def _result(
    path: Sequence[Action],
    status: CompilationStatus,
    final_state: State,
    architecture: Architecture,
    explored_nodes: int,
) -> CompilationResult:
    public_path = list(path)
    return CompilationResult(
        status=status,
        path=public_path,
        num_timesteps=sum(isinstance(action, AdvanceTime) for action in public_path),
        score=cost(final_state),
        final_state=final_state,
        architecture=architecture,
        explored_nodes=explored_nodes,
    )


def _with_public_metadata(
    result: CompilationResult,
    *,
    budget: _TimeBudget,
    architecture: Architecture,
    initial_state: State,
    action_types: Sequence[type[Action]],
) -> CompilationResult:
    return replace(
        result,
        wall_clock_s=budget.elapsed(),
        architecture=architecture,
        initial_state=initial_state,
        action_types=tuple(action_type.__name__ for action_type in action_types),
    )


def _rolling_result(
    path: Sequence[Action],
    status: CompilationStatus,
    final_state: State,
    initial_state: State,
    architecture: Architecture,
    explored_nodes: int,
    budget: _TimeBudget,
    action_types: Sequence[type[Action]],
) -> CompilationResult:
    result = _result(path, status, final_state, architecture, explored_nodes)
    return _with_public_metadata(
        result,
        budget=budget,
        architecture=architecture,
        initial_state=initial_state,
        action_types=action_types,
    )


def _local_predecessors(
    local_order: Sequence[int],
    predecessors: Mapping[int, frozenset[int]] | None,
    completed_gates: frozenset[int],
) -> dict[int, frozenset[int]] | None:
    if predecessors is None:
        return None
    included = set(local_order) | set(completed_gates)
    return {gate_id: predecessors.get(gate_id, frozenset()) & included for gate_id in local_order}


__all__ = ["exhaustive_search", "rolling_horizon_search", "search"]
