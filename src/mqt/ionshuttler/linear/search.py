# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Find Linear schedules with quality-oriented or exact search settings."""

from __future__ import annotations

import logging
from bisect import insort
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES
from mqt.ionshuttler.linear.config import LinearCompilerConfig
from mqt.ionshuttler.linear.cost import cost, heuristic, zero_heuristic
from mqt.ionshuttler.linear.expand import ExpansionOptions, GenerationMode, expand, replay_path
from mqt.ionshuttler.linear.result import LinearDiagnostics
from mqt.ionshuttler.linear.schedule import schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, State, normalize_initial_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.circuit import Circuit
    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.linear.architecture import LinearArchitecture
    from mqt.ionshuttler.linear.config import SearchConfig
    from mqt.ionshuttler.linear.cost import HeuristicFn
    from mqt.ionshuttler.linear.result import LinearCompilationResult
    from mqt.ionshuttler.linear.state import SearchTransition

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
    use_dependencies: bool
    heuristic: HeuristicFn | None = None

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
            use_dependencies=config.use_dependencies,
            heuristic=config.heuristic,
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
    path: tuple[SearchTransition, ...]
    cost_value: int
    heuristic_value: int
    generation_mode: GenerationMode


@dataclass(frozen=True)
class _SearchOutcome:
    """Store the internal result of one search invocation."""

    path: tuple[SearchTransition, ...]
    final_state: State
    status: CompilationStatus
    explored_nodes: int


@dataclass(frozen=True)
class _SearchContext:
    """Collect immutable circuit, hardware, and search-policy inputs."""

    architecture: LinearArchitecture
    circuit: Circuit
    active_gate_ids: tuple[int, ...]
    predecessors: tuple[frozenset[int], ...]
    policy: _SearchPolicy
    action_types: tuple[type[Action], ...]
    critical_path_cache: dict[tuple[int, ...], int] = field(default_factory=dict)
    gate_zone: Mapping[int, str] = field(default_factory=dict)
    zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] = field(default_factory=dict)


FrontierEntry = tuple[int, _SearchNode]
Frontier = list[FrontierEntry]

# Frontier entries sort by estimated total cost, then by insertion order. Packing
# both into one integer reduces every ordering comparison to a single integer
# compare, and makes keys unique so that comparisons never fall through to the
# node itself, which defines no ordering. The width bounds how many entries one
# search may enqueue and is far above any reachable count.
_INSERTION_ORDER_BITS = 44


@dataclass
class _SearchProgress:
    """Track frontier state, best-known paths, and completed solutions."""

    frontier: Frontier
    tie_breaker: count
    current_node: _SearchNode | None
    best_by_state: dict[State, tuple[int, int]]
    best_by_mode: dict[tuple[State, GenerationMode], int]
    explored_nodes: int
    best_path: tuple[SearchTransition, ...]
    best_state: State
    best_key: tuple[int, int, int]
    found_goal_states: set[State]
    best_solution: _SearchOutcome | None


@dataclass
class _RollingProgress:
    """Accumulate committed schedule windows and their explored-node count."""

    state: State
    schedule: list[SearchTransition]
    explored_nodes: int = 0


def search(
    initial_state: State,
    circuit: Circuit,
    architecture: LinearArchitecture,
    config: LinearCompilerConfig | None = None,
    *,
    action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES,
    gate_zone: Mapping[int, str] | None = None,
    zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
) -> LinearCompilationResult:
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
            circuit,
            config=compiler_config,
            action_types=action_types,
            gate_zone=gate_zone,
            zone_site_pairs=zone_site_pairs,
        )
    return rolling_horizon_search(
        normalized_state,
        architecture,
        circuit,
        config=compiler_config,
        action_types=action_types,
        gate_zone=gate_zone,
        zone_site_pairs=zone_site_pairs,
    )


def exhaustive_search(
    initial_state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    *,
    config: LinearCompilerConfig | None = None,
    action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES,
    gate_zone: Mapping[int, str] | None = None,
    zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
) -> LinearCompilationResult:
    """Search the complete circuit at once.

    Returns:
        The best complete schedule, or the best partial schedule if search stops early.
    """
    compiler_config = config or LinearCompilerConfig()
    budget = _TimeBudget.start(compiler_config.search.max_compile_time)
    context = _context(
        architecture,
        circuit,
        compiler_config,
        action_types,
        gate_zone,
        zone_site_pairs,
    )
    outcome = _search_with_budget(initial_state, context, budget)
    return _materialize_result(
        outcome,
        initial_state=initial_state,
        budget=budget,
        architecture=architecture,
    )


def rolling_horizon_search(
    initial_state: State,
    architecture: LinearArchitecture,
    circuit: Circuit,
    *,
    config: LinearCompilerConfig,
    action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES,
    gate_zone: Mapping[int, str] | None = None,
    zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
) -> LinearCompilationResult:
    """Plan a limited number of upcoming gates at a time.

    Returns:
        The combined schedule from every completed planning window.
        Not globally optimal.

    Raises:
        ValueError: If the configuration does not select a rolling horizon.
    """
    horizon = config.search.horizon
    if horizon is None:
        msg = "rolling-horizon search requires a finite horizon"
        raise ValueError(msg)

    budget = _TimeBudget.start(config.search.max_compile_time)
    context = _context(
        architecture,
        circuit,
        config,
        action_types,
        gate_zone,
        zone_site_pairs,
    )
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

    return _materialize_result(
        _SearchOutcome(
            path=tuple(progress.schedule),
            final_state=progress.state,
            status=status,
            explored_nodes=progress.explored_nodes,
        ),
        initial_state=initial_state,
        budget=budget,
        architecture=architecture,
    )


def _run_rolling_search(
    progress: _RollingProgress,
    context: _SearchContext,
    horizon: int,
    committed_gates: int,
    budget: _TimeBudget,
) -> CompilationStatus:
    goal = frozenset(context.active_gate_ids)
    while not goal.issubset(progress.state.completed_gates):
        if budget.expired():
            return CompilationStatus.TIMEOUT
        local_gate_ids = [
            gate_id for gate_id in context.active_gate_ids if gate_id not in progress.state.completed_gates
        ][:horizon]
        local_context = _SearchContext(
            architecture=context.architecture,
            circuit=context.circuit,
            active_gate_ids=tuple(local_gate_ids),
            predecessors=context.predecessors,
            policy=context.policy,
            action_types=context.action_types,
            gate_zone=context.gate_zone,
            zone_site_pairs=context.zone_site_pairs,
        )
        local_outcome = _search_with_budget(progress.state, local_context, budget)
        progress.explored_nodes += local_outcome.explored_nodes
        if local_outcome.status is not CompilationStatus.SUCCESS:
            return local_outcome.status
        _commit_window(
            progress,
            local_outcome.path,
            local_context,
            committed_gates,
        )
    return CompilationStatus.SUCCESS


def _commit_window(
    progress: _RollingProgress,
    path: Sequence[SearchTransition],
    context: _SearchContext,
    committed_gates: int,
) -> None:
    completed_before = progress.state.completed_gates
    for action in path:
        progress.state = replay_path(
            progress.state,
            context.architecture,
            [action],
            context.circuit,
            active_gate_ids=context.active_gate_ids,
            predecessors=context.predecessors,
        )
        progress.schedule.append(action)
        completed = progress.state.completed_gates.difference(completed_before)
        if len(completed & set(context.active_gate_ids)) >= committed_gates:
            return


def _context(
    architecture: LinearArchitecture,
    circuit: Circuit,
    config: LinearCompilerConfig,
    action_types: tuple[type[Action], ...],
    gate_zone: Mapping[int, str] | None = None,
    zone_site_pairs: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
) -> _SearchContext:
    policy = _SearchPolicy.from_config(config.search)
    return _SearchContext(
        architecture=architecture,
        circuit=circuit,
        active_gate_ids=tuple(circuit.gate_ids),
        predecessors=_effective_predecessors(circuit, use_dependencies=policy.use_dependencies),
        policy=policy,
        action_types=action_types,
        gate_zone={} if gate_zone is None else gate_zone,
        zone_site_pairs={} if zone_site_pairs is None else zone_site_pairs,
    )


def _search_with_budget(
    initial_state: State,
    context: _SearchContext,
    budget: _TimeBudget,
) -> _SearchOutcome:
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
        return _partial_outcome(
            progress.best_solution,
            progress.best_path,
            CompilationStatus.INTERRUPTED,
            progress.explored_nodes,
            best_state=progress.best_state,
        )


def _run_search(
    progress: _SearchProgress,
    context: _SearchContext,
    budget: _TimeBudget,
) -> _SearchOutcome:
    goal = frozenset(context.active_gate_ids)
    while progress.current_node is not None or progress.frontier:
        if budget.expired():
            return _partial_outcome(
                progress.best_solution,
                progress.best_path,
                CompilationStatus.TIMEOUT,
                progress.explored_nodes,
                best_state=progress.best_state,
            )

        node, progress.current_node = _take_node(
            progress.current_node,
            progress.frontier,
            context.policy.max_frontier_size,
        )
        if _is_dominated(node, progress.best_by_state, progress.best_by_mode):
            continue
        _record_exploration(progress, node, len(goal))

        if goal.issubset(node.state.completed_gates):
            solution = _better_solution(
                progress.best_solution,
                _SearchOutcome(
                    path=node.path,
                    final_state=node.state,
                    status=CompilationStatus.SUCCESS,
                    explored_nodes=progress.explored_nodes,
                ),
            )
            progress.best_solution = solution
            progress.found_goal_states.add(node.state)
            if len(progress.found_goal_states) >= context.policy.num_solutions:
                return replace(
                    solution,
                    status=CompilationStatus.SUCCESS,
                    explored_nodes=progress.explored_nodes,
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
        return replace(
            progress.best_solution,
            status=CompilationStatus.SUCCESS,
            explored_nodes=progress.explored_nodes,
        )
    return _SearchOutcome(
        path=progress.best_path,
        final_state=progress.best_state,
        status=CompilationStatus.FAILED,
        explored_nodes=progress.explored_nodes,
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
    best_path: tuple[SearchTransition, ...],
    best_state: State,
    best_key: tuple[int, int, int],
) -> tuple[list[_SearchNode], tuple[SearchTransition, ...], State, tuple[int, int, int]]:
    children: list[_SearchNode] = []
    options = ExpansionOptions(mode=node.generation_mode, action_types=context.action_types)
    for action, _, new_state in expand(
        node.state,
        context.architecture,
        context.circuit,
        active_gate_ids=context.active_gate_ids,
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
    max_size: int | None,
) -> tuple[_SearchNode, None]:
    if current_node is not None:
        return current_node, None
    if max_size is None:
        return heappop(frontier)[1], None
    return frontier.pop(0)[1], None


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
    """Add a node to the frontier, discarding the worst entry once it is full.

    An unbounded frontier grows to tens of thousands of entries, where heap
    ordering is the cheaper structure. A bounded frontier instead has to find
    and drop its worst entry on nearly every push, which a heap can only do by
    scanning; keeping it fully sorted puts both ends within reach and costs a
    binary search plus a block move per insertion.
    """
    entry = (_frontier_key(node, next(tie_breaker)), node)
    if max_size is None:
        heappush(frontier, entry)
        return
    insort(frontier, entry)
    if len(frontier) > max_size:
        del frontier[-1]


def _frontier_key(node: _SearchNode, insertion_order: int) -> int:
    """Return the packed heap key ordering a node by cost estimate, then arrival."""
    return (_priority(node) << _INSERTION_ORDER_BITS) | insertion_order


def _heuristic(state: State, context: _SearchContext) -> int:
    custom = context.policy.heuristic
    if custom is not None:
        # The zero estimate is evaluated for every expanded node, so the
        # built-in one short-circuits instead of paying for a call.
        if custom is zero_heuristic:
            return 0
        return custom(
            state,
            context.architecture,
            context.circuit,
            context.active_gate_ids,
            context.predecessors,
            use_dependencies=context.policy.use_dependencies,
            gate_zone=context.gate_zone,
            zone_site_pairs=context.zone_site_pairs,
        )
    return heuristic(
        state,
        context.architecture,
        context.circuit,
        context.active_gate_ids,
        context.predecessors,
        use_dependencies=context.policy.use_dependencies,
        critical_path_cache=context.critical_path_cache,
        gate_zone=context.gate_zone,
        zone_site_pairs=context.zone_site_pairs,
    )


def _priority(node: _SearchNode) -> int:
    return node.cost_value + node.heuristic_value


def _candidate_key(state: State, heuristic_value: int, cost_value: int) -> tuple[int, int, int]:
    return len(state.completed_gates), -heuristic_value, -cost_value


def _better_solution(
    current: _SearchOutcome | None,
    candidate: _SearchOutcome,
) -> _SearchOutcome:
    if current is None:
        return candidate
    return min(
        current,
        candidate,
        key=lambda solution: (cost(solution.final_state), len(solution.path)),
    )


def _partial_outcome(
    solution: _SearchOutcome | None,
    best_path: tuple[SearchTransition, ...],
    status: CompilationStatus,
    explored_nodes: int,
    *,
    best_state: State,
) -> _SearchOutcome:
    if solution is not None:
        return replace(solution, status=status, explored_nodes=explored_nodes)
    return _SearchOutcome(
        path=best_path,
        final_state=best_state,
        status=status,
        explored_nodes=explored_nodes,
    )


def _materialize_result(
    outcome: _SearchOutcome,
    *,
    initial_state: State,
    budget: _TimeBudget,
    architecture: LinearArchitecture,
) -> LinearCompilationResult:
    schedule = schedule_from_path(outcome.path, initial_state, architecture)
    return CompilationResult(
        status=outcome.status,
        schedule=schedule,
        architecture=architecture,
        final_state=architecture.replay_schedule(schedule),
        wall_clock_s=budget.elapsed(),
        diagnostics=LinearDiagnostics(
            score=cost(outcome.final_state),
            explored_nodes=outcome.explored_nodes,
        ),
    )


def _effective_predecessors(circuit: Circuit, *, use_dependencies: bool) -> tuple[frozenset[int], ...]:
    """Return the selected dependency policy in circuit-indexed form."""
    if use_dependencies:
        return circuit.predecessors
    return tuple(frozenset((gate_id - 1,)) if gate_id else frozenset() for gate_id in circuit.gate_ids)


__all__ = ["exhaustive_search", "rolling_horizon_search", "search"]
