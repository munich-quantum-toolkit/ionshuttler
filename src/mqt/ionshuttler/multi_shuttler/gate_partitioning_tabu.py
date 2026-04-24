from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
import random
import time

from .circuit_types import GateInfo


DistanceMatrix = list[list[float]]


@dataclass(slots=True)
class FineGrainedTabuConfig:
    """Configuration for the fine-grained tabu gate partitioner.

    The defaults mirror the prototype configuration that was used for the
    `fgp_tabu_global_2` partitioner. The target-side port keeps relaxed layering
    fixed internally, so that knob is intentionally not exposed here.
    """

    balance_penalty: float = 1.0
    capacity_weight: float = 0.5
    distance_weight_factor: float = 1.0
    max_iterations: int | None = None
    max_iterations_factor: float | None = 20.0
    tabu_list_length: int = 200
    candidate_list_length: int | None = 200
    per_slice_quota: int | None = None
    slack_dropoff: float | None = 1.0
    refresh_every: int | None = None
    randomize_initial: bool = False
    seed: int | None = 0
    max_layer_depth: int | None = None

    def resolve_max_iterations(self, num_qubits: int) -> int:
        """Resolve the iteration budget for a concrete circuit size."""

        if self.max_iterations is not None:
            return max(self.max_iterations, 1)
        if self.max_iterations_factor is None:
            return 100
        return max(int(self.max_iterations_factor * num_qubits), 1)


@dataclass(slots=True)
class GatePartitionResult:
    """Runtime-neutral output of the fine-grained tabu partitioner.

    Attributes:
        gate_partition_by_pz: Gate ids grouped by processing zone in execution order.
        gate_assignment: Direct gate-id to processing-zone mapping.
        time_slices: Relaxed gate slices used by the optimizer.
        qubit_assignments_by_slice: Per-slice qubit-to-cluster assignments.
        cost_before: Objective value before tabu refinement.
        cost_after: Best objective value reached by tabu refinement.
        move_distance_total: Aggregate move distance between adjacent slices.
        optimization_time: Wall-clock time spent in the tabu loop in seconds.
    """

    gate_partition_by_pz: dict[str, list[int]]
    gate_assignment: dict[int, str]
    time_slices: list[list[int]]
    qubit_assignments_by_slice: list[list[int]]
    cost_before: float
    cost_after: float
    move_distance_total: float
    optimization_time: float


@dataclass(slots=True)
class _Supernode:
    """Collapsed component of qubits that must stay together within one slice."""

    id: int
    qubits: tuple[int, ...]
    load: int


@dataclass(slots=True)
class _SliceContraction:
    """Internal slice representation used during global tabu refinement."""

    supernodes: list[_Supernode]
    qubit_to_supernode: dict[int, int]
    required_edges: dict[tuple[int, int], float]
    required_unary: set[int]
    cluster_assignment: list[int] | None
    cluster_loads: list[int] | None


def compute_fine_grained_gate_partition(
    sequence: Sequence[int],
    gate_info: dict[int, GateInfo],
    pz_names: Sequence[str],
    pz_distance_matrix: Sequence[Sequence[float]] | None,
    *,
    capacity: int | None = None,
    config: FineGrainedTabuConfig | None = None,
) -> GatePartitionResult:
    """Compute a fine-grained gate-to-processing-zone assignment.

    The algorithm first groups gates into relaxed time slices, contracts qubits
    that must move together inside each slice, and then refines all slice
    assignments jointly with a tabu search. The result is expressed in terms of
    gate ids and processing-zone names so later runtime code can consume it
    without depending on this module's internal optimization state.

    Args:
        sequence: Ordered gate ids to partition.
        gate_info: Gate metadata keyed by gate id.
        pz_names: Processing-zone names in cluster index order.
        pz_distance_matrix: Square distance matrix indexed by processing-zone order.
        capacity: Optional soft per-zone qubit capacity. When omitted, the port uses
            the balanced default `ceil(num_qubits / num_pzs)`.
        config: Optional configuration object. Defaults are chosen to match the
            prototype configuration used by `fgp_tabu_global_2`.

    Returns:
        A runtime-neutral gate partition result.

    Raises:
        ValueError: If the inputs are inconsistent or incomplete.
    """

    resolved_config = config or FineGrainedTabuConfig()
    _validate_gate_inputs(sequence, gate_info, pz_names, capacity)

    normalized_distances = _normalize_distance_matrix(pz_names, pz_distance_matrix)
    if not sequence:
        return GatePartitionResult(
            gate_partition_by_pz={pz_name: [] for pz_name in pz_names},
            gate_assignment={},
            time_slices=[],
            qubit_assignments_by_slice=[],
            cost_before=0.0,
            cost_after=0.0,
            move_distance_total=0.0,
            optimization_time=0.0,
        )

    num_pzs = len(pz_names)
    num_qubits = _infer_num_qubits(gate_info)
    resolved_capacity = capacity if capacity is not None else max(math.ceil(num_qubits / num_pzs), 1)
    time_slices = _build_time_slices_relaxed(
        sequence,
        gate_info,
        num_qubits,
        max_layer_depth=resolved_config.max_layer_depth,
    )

    slice_contractions: list[_SliceContraction] = []
    seed_assignment: list[int] | None = None
    for index, slice_gate_ids in enumerate(time_slices):
        contraction = _contract_slice(
            gate_info,
            slice_gate_ids,
            num_qubits=num_qubits,
            num_pzs=num_pzs,
            previous_qubit_assignment=seed_assignment,
            randomize_initial=resolved_config.randomize_initial,
            seed=resolved_config.seed,
        )
        slice_contractions.append(contraction)
        if index == 0:
            seed_assignment = _build_qubit_assignment(contraction, num_qubits, num_pzs)

    optimization_start = time.perf_counter()
    optimized_assignments, cost_before, cost_after = _optimize_gate_partition(
        slice_contractions,
        num_pzs=num_pzs,
        capacity=resolved_capacity,
        distance_matrix=normalized_distances,
        num_qubits=num_qubits,
        config=resolved_config,
    )
    optimization_time = time.perf_counter() - optimization_start

    for slice_index, contraction in enumerate(slice_contractions):
        contraction.cluster_assignment = optimized_assignments[slice_index]
        contraction.cluster_loads = _compute_cluster_loads(contraction, num_pzs)

    (
        qubit_assignments_by_slice,
        gate_partition_by_pz,
        gate_assignment,
    ) = _build_gate_partition_projection(
        slice_contractions,
        time_slices,
        gate_info,
        pz_names,
        num_qubits=num_qubits,
    )
    move_distance_total = _compute_total_move_distance(
        _compute_moves(qubit_assignments_by_slice),
        normalized_distances,
    )

    return GatePartitionResult(
        gate_partition_by_pz=gate_partition_by_pz,
        gate_assignment=gate_assignment,
        time_slices=time_slices,
        qubit_assignments_by_slice=qubit_assignments_by_slice,
        cost_before=cost_before,
        cost_after=cost_after,
        move_distance_total=move_distance_total,
        optimization_time=optimization_time,
    )


def _validate_gate_inputs(
    sequence: Sequence[int],
    gate_info: dict[int, GateInfo],
    pz_names: Sequence[str],
    capacity: int | None,
) -> None:
    if not pz_names:
        msg = "At least one processing zone is required."
        raise ValueError(msg)
    if len(set(pz_names)) != len(pz_names):
        msg = "Processing zone names must be unique."
        raise ValueError(msg)
    if capacity is not None and capacity <= 0:
        msg = "capacity must be positive when provided."
        raise ValueError(msg)
    missing_gate_ids = [gate_id for gate_id in sequence if gate_id not in gate_info]
    if missing_gate_ids:
        msg = f"Sequence references unknown gate ids: {missing_gate_ids[:5]}"
        raise ValueError(msg)


def _normalize_distance_matrix(
    pz_names: Sequence[str],
    pz_distance_matrix: Sequence[Sequence[float]] | None,
) -> DistanceMatrix | None:
    if pz_distance_matrix is None:
        return None

    num_pzs = len(pz_names)
    if len(pz_distance_matrix) != num_pzs:
        msg = "Distance matrix must have one row per processing zone."
        raise ValueError(msg)

    normalized: DistanceMatrix = []
    for row in pz_distance_matrix:
        if len(row) != num_pzs:
            msg = "Distance matrix must be square and aligned with processing-zone order."
            raise ValueError(msg)
        normalized.append([float(value) for value in row])
    return normalized


def _infer_num_qubits(gate_info: dict[int, GateInfo]) -> int:
    max_qubit = -1
    for info in gate_info.values():
        if info.qubits:
            max_qubit = max(max_qubit, max(info.qubits))
    if max_qubit < 0:
        msg = "Unable to infer qubit count from gate metadata."
        raise ValueError(msg)
    return max_qubit + 1


class _UnionFind:
    def __init__(self, elements: Sequence[int]) -> None:
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}

    def find(self, item: int) -> int:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        left_rank = self.rank.setdefault(left_root, 0)
        right_rank = self.rank.setdefault(right_root, 0)
        if left_rank < right_rank:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if left_rank == right_rank:
            self.rank[left_root] = left_rank + 1
        return True


def _can_accept_gate_without_triplet(
    qubits: Sequence[int],
    union_find: _UnionFind,
    component_sizes: dict[int, int],
) -> bool:
    if len(qubits) <= 1:
        return True
    if len(qubits) > 2:
        return False

    first_root = union_find.find(qubits[0])
    second_root = union_find.find(qubits[1])
    first_size = component_sizes.get(first_root, 1)
    second_size = component_sizes.get(second_root, 1)
    if first_root == second_root:
        return first_size <= 2
    return first_size + second_size <= 2


def _build_time_slices_relaxed(
    gate_ids: Sequence[int],
    gate_info: dict[int, GateInfo],
    num_qubits: int,
    *,
    max_layer_depth: int | None = None,
) -> list[list[int]]:
    """Group gates into relaxed slices while avoiding oversized live components."""

    ordered_gate_ids = list(gate_ids)
    processed: set[int] = set()
    time_slices: list[list[int]] = []
    total_gates = len(ordered_gate_ids)
    max_depth = None if max_layer_depth is None or max_layer_depth <= 0 else max_layer_depth

    while len(processed) < total_gates:
        blocked_qubits: set[int] = set()
        current_slice: list[int] = []
        progress = False
        gate_depth_by_qubit: defaultdict[int, int] = defaultdict(int)
        union_find = _UnionFind(list(range(num_qubits)))
        component_sizes: dict[int, int] = {qubit: 1 for qubit in range(num_qubits)}
        component_members: dict[int, set[int]] = {qubit: {qubit} for qubit in range(num_qubits)}

        for gate_id in ordered_gate_ids:
            if gate_id in processed:
                continue
            qubits = tuple(dict.fromkeys(gate_info[gate_id].qubits))
            if not qubits:
                processed.add(gate_id)
                progress = True
                continue
            if any(qubit in blocked_qubits for qubit in qubits):
                blocked_qubits.update(qubits)
                continue
            if max_depth is not None and any(gate_depth_by_qubit[qubit] >= max_depth for qubit in qubits):
                blocked_qubits.update(qubits)
                continue
            if not _can_accept_gate_without_triplet(qubits, union_find, component_sizes):
                if len(qubits) == 2:
                    first_root = union_find.find(qubits[0])
                    second_root = union_find.find(qubits[1])
                    blocked_qubits.update(component_members.get(first_root, {qubits[0]}))
                    blocked_qubits.update(component_members.get(second_root, {qubits[1]}))
                blocked_qubits.update(qubits)
                continue

            processed.add(gate_id)
            progress = True
            current_slice.append(gate_id)
            for qubit in qubits:
                gate_depth_by_qubit[qubit] += 1
                if max_depth is not None and gate_depth_by_qubit[qubit] >= max_depth:
                    blocked_qubits.add(qubit)

            if len(qubits) == 2:
                first_qubit, second_qubit = qubits
                first_root = union_find.find(first_qubit)
                second_root = union_find.find(second_qubit)
                if first_root != second_root:
                    first_members = component_members.get(first_root, {first_qubit})
                    second_members = component_members.get(second_root, {second_qubit})
                    union_find.union(first_qubit, second_qubit)
                    new_root = union_find.find(first_qubit)
                    merged_members = set(first_members) | set(second_members)
                    component_members[new_root] = merged_members
                    component_sizes[new_root] = len(merged_members)
                    if first_root != new_root:
                        component_members.pop(first_root, None)
                        component_sizes.pop(first_root, None)
                    if second_root != new_root:
                        component_members.pop(second_root, None)
                        component_sizes.pop(second_root, None)

        if current_slice:
            time_slices.append(current_slice)

        if not progress:
            for gate_id in ordered_gate_ids:
                if gate_id in processed:
                    continue
                processed.add(gate_id)
                if gate_info[gate_id].qubits:
                    time_slices.append([gate_id])
                break
            else:
                break

    return time_slices


def _contract_slice(
    gate_info: dict[int, GateInfo],
    slice_gate_ids: Sequence[int],
    *,
    num_qubits: int,
    num_pzs: int,
    previous_qubit_assignment: list[int] | None,
    randomize_initial: bool,
    seed: int | None,
) -> _SliceContraction:
    if not slice_gate_ids:
        msg = "Slice must contain at least one gate to contract."
        raise ValueError(msg)

    required_edges = _build_edge_weights(slice_gate_ids, gate_info)
    required_unary = {
        gate_info[gate_id].qubits[0]
        for gate_id in slice_gate_ids
        if len(gate_info[gate_id].qubits) == 1
    }
    qubits = list(range(num_qubits))
    supernodes, qubit_to_supernode = _contract_supernodes(qubits, required_edges)

    supernode_loads = [0] * len(supernodes)
    for gate_id in slice_gate_ids:
        qubits_in_gate = gate_info[gate_id].qubits
        if not qubits_in_gate:
            continue
        touched_supernodes = {
            qubit_to_supernode[qubit]
            for qubit in set(qubits_in_gate)
            if qubit in qubit_to_supernode
        }
        for supernode_id in touched_supernodes:
            supernode_loads[supernode_id] += len(qubits_in_gate)
    for supernode in supernodes:
        supernode.load = max(1, supernode_loads[supernode.id])

    cluster_assignment: list[int] | None = None
    cluster_loads: list[int] | None = None
    if previous_qubit_assignment is not None and not randomize_initial:
        cluster_assignment, cluster_loads = _seed_assignment_from_previous(
            supernodes,
            previous_qubit_assignment,
            num_pzs,
        )
    if cluster_assignment is None:
        cluster_assignment, cluster_loads = _greedy_initial_partition(
            supernodes,
            num_pzs,
            randomize_initial,
            seed,
        )

    return _SliceContraction(
        supernodes=supernodes,
        qubit_to_supernode=qubit_to_supernode,
        required_edges=required_edges,
        required_unary=required_unary,
        cluster_assignment=cluster_assignment,
        cluster_loads=cluster_loads,
    )


def _build_edge_weights(
    gate_ids: Sequence[int],
    gate_info: dict[int, GateInfo],
) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    for gate_id in gate_ids:
        qubits = gate_info[gate_id].qubits
        if len(qubits) != 2:
            continue
        edge = tuple(sorted(qubits))
        weights[edge] = weights.get(edge, 0.0) + 1.0
    return weights


def _contract_supernodes(
    qubits: Sequence[int],
    required_edges: dict[tuple[int, int], float],
) -> tuple[list[_Supernode], dict[int, int]]:
    union_find = _UnionFind(list(qubits))
    for left, right in required_edges:
        union_find.union(left, right)

    components: dict[int, list[int]] = {}
    for qubit in qubits:
        root = union_find.find(qubit)
        components.setdefault(root, []).append(qubit)

    supernodes: list[_Supernode] = []
    qubit_to_supernode: dict[int, int] = {}
    for index, nodes in enumerate(components.values()):
        nodes_sorted = tuple(sorted(nodes))
        supernode = _Supernode(id=index, qubits=nodes_sorted, load=len(nodes_sorted))
        supernodes.append(supernode)
        for qubit in nodes_sorted:
            qubit_to_supernode[qubit] = index
    return supernodes, qubit_to_supernode


def _seed_assignment_from_previous(
    supernodes: list[_Supernode],
    previous_qubit_assignment: list[int],
    num_pzs: int,
) -> tuple[list[int] | None, list[int] | None]:
    if not previous_qubit_assignment:
        return None, None

    cluster_assignment = [-1] * len(supernodes)
    cluster_loads = [0] * num_pzs
    for supernode in supernodes:
        previous_clusters: dict[int, int] = defaultdict(int)
        for qubit in supernode.qubits:
            previous_cluster = previous_qubit_assignment[qubit] if 0 <= qubit < len(previous_qubit_assignment) else -1
            if 0 <= previous_cluster < num_pzs:
                previous_clusters[previous_cluster] += 1
        if not previous_clusters:
            return None, None
        target_cluster = max(previous_clusters.items(), key=lambda item: item[1])[0]
        cluster_assignment[supernode.id] = target_cluster
        cluster_loads[target_cluster] += supernode.load
    return cluster_assignment, cluster_loads


def _greedy_initial_partition(
    supernodes: list[_Supernode],
    num_pzs: int,
    randomize_initial: bool,
    seed: int | None,
) -> tuple[list[int], list[int]]:
    if num_pzs <= 0:
        msg = "Number of processing zones must be positive for initial partitioning."
        raise ValueError(msg)

    cluster_assignment = [-1] * len(supernodes)
    cluster_loads = [0] * num_pzs
    if randomize_initial and seed is None:
        seed = random.randint(0, 1000)
    if randomize_initial and seed is not None:
        rng = random.Random(seed)
        for supernode in supernodes:
            target_cluster = rng.randrange(num_pzs)
            cluster_assignment[supernode.id] = target_cluster
            cluster_loads[target_cluster] += supernode.load
        return cluster_assignment, cluster_loads

    for supernode in sorted(supernodes, key=lambda item: item.load, reverse=True):
        target_cluster = min(range(num_pzs), key=lambda index: cluster_loads[index])
        cluster_assignment[supernode.id] = target_cluster
        cluster_loads[target_cluster] += supernode.load
    return cluster_assignment, cluster_loads


def _optimize_gate_partition(
    slice_contractions: Sequence[_SliceContraction],
    *,
    num_pzs: int,
    capacity: int | None,
    distance_matrix: DistanceMatrix | None,
    num_qubits: int,
    config: FineGrainedTabuConfig,
) -> tuple[list[list[int]], float, float]:
    if not slice_contractions or num_pzs <= 0:
        return [], 0.0, 0.0

    slack_weights = _build_slack_weights(slice_contractions, num_qubits, config.slack_dropoff)
    assignments_by_slice: list[list[int]] = []
    slice_counts: list[list[int]] = []
    slice_loads: list[list[int]] = []
    qubit_assignments_by_slice: list[list[int]] = [[-1] * num_qubits for _ in slice_contractions]
    active_counts_per_slice: list[dict[int, int]] = []
    active_loads_per_slice: list[dict[int, int]] = []

    for slice_index, contraction in enumerate(slice_contractions):
        if contraction.cluster_assignment is None:
            msg = "Slice contraction is missing an initial assignment."
            raise ValueError(msg)
        assignment_copy = contraction.cluster_assignment.copy()
        assignments_by_slice.append(assignment_copy)
        counts = [0] * num_pzs
        loads = [0] * num_pzs
        active_qubits = set(contraction.required_unary)
        for left, right in contraction.required_edges:
            active_qubits.add(left)
            active_qubits.add(right)

        active_counts: dict[int, int] = {}
        active_loads: dict[int, int] = {}
        for supernode in contraction.supernodes:
            cluster = assignment_copy[supernode.id]
            if cluster < 0 or cluster >= num_pzs:
                msg = f"Supernode {supernode.id} assigned to invalid cluster {cluster}."
                raise ValueError(msg)
            active_count = sum(1 for qubit in supernode.qubits if qubit in active_qubits)
            active_load = supernode.load if active_count > 0 else 0
            active_counts[supernode.id] = active_count
            active_loads[supernode.id] = active_load
            counts[cluster] += active_count
            loads[cluster] += active_load
            for qubit in supernode.qubits:
                if 0 <= qubit < num_qubits:
                    qubit_assignments_by_slice[slice_index][qubit] = cluster

        slice_counts.append(counts)
        slice_loads.append(loads)
        active_counts_per_slice.append(active_counts)
        active_loads_per_slice.append(active_loads)

    capacity_cost = _compute_capacity_cost(slice_counts, slice_loads, capacity)
    distance_cost = _compute_distance_cost(qubit_assignments_by_slice, distance_matrix, slack_weights)
    balance_cost = _compute_balance_cost(slice_loads)
    current_cost = (
        config.distance_weight_factor * distance_cost
        + config.capacity_weight * capacity_cost
        + config.balance_penalty * balance_cost
    )
    initial_cost = current_cost
    best_cost = current_cost
    best_assignments = [assignment.copy() for assignment in assignments_by_slice]

    tabu_list: list[tuple[int, int, int]] = []
    candidate_list_length = config.candidate_list_length if config.candidate_list_length and config.candidate_list_length > 0 else None
    per_slice_quota = config.per_slice_quota if config.per_slice_quota and config.per_slice_quota > 0 else None
    refresh_every = config.refresh_every if config.refresh_every and config.refresh_every > 0 else None
    candidate_k: int | None = None
    candidate_pool: list[tuple[int, int]] | None = None
    candidate_scores_by_slice: list[list[float]] | None = None
    total_candidate_pairs = sum(len(contraction.supernodes) for contraction in slice_contractions)

    if candidate_list_length is not None:
        candidate_k = min(max(candidate_list_length, 1), total_candidate_pairs)
        candidate_scores_by_slice = _build_candidate_scores(
            slice_contractions,
            qubit_assignments_by_slice,
            slice_counts,
            slice_loads,
            capacity,
            config.balance_penalty,
            config.capacity_weight,
            distance_matrix,
        )
        candidate_pool = _build_candidate_pool(candidate_scores_by_slice, candidate_k, per_slice_quota)

    max_iterations = config.resolve_max_iterations(num_qubits)
    tabu_list_length = max(config.tabu_list_length, 1)

    for iteration in range(max_iterations):
        if (
            candidate_list_length is not None
            and refresh_every is not None
            and iteration > 0
            and iteration % refresh_every == 0
        ):
            if candidate_k is None:
                candidate_k = max(candidate_list_length, 1)
            candidate_scores_by_slice = _build_candidate_scores(
                slice_contractions,
                qubit_assignments_by_slice,
                slice_counts,
                slice_loads,
                capacity,
                config.balance_penalty,
                config.capacity_weight,
                distance_matrix,
            )
            candidate_pool = _build_candidate_pool(candidate_scores_by_slice, candidate_k, per_slice_quota)

        best_move: tuple[int, _Supernode, int] | None = None
        best_move_score = math.inf
        best_move_capacity_delta = 0.0
        best_move_distance_delta = 0.0

        while True:
            move_sources = candidate_pool
            if move_sources is None:
                for slice_index, contraction in enumerate(slice_contractions):
                    for supernode_id in range(len(contraction.supernodes)):
                        (
                            best_move,
                            best_move_score,
                            best_move_capacity_delta,
                            best_move_distance_delta,
                        ) = _consider_supernode_moves(
                            contraction=contraction,
                            slice_index=slice_index,
                            supernode_id=supernode_id,
                            num_pzs=num_pzs,
                            slice_counts=slice_counts,
                            slice_loads=slice_loads,
                            active_counts_per_slice=active_counts_per_slice,
                            active_loads_per_slice=active_loads_per_slice,
                            qubit_assignments_by_slice=qubit_assignments_by_slice,
                            distance_matrix=distance_matrix,
                            slack_weights=slack_weights,
                            capacity=capacity,
                            config=config,
                            current_cost=current_cost,
                            best_cost=best_cost,
                            tabu_list=tabu_list,
                            best_move_state=(best_move, best_move_score, best_move_capacity_delta, best_move_distance_delta),
                        )
            else:
                for slice_index, supernode_id in move_sources:
                    contraction = slice_contractions[slice_index]
                    (
                        best_move,
                        best_move_score,
                        best_move_capacity_delta,
                        best_move_distance_delta,
                    ) = _consider_supernode_moves(
                        contraction=contraction,
                        slice_index=slice_index,
                        supernode_id=supernode_id,
                        num_pzs=num_pzs,
                        slice_counts=slice_counts,
                        slice_loads=slice_loads,
                        active_counts_per_slice=active_counts_per_slice,
                        active_loads_per_slice=active_loads_per_slice,
                        qubit_assignments_by_slice=qubit_assignments_by_slice,
                        distance_matrix=distance_matrix,
                        slack_weights=slack_weights,
                        capacity=capacity,
                        config=config,
                        current_cost=current_cost,
                        best_cost=best_cost,
                        tabu_list=tabu_list,
                        best_move_state=(best_move, best_move_score, best_move_capacity_delta, best_move_distance_delta),
                    )

            if best_move is not None or candidate_list_length is None:
                break
            if candidate_k is None or candidate_k >= total_candidate_pairs:
                break
            candidate_k = min(total_candidate_pairs, candidate_k * 2)
            if candidate_scores_by_slice is None:
                candidate_scores_by_slice = _build_candidate_scores(
                    slice_contractions,
                    qubit_assignments_by_slice,
                    slice_counts,
                    slice_loads,
                    capacity,
                    config.balance_penalty,
                    config.capacity_weight,
                    distance_matrix,
                )
            candidate_pool = _build_candidate_pool(candidate_scores_by_slice, candidate_k, per_slice_quota)

        if best_move is None:
            break

        slice_index, supernode, target_cluster = best_move
        previous_cluster = assignments_by_slice[slice_index][supernode.id]
        assignments_by_slice[slice_index][supernode.id] = target_cluster
        active_count = active_counts_per_slice[slice_index].get(supernode.id, 0)
        active_load = active_loads_per_slice[slice_index].get(supernode.id, 0)
        slice_counts[slice_index][previous_cluster] -= active_count
        slice_counts[slice_index][target_cluster] += active_count
        slice_loads[slice_index][previous_cluster] -= active_load
        slice_loads[slice_index][target_cluster] += active_load
        for qubit in supernode.qubits:
            if 0 <= qubit < num_qubits:
                qubit_assignments_by_slice[slice_index][qubit] = target_cluster

        capacity_cost += best_move_capacity_delta
        distance_cost += best_move_distance_delta
        balance_cost += _balance_delta(
            slice_loads[slice_index],
            previous_cluster,
            target_cluster,
            active_load,
            num_pzs,
        )
        current_cost = best_move_score

        tabu_list.append((slice_index, supernode.id, previous_cluster))
        if len(tabu_list) > tabu_list_length:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best_cost = current_cost
            best_assignments = [assignment.copy() for assignment in assignments_by_slice]

        if candidate_list_length is not None and refresh_every is None:
            if candidate_k is None:
                candidate_k = max(candidate_list_length, 1)
            if candidate_scores_by_slice is None:
                candidate_scores_by_slice = _build_candidate_scores(
                    slice_contractions,
                    qubit_assignments_by_slice,
                    slice_counts,
                    slice_loads,
                    capacity,
                    config.balance_penalty,
                    config.capacity_weight,
                    distance_matrix,
                )
            affected_slices = {slice_index}
            if slice_index > 0:
                affected_slices.add(slice_index - 1)
            if slice_index < len(slice_contractions) - 1:
                affected_slices.add(slice_index + 1)
            candidate_scores_by_slice = _update_candidate_scores(
                slice_contractions,
                qubit_assignments_by_slice,
                slice_counts,
                slice_loads,
                capacity,
                config.balance_penalty,
                config.capacity_weight,
                distance_matrix,
                candidate_scores_by_slice,
                sorted(affected_slices),
            )
            candidate_pool = _build_candidate_pool(candidate_scores_by_slice, candidate_k, per_slice_quota)

    return best_assignments if best_assignments else assignments_by_slice, initial_cost, best_cost

def _consider_supernode_moves(
    *,
    contraction: _SliceContraction,
    slice_index: int,
    supernode_id: int,
    num_pzs: int,
    slice_counts: Sequence[Sequence[int]],
    slice_loads: Sequence[Sequence[int]],
    active_counts_per_slice: Sequence[dict[int, int]],
    active_loads_per_slice: Sequence[dict[int, int]],
    qubit_assignments_by_slice: Sequence[Sequence[int]],
    distance_matrix: DistanceMatrix | None,
    slack_weights: Sequence[Sequence[float]] | None,
    capacity: int | None,
    config: FineGrainedTabuConfig,
    current_cost: float,
    best_cost: float,
    tabu_list: Sequence[tuple[int, int, int]],
    best_move_state: tuple[tuple[int, _Supernode, int] | None, float, float, float],
) -> tuple[tuple[int, _Supernode, int] | None, float, float, float]:
    best_move, best_move_score, best_move_capacity_delta, best_move_distance_delta = best_move_state
    supernode = contraction.supernodes[supernode_id]
    current_cluster = qubit_assignments_by_slice[slice_index][supernode.qubits[0]]

    for target_cluster in range(num_pzs):
        if target_cluster == current_cluster:
            continue
        active_count = active_counts_per_slice[slice_index].get(supernode.id, 0)
        active_load = active_loads_per_slice[slice_index].get(supernode.id, 0)
        capacity_delta = _capacity_delta(
            slice_counts[slice_index],
            slice_loads[slice_index],
            current_cluster,
            target_cluster,
            active_count,
            active_load,
            capacity,
        )
        distance_delta = _distance_delta(
            slice_index,
            supernode.qubits,
            current_cluster,
            target_cluster,
            qubit_assignments_by_slice,
            distance_matrix,
            slack_weights,
        )
        balance_delta = _balance_delta(
            slice_loads[slice_index],
            current_cluster,
            target_cluster,
            active_load,
            num_pzs,
        )
        move_delta = (
            config.distance_weight_factor * distance_delta
            + config.capacity_weight * capacity_delta
            + config.balance_penalty * balance_delta
        )
        candidate_cost = current_cost + move_delta
        move_key = (slice_index, supernode.id, target_cluster)
        if move_key in tabu_list and candidate_cost >= best_cost:
            continue
        if candidate_cost < best_move_score:
            best_move = (slice_index, supernode, target_cluster)
            best_move_score = candidate_cost
            best_move_capacity_delta = capacity_delta
            best_move_distance_delta = distance_delta

    return (
        best_move,
        best_move_score,
        best_move_capacity_delta,
        best_move_distance_delta,
    )


def _capacity_delta(
    count_vector: Sequence[int],
    load_vector: Sequence[int],
    src_cluster: int,
    dst_cluster: int,
    supernode_count: int,
    supernode_load: int,
    capacity: int | None,
) -> float:
    if capacity is None:
        return 0.0
    if not (
        0 <= src_cluster < len(load_vector)
        and 0 <= dst_cluster < len(load_vector)
        and 0 <= src_cluster < len(count_vector)
        and 0 <= dst_cluster < len(count_vector)
    ):
        return 0.0

    before = (
        (count_vector[src_cluster] - capacity) * load_vector[src_cluster]
        if count_vector[src_cluster] > capacity
        else 0.0
    ) + (
        (count_vector[dst_cluster] - capacity) * load_vector[dst_cluster]
        if count_vector[dst_cluster] > capacity
        else 0.0
    )
    after_src_count = count_vector[src_cluster] - supernode_count
    after_dst_count = count_vector[dst_cluster] + supernode_count
    after = (
        (after_src_count - capacity) * (load_vector[src_cluster] - supernode_load)
        if after_src_count > capacity
        else 0.0
    ) + (
        (after_dst_count - capacity) * (load_vector[dst_cluster] + supernode_load)
        if after_dst_count > capacity
        else 0.0
    )
    return float(after - before)


def _distance_delta(
    slice_index: int,
    qubits: Sequence[int],
    current_cluster: int,
    target_cluster: int,
    qubit_assignments_by_slice: Sequence[Sequence[int]],
    distance_matrix: DistanceMatrix | None,
    slack_weights: Sequence[Sequence[float]] | None,
) -> float:
    delta = 0.0
    num_slices = len(qubit_assignments_by_slice)
    if num_slices <= 1:
        return 0.0

    current_slice_assignments = qubit_assignments_by_slice[slice_index]
    if slice_index > 0:
        previous_assignments = qubit_assignments_by_slice[slice_index - 1]
        for qubit in qubits:
            if qubit < 0 or qubit >= len(current_slice_assignments):
                continue
            previous_cluster = previous_assignments[qubit]
            current = current_slice_assignments[qubit]
            weight = slack_weights[slice_index - 1][qubit] if slack_weights is not None else 1.0
            delta += weight * (
                _distance_between_clusters(previous_cluster, target_cluster, distance_matrix)
                - _distance_between_clusters(previous_cluster, current, distance_matrix)
            )
    if slice_index < num_slices - 1:
        next_assignments = qubit_assignments_by_slice[slice_index + 1]
        for qubit in qubits:
            if qubit < 0 or qubit >= len(current_slice_assignments):
                continue
            next_cluster = next_assignments[qubit]
            current = current_slice_assignments[qubit]
            weight = slack_weights[slice_index][qubit] if slack_weights is not None else 1.0
            delta += weight * (
                _distance_between_clusters(target_cluster, next_cluster, distance_matrix)
                - _distance_between_clusters(current, next_cluster, distance_matrix)
            )
    return delta


def _compute_capacity_cost(
    slice_counts: Sequence[Sequence[int]],
    slice_loads: Sequence[Sequence[int]],
    capacity: int | None,
) -> float:
    if capacity is None:
        return 0.0
    total = 0.0
    for counts, loads in zip(slice_counts, slice_loads):
        for count, load in zip(counts, loads):
            if count > capacity:
                total += (count - capacity) * load
    return float(total)


def _compute_balance_cost(slice_loads: Sequence[Sequence[int]]) -> float:
    total = 0.0
    for loads in slice_loads:
        if not loads:
            continue
        mean_load = sum(loads) / len(loads)
        total += sum(max(0.0, load - mean_load) for load in loads)
    return float(total)


def _balance_delta(
    load_vector: Sequence[int],
    src_cluster: int,
    dst_cluster: int,
    supernode_load: int,
    num_pzs: int,
) -> float:
    if not (0 <= src_cluster < len(load_vector) and 0 <= dst_cluster < len(load_vector)):
        return 0.0
    if num_pzs <= 0:
        return 0.0

    mean_load = sum(load_vector) / num_pzs
    before = max(0.0, load_vector[src_cluster] - mean_load) + max(0.0, load_vector[dst_cluster] - mean_load)
    after = max(0.0, load_vector[src_cluster] - supernode_load - mean_load) + max(
        0.0,
        load_vector[dst_cluster] + supernode_load - mean_load,
    )
    return float(after - before)


def _compute_distance_cost(
    qubit_assignments_by_slice: Sequence[Sequence[int]],
    distance_matrix: DistanceMatrix | None,
    slack_weights: Sequence[Sequence[float]] | None = None,
) -> float:
    if len(qubit_assignments_by_slice) <= 1:
        return 0.0
    total = 0.0
    for slice_index in range(len(qubit_assignments_by_slice) - 1):
        current = qubit_assignments_by_slice[slice_index]
        nxt = qubit_assignments_by_slice[slice_index + 1]
        for qubit, current_cluster in enumerate(current):
            next_cluster = nxt[qubit] if qubit < len(nxt) else -1
            weight = slack_weights[slice_index][qubit] if slack_weights is not None else 1.0
            total += weight * _distance_between_clusters(current_cluster, next_cluster, distance_matrix)
    return float(total)


def _distance_between_clusters(
    src_cluster: int,
    dst_cluster: int,
    distance_matrix: DistanceMatrix | None,
) -> float:
    if src_cluster == dst_cluster or src_cluster < 0 or dst_cluster < 0:
        return 0.0
    if (
        distance_matrix is not None
        and 0 <= src_cluster < len(distance_matrix)
        and 0 <= dst_cluster < len(distance_matrix[src_cluster])
    ):
        return distance_matrix[src_cluster][dst_cluster]
    return 1.0


def _build_slack_weights(
    slice_contractions: Sequence[_SliceContraction],
    num_qubits: int,
    slack_dropoff: float | None,
) -> list[list[float]] | None:
    if not slack_dropoff or slack_dropoff <= 0:
        return None
    num_slices = len(slice_contractions)
    if num_slices <= 1:
        return []

    last_active_per_slice: list[list[int]] = []
    last_active = [-1] * num_qubits
    for slice_index, contraction in enumerate(slice_contractions):
        active_qubits = set(contraction.required_unary)
        for left, right in contraction.required_edges:
            active_qubits.add(left)
            active_qubits.add(right)
        for qubit in active_qubits:
            if 0 <= qubit < num_qubits:
                last_active[qubit] = slice_index
        last_active_per_slice.append(list(last_active))

    next_active_per_slice: list[list[int]] = []
    next_active = [num_slices] * num_qubits
    for slice_index in range(num_slices - 1, -1, -1):
        contraction = slice_contractions[slice_index]
        active_qubits = set(contraction.required_unary)
        for left, right in contraction.required_edges:
            active_qubits.add(left)
            active_qubits.add(right)
        for qubit in active_qubits:
            if 0 <= qubit < num_qubits:
                next_active[qubit] = slice_index
        next_active_per_slice.insert(0, list(next_active))

    weights: list[list[float]] = []
    for slice_index in range(num_slices - 1):
        edge_weights = [1.0] * num_qubits
        last_active_at = last_active_per_slice[slice_index]
        next_active_at = next_active_per_slice[slice_index + 1]
        for qubit in range(num_qubits):
            window = max(0, next_active_at[qubit] - last_active_at[qubit] - 1)
            edge_weights[qubit] = 1.0 / (1.0 + slack_dropoff * window)
        weights.append(edge_weights)
    return weights


def _compute_moves(assignments: list[list[int]]) -> list[list[tuple[int, int, int]]]:
    moves: list[list[tuple[int, int, int]]] = []
    for previous, current in zip(assignments, assignments[1:]):
        slice_moves: list[tuple[int, int, int]] = []
        for qubit, (src, dst) in enumerate(zip(previous, current)):
            if src != dst:
                slice_moves.append((qubit, src, dst))
        moves.append(slice_moves)
    return moves


def _compute_total_move_distance(
    moves: list[list[tuple[int, int, int]]],
    distance_matrix: DistanceMatrix | None,
) -> float:
    total = 0.0
    for slice_moves in moves:
        for _, src, dst in slice_moves:
            total += _distance_between_clusters(src, dst, distance_matrix)
    return total


def _build_qubit_assignment(
    contraction: _SliceContraction,
    num_qubits: int,
    num_clusters: int,
) -> list[int] | None:
    if contraction.cluster_assignment is None:
        return None
    qubit_assignment = [-1] * num_qubits
    for supernode in contraction.supernodes:
        cluster = contraction.cluster_assignment[supernode.id]
        if cluster < 0 or cluster >= num_clusters:
            continue
        for qubit in supernode.qubits:
            if 0 <= qubit < num_qubits:
                qubit_assignment[qubit] = cluster
    return qubit_assignment


def _compute_cluster_loads(contraction: _SliceContraction, num_pzs: int) -> list[int]:
    loads = [0] * num_pzs
    if contraction.cluster_assignment is None:
        return loads
    for supernode in contraction.supernodes:
        cluster = contraction.cluster_assignment[supernode.id]
        if 0 <= cluster < num_pzs:
            loads[cluster] += supernode.load
    return loads


def _compute_supernode_scores_for_slice(
    contraction: _SliceContraction,
    slice_index: int,
    qubit_assignments_by_slice: Sequence[Sequence[int]],
    slice_counts: Sequence[Sequence[int]],
    slice_loads: Sequence[Sequence[int]],
    capacity: int | None,
    balance_penalty: float,
    capacity_weight: float,
    distance_matrix: DistanceMatrix | None,
) -> list[float]:
    scores = [0.0] * len(contraction.supernodes)
    current = qubit_assignments_by_slice[slice_index]
    previous = qubit_assignments_by_slice[slice_index - 1] if slice_index > 0 else None
    nxt = qubit_assignments_by_slice[slice_index + 1] if slice_index < len(qubit_assignments_by_slice) - 1 else None
    counts = slice_counts[slice_index]
    loads = slice_loads[slice_index]

    for supernode in contraction.supernodes:
        total = 0.0
        for qubit in supernode.qubits:
            if qubit < 0 or qubit >= len(current):
                continue
            current_cluster = current[qubit]
            if previous is not None:
                total += _distance_between_clusters(previous[qubit], current_cluster, distance_matrix)
            if nxt is not None:
                total += _distance_between_clusters(current_cluster, nxt[qubit], distance_matrix)

        cluster = current[supernode.qubits[0]] if supernode.qubits else -1
        overflow_penalty = 0.0
        if (
            capacity is not None
            and 0 <= cluster < len(loads)
            and 0 <= cluster < len(counts)
            and counts[cluster] > capacity
        ):
            overflow_penalty = (counts[cluster] - capacity) * loads[cluster]
        total += capacity_weight * overflow_penalty

        mean_load = sum(loads) / len(loads) if loads else 0.0
        total += balance_penalty * max(0.0, loads[cluster] - mean_load)
        scores[supernode.id] = total
    return scores


def _build_candidate_scores(
    slice_contractions: Sequence[_SliceContraction],
    qubit_assignments_by_slice: Sequence[Sequence[int]],
    slice_counts: Sequence[Sequence[int]],
    slice_loads: Sequence[Sequence[int]],
    capacity: int | None,
    balance_penalty: float,
    capacity_weight: float,
    distance_matrix: DistanceMatrix | None,
) -> list[list[float]]:
    return [
        _compute_supernode_scores_for_slice(
            contraction,
            slice_index,
            qubit_assignments_by_slice,
            slice_counts,
            slice_loads,
            capacity,
            balance_penalty,
            capacity_weight,
            distance_matrix,
        )
        for slice_index, contraction in enumerate(slice_contractions)
    ]


def _update_candidate_scores(
    slice_contractions: Sequence[_SliceContraction],
    qubit_assignments_by_slice: Sequence[Sequence[int]],
    slice_counts: Sequence[Sequence[int]],
    slice_loads: Sequence[Sequence[int]],
    capacity: int | None,
    balance_penalty: float,
    capacity_weight: float,
    distance_matrix: DistanceMatrix | None,
    scores_by_slice: list[list[float]],
    slice_indices: Sequence[int],
) -> list[list[float]]:
    for slice_index in slice_indices:
        scores_by_slice[slice_index] = _compute_supernode_scores_for_slice(
            slice_contractions[slice_index],
            slice_index,
            qubit_assignments_by_slice,
            slice_counts,
            slice_loads,
            capacity,
            balance_penalty,
            capacity_weight,
            distance_matrix,
        )
    return scores_by_slice


def _build_candidate_pool(
    scores_by_slice: Sequence[Sequence[float]],
    candidate_k: int,
    per_slice_quota: int | None,
) -> list[tuple[int, int]]:
    ranked: list[tuple[float, int, int]] = []
    for slice_index, scores in enumerate(scores_by_slice):
        for supernode_id, score in enumerate(scores):
            ranked.append((score, slice_index, supernode_id))
    ranked.sort(reverse=True, key=lambda item: item[0])

    pool: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for _, slice_index, supernode_id in ranked[:candidate_k]:
        key = (slice_index, supernode_id)
        if key in seen:
            continue
        seen.add(key)
        pool.append(key)

    if per_slice_quota and per_slice_quota > 0:
        for slice_index, scores in enumerate(scores_by_slice):
            per_slice_ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            for supernode_id in per_slice_ranked[:per_slice_quota]:
                key = (slice_index, supernode_id)
                if key in seen:
                    continue
                seen.add(key)
                pool.append(key)
    return pool


def _build_gate_partition_projection(
    slice_contractions: Sequence[_SliceContraction],
    time_slices: Sequence[Sequence[int]],
    gate_info: dict[int, GateInfo],
    pz_names: Sequence[str],
    *,
    num_qubits: int,
) -> tuple[list[list[int]], dict[str, list[int]], dict[int, str]]:
    gate_partition_by_pz = {pz_name: [] for pz_name in pz_names}
    gate_assignment: dict[int, str] = {}
    qubit_assignments_by_slice: list[list[int]] = []
    num_clusters = len(pz_names)

    if len(slice_contractions) != len(time_slices):
        msg = "Slice contractions and time slices must have matching length."
        raise ValueError(msg)

    for contraction, slice_gate_ids in zip(slice_contractions, time_slices):
        if contraction.cluster_assignment is None:
            msg = "Slice contraction is missing a finalized assignment."
            raise ValueError(msg)
        if len(contraction.cluster_assignment) != len(contraction.supernodes):
            msg = "Cluster assignment length mismatch when projecting the result."
            raise ValueError(msg)

        qubit_assignment = [-1] * num_qubits
        for supernode in contraction.supernodes:
            cluster = contraction.cluster_assignment[supernode.id]
            if cluster < 0 or cluster >= num_clusters:
                msg = f"Supernode {supernode.id} assigned to invalid cluster {cluster}."
                raise ValueError(msg)
            for qubit in supernode.qubits:
                qubit_assignment[qubit] = cluster
        qubit_assignments_by_slice.append(qubit_assignment)

        for gate_id in slice_gate_ids:
            qubits = gate_info[gate_id].qubits
            if not qubits:
                continue
            cluster = qubit_assignment[qubits[0]]
            if cluster < 0 or cluster >= num_clusters:
                msg = f"Gate {gate_id} references an unassigned qubit {qubits[0]}."
                raise ValueError(msg)
            if any(qubit_assignment[qubit] != cluster for qubit in qubits[1:]):
                msg = f"Gate {gate_id} spans multiple clusters after refinement."
                raise ValueError(msg)
            pz_name = pz_names[cluster]
            gate_partition_by_pz[pz_name].append(gate_id)
            gate_assignment[gate_id] = pz_name

    return qubit_assignments_by_slice, gate_partition_by_pz, gate_assignment


__all__ = [
    "FineGrainedTabuConfig",
    "GatePartitionResult",
    "compute_fine_grained_gate_partition",
]