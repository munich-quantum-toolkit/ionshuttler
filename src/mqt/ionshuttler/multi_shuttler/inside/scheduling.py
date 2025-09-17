from __future__ import annotations

import contextlib
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

import numpy as np
from more_itertools import distinct_combinations, pairwise

from .cycles import (
    check_if_edge_is_filled,
    create_cycle,
    find_next_edge,
    find_ordered_edges,
    find_path_edge_to_edge,
    get_edge_state,
    get_ion_chains,
    have_common_junction_node,
)
from .graph_utils import get_idx_from_idc
from .paths import create_path_via_bfs_directional, find_nonfree_paths

if TYPE_CHECKING:
    from .graph import Graph
    from .processing_zone import ProcessingZone
    from .types import Edge, Node


def preprocess(graph: Graph, priority_queue: dict[int, str]) -> None:
    need_rotate = [False] * len(priority_queue)
    while sum(need_rotate) < len(priority_queue):
        for i, rotate_chain in enumerate(priority_queue):
            pz_name = priority_queue[rotate_chain]
            pz_edge_idc = next(pz.edge_idc for pz in graph.pzs if pz.name == pz_name)
            edge_idc = graph.state[rotate_chain]
            next_edge_ = find_next_edge(graph, edge_idc, pz_edge_idc)
            next_node1, next_node2 = tuple(sorted(next_edge_, key=sum))
            next_edge = (next_node1, next_node2)

            state_edges_idc = get_edge_state(graph)

            # if next_edge is free, not a stop move (same edge) and not at junction node
            # move the ion to the next edge
            if (
                have_common_junction_node(graph, edge_idc, next_edge) is False
                and state_edges_idc[next_edge] == []
                and edge_idc != next_edge
            ):
                graph.edges[edge_idc]["ions"].remove(rotate_chain)
                graph.edges[next_edge]["ions"].append(rotate_chain)
            else:
                need_rotate[i] = True


# def create_general_priority_queue(graph, sequence, max_length=10):
#     logging.debug("Entered create_priority_queue function")
#     # TODO create real flat unique sequence?
#     unique_sequence = []
#     for seq_elem in sequence:
#         for elem in seq_elem:
#             if elem not in unique_sequence:
#                 unique_sequence.append(elem)
#                 if len(unique_sequence) == max_length:
#                     break
#     logging.debug(f"Priority queue created: {unique_sequence}")
#     return unique_sequence


def get_edge_idc_by_pz_name(graph: Graph, pz_name: str) -> Edge:
    for pz in graph.pzs:
        if pz.name == pz_name:
            return pz.edge_idc
    msg = f"Processing zone with name {pz_name} not found."
    raise ValueError(msg)


def pick_pz_for_2_q_gate(graph: Graph, ion0: int, ion1: int) -> str:
    # TODO implement a better way to pick the processing zone for 2-qubit gates
    # pick the processing zone that both ions are closest to (so sum of distances is minimal)
    min_distance = float("inf")
    for pz_name in [graph.map_to_pz[ion0], graph.map_to_pz[ion1]]:
        pz_edge_idc = get_edge_idc_by_pz_name(graph, pz_name)
        distance = len(find_path_edge_to_edge(graph, graph.state[ion0], pz_edge_idc)) + len(
            find_path_edge_to_edge(graph, graph.state[ion1], pz_edge_idc)
        )
        if distance < min_distance:
            min_distance = distance
            closest_pz = pz_name
    return closest_pz


def create_priority_queue(
    graph: Graph,
    sequence: list[tuple[int, ...]],
    max_length: int = 10,
) -> tuple[dict[int, str], dict[str, tuple[int, ...]]]:
    """
    Create a priority queue based on a given graph and sequence of gates.
    Also creates a dictionary of the next gate of each processing zone.

    Args:
        graph: The graph representing the QCCD architecture.
        sequence: The sequence of gates.
        max_length: The maximum length of the priority queue. Defaults to 10.

    Returns:
        - The priority queue
        - The next gate at each processing zone
    """
    unique_sequence: dict[int, str] = OrderedDict()
    next_gate_at_pz: dict[str, tuple[int, ...]] = {}
    for seq_elem in sequence:
        # 1-qubit gate
        if len(seq_elem) == 1:
            elem = seq_elem[0]

            # add first gate of pz to next_gate_at_pz (if not already there)
            if graph.map_to_pz[elem] not in next_gate_at_pz:
                next_gate_at_pz[graph.map_to_pz[elem]] = seq_elem

            # add ion to unique_sequence
            if elem not in unique_sequence:
                unique_sequence[elem] = graph.map_to_pz[elem]
                if len(unique_sequence) == max_length:
                    break

        # 2-qubit gate
        elif len(seq_elem) == 2:
            if seq_elem not in graph.locked_gates:
                # pick processing zone for 2-qubit gate
                pz_for_2_q_gate = pick_pz_for_2_q_gate(graph, seq_elem[0], seq_elem[1])
                # new in multishuttler outside:
                # graph.locked_gates[seq_elem] = pz_for_2_q_gate
            else:
                pz_for_2_q_gate = graph.locked_gates[seq_elem]

            # add first gate of pz to next_gate_at_pz (if not already there)
            if pz_for_2_q_gate not in next_gate_at_pz:
                next_gate_at_pz[pz_for_2_q_gate] = seq_elem
                # lock the processing zone for the 2-qubit gate for later iterations
                # otherwise maybe pz changes if both move in a way, that favors a new pz
                # -> could result in a bug, if the very next iterations
                # change state back to old pz
                graph.locked_gates[seq_elem] = pz_for_2_q_gate

            # add ions to unique_sequence
            for elem in seq_elem:
                if elem not in unique_sequence:
                    unique_sequence[elem] = pz_for_2_q_gate
                    if len(unique_sequence) == max_length:
                        break
        else:
            msg = "len gate 0 or > 2? - can only process 1 or 2-qubit gates"
            raise ValueError(msg)

        # at the end fill all empty pzs with ()
        for pz in graph.pzs:
            if pz.name not in next_gate_at_pz:
                next_gate_at_pz[pz.name] = ()
    return unique_sequence, next_gate_at_pz


def get_partitioned_priority_queues(priority_queue: dict[int, str]) -> dict[str, list[int]]:
    # partitioned_priority_queue is a dictionary with pzs as keys and ions as values
    # represents priority queue for each processing zone individually
    # is just priority reversed (keys and values exchanged)
    partitioned_priority_queues = defaultdict(list)
    for key, value in priority_queue.items():
        # Append the key to the list of the corresponding value
        partitioned_priority_queues[value].append(key)

    # partitioned_priority_queues = {}
    # for pz in graph.pzs:
    #     partitioned_priority_queues[pz.name] = [
    #         elem for elem in priority_queue if elem in partition[pz.name]
    #     ]
    return partitioned_priority_queues


def create_gate_info_list(graph: Graph) -> dict[str, list[int]]:
    # create list of next gate at each processing zone
    gate_info_list: dict[str, list[int]] = {pz.name: [] for pz in graph.pzs}
    for seq_elem in graph.sequence:
        if len(seq_elem) == 1:
            elem = seq_elem[0]
            pz = graph.map_to_pz[elem]
            if gate_info_list[pz] == []:
                gate_info_list[pz].append(elem)
        elif len(seq_elem) == 2:
            # only pick processing zone for 2-qubit gate if not already locked
            if seq_elem not in graph.locked_gates:
                pz = pick_pz_for_2_q_gate(graph, seq_elem[0], seq_elem[1])
            else:
                pz = graph.locked_gates[seq_elem]
            if gate_info_list[pz] == []:
                gate_info_list[pz].append(seq_elem[0])
                gate_info_list[pz].append(seq_elem[1])
        # break if all pzs have a gate
        if all(gate_info_list.values()):
            break

    return gate_info_list


# if pz is occupied
# allow move onto pz?
# then fix situation?
# but change move list, so others move while first is in pz?
# created first_gate_at_pz dict -> can find out if next
# gate at a pz is 2-qubit gate -> then move over pz?


def create_move_list(graph: Graph, partitioned_priority_queue: list[int], pz: ProcessingZone) -> list[int]:
    ion_chains = get_ion_chains(graph)
    path_length_sequence = {}
    move_list: list[int] = []
    for i, rotate_chain in enumerate(partitioned_priority_queue):
        edge_idc = ion_chains[rotate_chain]
        # shortest path is also 1 edge if already at pz -> set to 0
        if edge_idc == pz.edge_idc:
            path_length_sequence[rotate_chain] = 0
        else:
            path_to_go = find_path_edge_to_edge(graph, edge_idc, pz.edge_idc)
            path_length_sequence[rotate_chain] = len(path_to_go)

        # if first ion or all paths are 0 (all ions to move are in pz already) or current path is longer than all other paths
        if (
            i == 0
            or sum(path_length_sequence.values()) == 0
            or sum(
                np.array([path_length_sequence[rotate_chain]] * len(move_list))
                > np.array([path_length_sequence[chain] for chain in move_list])
            )
            == len(move_list)
        ):
            move_list.append(rotate_chain)

    return move_list


def create_cycles_for_moves(
    graph: Graph,
    move_list: list[int],
    cycle_or_paths: str,
    pz: ProcessingZone,
) -> dict[int, list[Edge]]:
    all_cycles = {}
    ion_chains = graph.state
    for rotate_chain in move_list:
        edge_idc = ion_chains[rotate_chain]
        next_edge = find_next_edge(graph, edge_idc, pz.edge_idc)
        edge_idc, next_edge = find_ordered_edges(graph, edge_idc, next_edge)
        if not check_if_edge_is_filled(graph, next_edge) or edge_idc == next_edge:
            all_cycles[rotate_chain] = [edge_idc, next_edge]
        elif cycle_or_paths == "Cycles":
            all_cycles[rotate_chain] = create_cycle(graph, edge_idc, next_edge)
        else:
            # TODO: other next edges
            all_cycles[rotate_chain] = create_path_via_bfs_directional(graph, edge_idc, next_edge)
    return all_cycles


def find_conflict_cycle_idxs(graph: Graph, cycles_dict: dict[int, list[Edge]]) -> list[tuple[int, int]]:
    combinations_of_cycles = list(distinct_combinations(cycles_dict.keys(), 2))

    def get_cycle_nodes(cycle: int) -> set[Node]:
        # if cycle is two edges
        if len(cycles_dict[cycle]) == 2:
            # if not a stop move
            if cycles_dict[cycle][0] != cycles_dict[cycle][1]:
                cycle_or_path = [(cycles_dict[cycle][0][1], cycles_dict[cycle][1][0])]
                assert cycles_dict[cycle][0][1] == cycles_dict[cycle][1][0], (
                    "cycle is not two edges? Middle node should be the same"
                )
            # if a stop move and in stop moves (in pz for 2-qubit gate)
            elif cycle in graph.stop_moves:
                cycle_or_path = cycles_dict[cycle]
            else:
                cycle_or_path = []  # [(cycles_dict[cycle][0][0], cycles_dict[cycle][0][0])]
        elif cycles_dict[cycle][0] == cycles_dict[cycle][-1]:
            cycle_or_path = cycles_dict[cycle]
        else:
            cycle_or_path = cycles_dict[cycle]  # TODO should be only for paths, for cycles -> ValueError
            # raise ValueError("cycle is not two edges or a real cycle?")
        nodes = set()
        for edge in cycle_or_path:
            node1, node2 = edge
            if node1 == node2:
                nodes.add(node1)
            else:
                nodes.add(node1)
                nodes.add(node2)
        return nodes

    junction_shared_pairs = []
    for cycle1, cycle2 in combinations_of_cycles:
        nodes1 = get_cycle_nodes(cycle1)
        nodes2 = get_cycle_nodes(cycle2)
        if len(nodes1.intersection(nodes2)) > 0 or (
            get_idx_from_idc(graph.idc_dict, cycles_dict[cycle1][-1])
            == (get_idx_from_idc(graph.idc_dict, cycles_dict[cycle2][-1]))
        ):
            junction_shared_pairs.append((cycle1, cycle2))
    return junction_shared_pairs


def find_movable_cycles(
    graph: Graph,
    all_cycles: dict[int, list[Edge]],
    priority_queue: dict[int, str],
    cycle_or_paths: str,
) -> list[int]:
    print("all_cycles", all_cycles)
    if cycle_or_paths == "Cycles":
        nonfree_cycles = find_conflict_cycle_idxs(graph, all_cycles)
    else:
        nonfree_cycles = find_nonfree_paths(graph, all_cycles)

    # start with first ion in priority queue
    free_cycle_seq_idxs = [next(iter(priority_queue.keys()))]

    # check if ion can move while first ion is moving and so on
    for seq_cyc in list(priority_queue.keys())[1:]:
        # skip ion of priority_queue if it is not in all_cycles
        # -> was removed before in individual move_list
        if seq_cyc not in all_cycles:
            continue
        nonfree = False
        for mov_cyc in free_cycle_seq_idxs:
            if (seq_cyc, mov_cyc) in nonfree_cycles or (
                mov_cyc,
                seq_cyc,
            ) in nonfree_cycles:
                nonfree = True
                break
        if nonfree is False:
            free_cycle_seq_idxs.append(seq_cyc)
    return free_cycle_seq_idxs


def rotate(graph: Graph, ion: int, cycle_idcs: list[Edge]) -> None:
    # print(f"Rotating ion {ion} along cycle {cycle_idcs}", graph.in_process)
    state_dict = get_edge_state(graph)
    first = True
    last_ion = -1
    for current_edge, new_edge in pairwise(cycle_idcs):
        current_node1, current_node2 = tuple(sorted(current_edge, key=sum))
        current_edge_ = (current_node1, current_node2)

        new_node1, new_node2 = tuple(sorted(new_edge, key=sum))
        new_edge_ = (new_node1, new_node2)

        # take first ion in list to rotate
        # if len(current_ion) <= 1:
        current_ion_ = state_dict.get(current_edge_)
        if current_ion_ is not None:
            with contextlib.suppress(IndexError):
                current_ion = current_ion_[0]

        # if ion already rotated via previous cycle
        # (now checks directly in state_dict, in case two ions on one edge)
        if first and ion not in state_dict[current_edge_]:  # current_ion != ion:
            # print(f"Ion {ion} already rotated via previous cycle")
            return
        first = False
        if current_ion in graph.in_process:
            # print(f"didn't rotate {current_ion}")
            pass
        if (
            current_ion not in ([], last_ion) and current_ion not in graph.in_process
        ):  # and not ion in pz and needed in 2-qubit gate
            graph.edges[current_edge_]["ions"].remove(current_ion)
            graph.edges[new_edge_]["ions"].append(current_ion)
            # print(f"Rotated ion {current_ion} from {current_edge} to {new_edge}")

        # save last ion so each ion only rotates once
        last_ion = current_ion


def rotate_free_cycles(graph: Graph, all_cycles: dict[int, list[Edge]], free_cycles_idxs: list[int]) -> None:
    rotate_cycles_idcs = {}
    for cycle_ion in free_cycles_idxs:
        with contextlib.suppress(KeyError):
            rotate_cycles_idcs[cycle_ion] = all_cycles[cycle_ion]

    # skip stop moves
    for ion, indiv_cycle_idcs in rotate_cycles_idcs.items():
        if len(indiv_cycle_idcs) == 2 and indiv_cycle_idcs[0] == indiv_cycle_idcs[1]:
            # print(f"Skipping rotating ion {ion} along
            # cycle {indiv_cycle_idcs}, since it is a stop move")
            continue
        # print(f"Rotating ion {ion} along cycle {indiv_cycle_idcs}")
        rotate(graph, ion, indiv_cycle_idcs)
