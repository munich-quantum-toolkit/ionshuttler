from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from more_itertools import distinct_combinations

from .cycles import check_if_edge_is_filled
from .graph_utils import get_idc_from_idx, get_idx_from_idc

if TYPE_CHECKING:
    from .graph import Graph
    from .types import Edge, Node


# BFS with direction based on a starting edge and a next edge
def create_path_via_bfs_directional(
    graph: Graph,
    current_edge: Edge,
    next_edge: Edge,
    towards: Node = (0, 0),
) -> list[Edge]:
    if towards == (0, 0):
        # towards is first edge in graph (can't be (0, 0) because it may be deleted)
        towards = next(iter(graph.edges()))[0]

    # move from entry to memory zone
    # if get_idx_from_idc(graph.idc_dict, current_edge) == get_idx_from_idc(
    #     graph.idc_dict, graph.pzgraph_creator.entry_edge
    # ):  # in graph.pzgraph_creator.path_from_pz_idxs:
    #     target_edge = graph.bfs_free_edge(towards, other_next_edges)
    #     # calc path to target edge
    #     path0 = get_path_to_node(
    #         graph,
    #         graph.pzgraph_creator.processing_zone,
    #         target_edge[0],
    #         exclude_exit=True,
    #         exclude_first_entry_connection=False,
    #     )
    #     path1 = get_path_to_node(
    #         graph,
    #         graph.pzgraph_creator.processing_zone,
    #         target_edge[1],
    #         exclude_exit=True,
    #         exclude_first_entry_connection=False,
    #     )
    #     if len(path1) > len(path0):
    #         edge_path = [*path0, (target_edge[0], target_edge[1])]
    #     else:
    #         edge_path = [*path1, (target_edge[1], target_edge[0])]
    #     return edge_path

    # Define the starting node (the middle node where edges meet)
    common_node, next_node = (
        (next_edge[0], next_edge[1]) if next_edge[0] in current_edge else (next_edge[1], next_edge[0])
    )

    # Perform BFS starting from the middle node
    queue = [(next_node, [common_node])]
    visited = set()
    visited.add(common_node)

    while queue:
        current_node, path = queue.pop(0)

        if current_node in visited:
            continue
        visited.add(current_node)

        # Explore neighbors
        # not already visited and not entry as goal (can't enter via entry - so not entry edge and not a first entry connection node)
        for neighbor in [
            node
            for node in graph.neighbors(current_node)
            if node not in visited
            and nx.get_node_attributes(graph, "node_type")[node]
            not in {"entry_connection_node", "processing_zone_node"}
        ]:
            # if get_idx_from_idc(graph.idc_dict, (current_node, neighbor)) != get_idx_from_idc(graph.idc_dict, graph.pzgraph_creator.entry_edge):

            edge = (current_node, neighbor)

            # Check if the edge is free
            if not check_if_edge_is_filled(graph, edge):
                path_to_edge = [*path, current_node, neighbor]
                # return as list of edges
                return [current_edge, *[(path_to_edge[i], path_to_edge[i + 1]) for i in range(len(path_to_edge) - 1)]]

            # Continue BFS
            if neighbor not in visited:
                queue.append((neighbor, [*path, current_node]))

    msg = f"No path found for edge {current_edge}"
    raise RuntimeError(msg)


def find_nonfree_paths(graph: Graph, paths_idcs_dict: dict[int, list[Edge]]) -> list[tuple[int, int]]:
    # TODO change pz (uncomment)
    paths_idxs_dict = {}
    for key, value in paths_idcs_dict.items():
        paths_idxs_dict[key] = {get_idx_from_idc(graph.idc_dict, edge_idc) for edge_idc in value}

    common_edges = set()
    conflicting_paths = []

    combinations_of_paths = list(distinct_combinations(paths_idcs_dict.keys(), 2))

    # Compare every pair of paths to check for common edges (with edge_IDX)
    for path_ion_1, path_ion_2 in combinations_of_paths:
        intersection = paths_idxs_dict[path_ion_1].intersection(paths_idxs_dict[path_ion_2])
        # Skip if the intersection is the exit or entry edge -> allows ions to move to exit and entry right after another ion
        # remove exit and entry edges from intersection -> can push through to parking edge -> conflicts are managed in scheduling.py - create_circles_for_moves()
        intersection = {
            edge
            for edge in intersection
            if graph.get_edge_data(*get_idc_from_idx(graph.idc_dict, edge))["edge_type"]
            not in {"exit", "entry", "first_entry_connection"}
        }

        # Store the common edges and the conflicting paths
        if intersection:
            common_edges.update(intersection)
            conflicting_paths.append((path_ion_1, path_ion_2))  # Store indices of conflicting paths
    print("conflicting_paths edges\n", conflicting_paths)
    # Compare junction nodes (with edge_IDC)
    junction_nodes = [*graph.junction_nodes]  # , graph.pzgraph_creator.processing_zone] TODO

    for path_ion_1, path_ion_2 in combinations_of_paths:
        if len(paths_idcs_dict[path_ion_1]) == 2:
            # if same edge twice -> skip (no edge if twice parking edge, otherwise only first node)
            if paths_idcs_dict[path_ion_1][0] == paths_idcs_dict[path_ion_1][1]:
                # TODO
                # if get_idx_from_idc(graph.idc_dict, paths_idcs_dict[path_ion_1][0]) == get_idx_from_idc(
                #     graph.idc_dict, graph.pzgraph_creator.parking_edge
                if (
                    graph.get_edge_data(
                        *get_idc_from_idx(
                            graph.idc_dict, get_idx_from_idc(graph.idc_dict, paths_idcs_dict[path_ion_1][0])
                        )
                    )["edge_type"]
                    == "processing"
                ):
                    nodes1 = set()
                else:
                    nodes1 = {paths_idcs_dict[path_ion_1][0][0]}
            else:
                # path - only middle node of path (two edges)
                nodes1 = {paths_idcs_dict[path_ion_1][0][1]}
                assert paths_idcs_dict[path_ion_1][0][1] == paths_idcs_dict[path_ion_1][1][0], (
                    f"not middle node? {paths_idcs_dict[path_ion_1]}"
                )
        else:
            nodes1 = {node for edge in paths_idcs_dict[path_ion_1][1:-1] for node in edge}

        if len(paths_idcs_dict[path_ion_2]) == 2:
            # if same edge twice -> skip (no edge if twice parking edge, otherwise only first node)
            if paths_idcs_dict[path_ion_2][0] == paths_idcs_dict[path_ion_2][1]:
                # TODO
                # if get_idx_from_idc(graph.idc_dict, paths_idcs_dict[path_ion_2][0]) == get_idx_from_idc(
                #     graph.idc_dict, graph.pzgraph_creator.parking_edge
                # ):
                #     nodes2 = set()
                if (
                    graph.get_edge_data(
                        *get_idc_from_idx(
                            graph.idc_dict, get_idx_from_idc(graph.idc_dict, paths_idcs_dict[path_ion_2][0])
                        )
                    )["edge_type"]
                    == "processing"
                ):
                    nodes2 = set()
                else:
                    nodes2 = {paths_idcs_dict[path_ion_2][0][0]}
            else:
                # path - only middle node of path (two edges)
                nodes2 = {paths_idcs_dict[path_ion_2][0][1]}
                assert paths_idcs_dict[path_ion_2][0][1] == paths_idcs_dict[path_ion_2][1][0], (
                    f"not middle node? {paths_idcs_dict[path_ion_2]}"
                )
        else:
            nodes2 = {node for edge in paths_idcs_dict[path_ion_2][1:-1] for node in edge}

        # new: exclude processing zone node -> if pz node in circles -> can both be executed (TODO check again for moves out of pz)
        # extra: if both end in same edge -> don't execute (scenario where path out of pz ends in same edge as next edge for other)
        if (
            len(nodes1.intersection(nodes2).intersection(junction_nodes)) > 0
            # and graph.pzgraph_creator.processing_zone not in nodes1.intersection(nodes2)
        ):
            conflicting_paths.append((path_ion_1, path_ion_2))
    print("conflicting_paths", conflicting_paths)
    return conflicting_paths
