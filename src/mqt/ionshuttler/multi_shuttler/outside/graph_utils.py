from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from .graph import Graph
    from .types import Edge, Node


# create dictionary to swap from idx to idc and vice versa
# reversed in comparison with previous versions -> edge_idc key, edge_idx value now -> can also have an entry for the reversed edge_idc
def create_idc_dictionary(graph: Graph) -> dict[Edge, int]:
    edge_dict = {}
    for edge_idx, edge_idc in enumerate(graph.edges()):
        node1, node2 = tuple(sorted(edge_idc, key=sum))
        edge_dict[node1, node2] = edge_idx
        edge_dict[node2, node1] = edge_idx
    return edge_dict


def get_idx_from_idc(edge_dictionary: dict[Edge, int], idc: Edge) -> int:
    node1, node2 = tuple(sorted(idc, key=sum))
    return edge_dictionary[node1, node2]


def get_idc_from_idx(edge_dictionary: dict[Edge, int], idx: int) -> Edge:
    return next((k for k, v in edge_dictionary.items() if v == idx))  # list(edge_dictionary.values()).index(idx)


def create_dist_dict(graph: Graph) -> dict[str, dict[Edge, list[Node]]]:
    # create dictionary of dictionary with all distances to entry of each edge for each pz
    from .cycles import find_path_edge_to_edge

    dist_dict = {}
    for pz in graph.pzs:
        pz_dict = {}
        for edge_idc in graph.edges():
            # keep node ordering consistent:
            edge_idx = get_idx_from_idc(graph.idc_dict, edge_idc)
            # for pz_path_idx in pz.path_to_pz_idxs:
            #     if edge_idx == pz.path_to_pz:
            path = find_path_edge_to_edge(graph, edge_idc, pz.parking_edge)
            assert path is not None
            pz_dict[get_idc_from_idx(graph.idc_dict, edge_idx)] = path

        dist_dict[pz.name] = pz_dict
    return dist_dict


# calc distance to parking edge for all ions
def update_distance_map(graph: Graph, state: dict[int, int]) -> dict[int, dict[str, int]]:
    """Update a distance map that tracks the distances to each pz for each ion of current state.
    Dict: {ion: {'pz_name': distance}},
    e.g.,  {0: {'pz1': 2, 'pz2': 2, 'pz3': 1}, 1: {'pz1': 4, 'pz2': 1, 'pz3': 2}, 2: {'pz1': 3, 'pz2': 1, 'pz3': 3}}"""
    distance_map = {}
    for ion, edge_idx in state.items():
        pz_dict = {}
        for pz in graph.pzs:
            pz_dict[pz.name] = len(graph.dist_dict[pz.name][get_idc_from_idx(graph.idc_dict, edge_idx)])
        distance_map[ion] = pz_dict
    return distance_map


# Function to convert all nodes to float
def convert_nodes_to_float(graph: Graph) -> Graph:
    mapping = {node: (float(node[0]), float(node[1])) for node in graph.nodes}
    return nx.relabel_nodes(graph, mapping, copy=False)  # type: ignore [return-value]
