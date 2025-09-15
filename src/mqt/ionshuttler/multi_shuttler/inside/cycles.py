import random

import networkx as nx
from more_itertools import pairwise

from .graph_utils import get_idx_from_idc


def create_starting_config(graph, n_of_chains, seed=None):
    # Initialize ions on edges using an edge attribute
    nx.set_edge_attributes(graph, {edge: [] for edge in graph.edges}, "ions")

    if seed is not None:
        random.seed(seed)
        starting_traps = []
        traps = list(graph.edges())
        n_of_traps = len(traps)

        random_starting_traps = random.sample(range(n_of_traps), (n_of_chains))
        for trap in random_starting_traps:
            starting_traps.append(traps[trap])
    else:
        starting_traps = [
            edges for edges in graph.edges() if graph.get_edge_data(edges[0], edges[1])["edge_type"] == "trap"
        ][:n_of_chains]
    number_of_registers = len(starting_traps)

    # place ions onto traps (ion0 on starting_trap0)
    ion_chains = {}
    for ion, idc in enumerate(starting_traps):
        graph.edges[idc]["ions"] = [ion]

    return ion_chains, number_of_registers


def get_ion_chains(graph):
    ion_chains = {}
    # Iterate over all edges in the graph
    for u, v, data in graph.edges(data=True):
        try:
            chains = data["ions"]
            # make indices of edge consistent
            edge_idc = tuple(sorted((u, v), key=sum))

            if len(data["ions"]) > 2:
                msg = f"Edge ({u}, {v}) has more than two ions: {data['ions']}"
                raise ValueError(msg)
            for chain in chains:
                ion_chains[chain] = edge_idc

        except (KeyError, IndexError):
            pass

    return ion_chains


def get_edge_state(graph):
    # TODO is wrong for multiple ions on one edge (only returns one ion)
    state_dict = {}
    # Iterate over all edges in the graph
    for u, v, data in graph.edges(data=True):
        try:
            chains = data["ions"]
            # if len(data["ions"]) > 1:
            #     raise ValueError(
            #         f"Edge ({u}, {v}) has more than one ion entry: {data['ions']}"
            #     )
            # make indices of edge consistent
            edge_idc = tuple(sorted((u, v), key=sum))
            state_dict[edge_idc] = chains
            # assert chains is list type
            assert isinstance(chains, list)

        except (KeyError, IndexError):
            pass
    return state_dict


def have_common_junction_node(graph, edge1, edge2):
    # Extract nodes from the edges
    nodes_edge1 = set(edge1)
    nodes_edge2 = set(edge2)

    # Check if the edges have any common junction nodes
    common_junction_nodes = nodes_edge1.intersection(nodes_edge2).intersection(graph.junction_nodes)

    return len(common_junction_nodes) > 0


def check_if_edge_is_filled(graph, edge_idc):
    chain = graph.edges()[edge_idc]["ions"]
    if len(chain) > 1:
        # raise ValueError(f"Edge {edge_idc} has more than one ion entry: {chain}")
        print(f"{edge_idc} has more than one ion: {chain} (while check if edge filled)")
    return len(chain) > 0  # == 1


def find_path_node_to_edge(graph, node, goal_edge):
    # manipulate graph weights
    original_weight = graph[goal_edge[0]][goal_edge[1]].get("weight", 1)
    # set weight of goal edge to inf (so it can't move past the edge)
    graph[goal_edge[0]][goal_edge[1]]["weight"] = float("inf")

    # find shortest path towards both sides (nodes of goal edge)
    path0 = nx.shortest_path(graph, node, goal_edge[0], weight="weight")
    path1 = nx.shortest_path(graph, node, goal_edge[1], weight="weight")

    # restore the original weight of the edge
    graph[goal_edge[0]][goal_edge[1]]["weight"] = original_weight

    # return min path
    if len(path1) < len(path0):
        return path1
    return path0


def find_path_edge_to_edge(graph, edge_idc, goal_edge):
    # find path to goal edge from both nodes
    path0 = find_path_node_to_edge(graph, edge_idc[0], goal_edge)
    path1 = find_path_node_to_edge(graph, edge_idc[1], goal_edge)

    # return min path
    if len(path1) < len(path0):
        return path1
    return path0


def find_next_edge(graph, edge_idc, goal_edge):
    if edge_idc == goal_edge:
        return goal_edge

    for node in goal_edge:
        if node in edge_idc:
            return goal_edge
    node_path = find_path_edge_to_edge(graph, edge_idc, goal_edge)
    return (node_path[0], node_path[1])


def find_ordered_edges(graph, edge1, edge2):
    idc_dict = graph.idc_dict

    # Find the common node shared between the two edges
    common_node = set(edge1).intersection(set(edge2))

    if len(common_node) != 1 and edge1 != edge2:
        msg = f"The input edges are not connected. Edges: {edge1}, {edge2}"
        raise ValueError(msg)

    common_node = common_node.pop()
    if edge1[0] == common_node:
        edge1_in_order = (edge1[1], common_node)
        edge2_in_order = (common_node, edge2[1]) if edge2[0] == common_node else (common_node, edge2[0])
    else:
        edge1_in_order = (edge1[0], common_node)
        edge2_in_order = (common_node, edge2[1]) if edge2[0] == common_node else (common_node, edge2[0])

    # if same edge twice don't change order (for blocking moves)
    if get_idx_from_idc(idc_dict, edge1_in_order) == get_idx_from_idc(idc_dict, edge2_in_order):
        edge2_in_order = edge1_in_order

    return edge1_in_order, edge2_in_order


def create_cycle(
    graph,
    edge_idc,
    next_edge,
):
    idc_dict = graph.idc_dict

    # cycles within memory zone
    node_path = nx.shortest_path(
        graph,
        next_edge[1],
        edge_idc[0],
        lambda node0, node1, _: [
            1e8
            if (
                get_idx_from_idc(idc_dict, (node0, node1)) == get_idx_from_idc(idc_dict, edge_idc)
                or get_idx_from_idc(idc_dict, (node0, node1)) == get_idx_from_idc(idc_dict, next_edge)
            )
            else 1
        ][0],
    )
    edge_path = []
    for edge in pairwise(node_path):
        edge_path.append(edge)

    return [edge_idc, next_edge, *edge_path, edge_idc]
