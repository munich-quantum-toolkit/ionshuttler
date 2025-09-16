from __future__ import annotations

import random
from typing import TYPE_CHECKING

import networkx as nx
from more_itertools import distinct_combinations, pairwise

from .graph_utils import get_idx_from_idc

if TYPE_CHECKING:
    from .graph import Graph
    from .processing_zone import ProcessingZone
    from .types import Edge, Node


def get_ions_in_pz_and_connections(graph: Graph, pz: ProcessingZone) -> int:
    return len([
        ion_idx
        for ion_idx, edge_idc in get_ions(graph).items()
        if get_idx_from_idc(graph.idc_dict, edge_idc) in pz.pz_edges_idx
    ])


def get_ions_in_exit_connections(graph: Graph, pz: ProcessingZone) -> int:
    return len([
        ion_idx
        for ion_idx, edge_idc in get_ions(graph).items()
        if get_idx_from_idc(graph.idc_dict, edge_idc) in pz.path_to_pz_idxs
    ])


def get_ions_in_parking(graph: Graph, pz: ProcessingZone) -> int:
    return len([
        ion_idx
        for ion_idx, edge_idc in get_ions(graph).items()
        if get_idx_from_idc(graph.idc_dict, edge_idc) == get_idx_from_idc(graph.idc_dict, pz.parking_edge)
    ])


def find_ion_in_edge(graph: Graph, edge_idc: Edge) -> int | None:
    ions = [
        ion
        for ion, ion_edge_idc in get_ions(graph).items()
        if get_idx_from_idc(graph.idc_dict, edge_idc) == get_idx_from_idc(graph.idc_dict, ion_edge_idc)
    ]
    assert len(ions) <= 1, f"more than one ion ({ions}) in edge {edge_idc}"
    if len(ions) == 0:
        return None
    return ions[0]


def find_ions_in_parking(graph: Graph, pz: ProcessingZone) -> list[int]:
    return [
        ion
        for ion, ion_edge_idc in get_ions(graph).items()
        if get_idx_from_idc(graph.idc_dict, ion_edge_idc) == get_idx_from_idc(graph.idc_dict, pz.parking_edge)
    ]


# get dict of edge idxs of ions
def get_state_idxs(graph: Graph) -> dict[int, int]:
    ions_edge_idxs = {}
    for ion, edge_idc in get_ions(graph).items():
        ions_edge_idxs[ion] = get_idx_from_idc(graph.idc_dict, edge_idc)
    return ions_edge_idxs


def find_least_import_ion_in_parking(seq: list[int], ions_in_parking: list[int]) -> int:
    for num in seq:
        if num in ions_in_parking:
            ions_in_parking.remove(num)
            if len(ions_in_parking) == 1:
                return ions_in_parking[0]
    return ions_in_parking[-1]


def create_starting_config(graph: Graph, n_of_ions: int, seed: int | None = None) -> int:
    # Initialize ions on edges using an edge attribute
    nx.set_edge_attributes(graph, {key: [] for key in graph.edges()}, "ions")

    if seed is not None:
        random.seed(seed)
        starting_traps = []
        traps = [edges for edges in graph.edges() if graph.get_edge_data(edges[0], edges[1])["edge_type"] == "trap"]
        n_of_traps = len(traps)

        random_starting_traps = random.sample(range(n_of_traps), (n_of_ions))
        for trap in random_starting_traps:
            starting_traps.append(traps[trap])
    else:
        starting_traps = [
            edges for edges in graph.edges() if graph.get_edge_data(edges[0], edges[1])["edge_type"] == "trap"
        ][:n_of_ions]
    number_of_registers = len(starting_traps)

    # place ions onto traps (ion0 on starting_trap0)
    for ion, idc in enumerate(starting_traps):
        graph.edges[idc]["ions"] = [ion]

    return number_of_registers


def get_ions(graph: Graph) -> dict[int, Edge]:
    ions = {}
    # Iterate over all edges in the graph
    for u, v, data in graph.edges(data=True):
        try:
            ions_on_edge = data["ions"]
            # make indices of edge consistent
            node1, node2 = tuple(sorted((u, v), key=sum))

            for ion in ions_on_edge:
                ions[ion] = (node1, node2)

        except (KeyError, IndexError):
            pass

    return ions


def get_edge_state(graph: Graph) -> dict[Edge, list[int]]:
    state_dict = {}
    # Iterate over all edges in the graph
    for u, v, data in graph.edges(data=True):
        try:
            ions = data["ions"]
            # make indices of edge consistent
            node1, node2 = tuple(sorted((u, v), key=sum))
            state_dict[node1, node2] = ions
            # assert ions is list type
            assert isinstance(ions, list)

        except (KeyError, IndexError):
            pass
    return state_dict


def have_common_junction_node(graph: Graph, edge1: Edge, edge2: Edge) -> bool:
    # Extract nodes from the edges
    nodes_edge1 = set(edge1)
    nodes_edge2 = set(edge2)
    # Check if the edges have any common junction nodes
    common_junction_nodes = nodes_edge1.intersection(nodes_edge2).intersection(graph.junction_nodes)

    return len(common_junction_nodes) > 0


def check_if_edge_is_filled(graph: Graph, edge_idc: Edge) -> bool:
    ion = graph.edges()[edge_idc]["ions"]
    return len(ion) > 0  # == 1


def shortest_path_to_node(
    nx_g: Graph,
    src: Node,
    tar: Node,
    exclude_exit: bool = False,
    exclude_first_entry_connection: bool = True,
) -> list[Node]:
    if exclude_first_entry_connection:
        if exclude_exit:
            return nx.shortest_path(
                nx_g,
                src,
                tar,
                lambda _, __, edge_attr_dict: (edge_attr_dict["edge_type"] in {"first_entry_connection", "exit"}) * 1e8
                + 1,
            )
        return nx.shortest_path(
            nx_g,
            src,
            tar,
            lambda _, __, edge_attr_dict: (edge_attr_dict["edge_type"] == "first_entry_connection") * 1e8 + 1,
        )

    if exclude_exit:
        return nx.shortest_path(
            nx_g,
            src,
            tar,
            lambda _, __, edge_attr_dict: (edge_attr_dict["edge_type"] == "exit") * 1e8 + 1,
        )

    return nx.shortest_path(nx_g, src, tar)


def find_path_node_to_edge(
    graph: Graph,
    node: Node,
    goal_edge: Edge,
    exclude_exit: bool = False,
    exclude_first_entry_connection: bool = True,
) -> list[Node] | None:
    if goal_edge[0] in graph.nodes() and goal_edge[1] in graph.nodes():
        original_weight = graph[goal_edge[0]][goal_edge[1]].get("weight", 1)
        # set weight of goal edge to inf (so it can't move past the edge)
        graph[goal_edge[0]][goal_edge[1]]["weight"] = float("inf")

    # find shortest path towards both sides (nodes of goal edge)
    try:
        path0 = shortest_path_to_node(
            graph,
            node,
            goal_edge[0],
            exclude_exit=exclude_exit,
            exclude_first_entry_connection=exclude_first_entry_connection,
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        path0 = None
    try:
        path1 = shortest_path_to_node(
            graph,
            node,
            goal_edge[1],
            exclude_exit=exclude_exit,
            exclude_first_entry_connection=exclude_first_entry_connection,
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        path1 = None

    # now still works if one node of target edge is not present in graph
    if path0 is None:
        if path1 is not None:
            return path1
        return None
    if path1 is None and path0 is not None:
        return path0

    # restore the original weight of the edge
    graph[goal_edge[0]][goal_edge[1]]["weight"] = original_weight

    # return min path
    if len(path1) < len(path0):
        return path1
    return path0


def find_path_edge_to_edge(
    graph: Graph,
    edge_idc: Edge,
    goal_edge: Edge,
    exclude_exit: bool = False,
    exclude_first_entry_connection: bool = True,
) -> list[Node] | None:
    # Find path from edge_idc to goal_edge
    # does not include the target edge itself

    # if in entry and entry==first_entry_connection -> can't use exclude_first_entry_connection
    if graph.get_edge_data(edge_idc[0], edge_idc[1])["edge_type"] == "first_entry_connection":
        if graph.nodes(data=True)[edge_idc[0]]["node_type"] == "processing_zone_node":
            return find_path_node_to_edge(
                graph, edge_idc[1], goal_edge, exclude_exit=False, exclude_first_entry_connection=True
            )
        if graph.nodes(data=True)[edge_idc[1]]["node_type"] == "processing_zone_node":
            return find_path_node_to_edge(
                graph, edge_idc[0], goal_edge, exclude_exit=False, exclude_first_entry_connection=True
            )
        msg = "Edge is not an entry edge"
        raise ValueError(msg)

    # find path to goal edge from both nodes
    path0 = find_path_node_to_edge(
        graph,
        edge_idc[0],
        goal_edge,
        exclude_exit=exclude_exit,
        exclude_first_entry_connection=exclude_first_entry_connection,
    )
    assert path0 is not None

    path1 = find_path_node_to_edge(
        graph,
        edge_idc[1],
        goal_edge,
        exclude_exit=exclude_exit,
        exclude_first_entry_connection=exclude_first_entry_connection,
    )
    assert path1 is not None

    # return min path
    if len(path1) < len(path0):
        return path1
    return path0


def find_next_edge(
    graph: Graph,
    edge_idc: Edge,
    goal_edge: Edge,
    exclude_exit: bool = False,
    exclude_first_entry_connection: bool = True,
) -> Edge:
    if get_idx_from_idc(graph.idc_dict, edge_idc) == get_idx_from_idc(graph.idc_dict, goal_edge):
        return goal_edge

    # if in pz next edge is first_entry_connection (needed for rare case that ion is in pz but has to move to other pz for 2-qubit gate)
    if graph.get_edge_data(edge_idc[0], edge_idc[1])["edge_type"] == "parking_edge":
        for node in edge_idc:
            if nx.get_node_attributes(graph, "node_type")[node] == "processing_zone_node":
                return next(
                    edge
                    for edge in graph.edges(node)
                    if graph.get_edge_data(edge[0], edge[1])["edge_type"] == "first_entry_connection"
                )

    # if goal edge is next edge, return goal edge (now also entry is excluded)
    if (
        graph.get_edge_data(edge_idc[0], edge_idc[1])["edge_type"] != "first_entry_connection"
        and graph.get_edge_data(edge_idc[0], edge_idc[1])["edge_type"] != "entry"
    ):
        for node in goal_edge:
            if node in edge_idc:
                return goal_edge

    node_path = find_path_edge_to_edge(
        graph,
        edge_idc,
        goal_edge,
        exclude_exit=exclude_exit,
        exclude_first_entry_connection=exclude_first_entry_connection,
    )
    assert node_path is not None

    return (node_path[0], node_path[1])


def find_ordered_edges(graph: Graph, edge1: Edge, edge2: Edge) -> tuple[Edge, Edge]:
    idc_dict = graph.idc_dict

    # Find the common node shared between the two edges
    common_nodes = set(edge1).intersection(set(edge2))

    if len(common_nodes) != 1 and get_idx_from_idc(idc_dict, edge1) != get_idx_from_idc(idc_dict, edge2):
        msg = f"The input edges are not connected. Edges: {edge1}, {edge2}"
        raise ValueError(msg)

    common_node = common_nodes.pop()
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


def create_cycle(graph: Graph, edge_idc: Edge, next_edge: Edge) -> list[Edge]:
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
                or graph.get_edge_data(node0, node1)["edge_type"] == "entry"
                or graph.get_edge_data(node0, node1)["edge_type"] == "first_entry_connection"
            )
            else 1
        ][0],
    )
    edge_path = []
    for edge in pairwise(node_path):
        edge_path.append(edge)

    return [edge_idc, next_edge, *edge_path, edge_idc]


def find_conflict_cycle_idxs(graph: Graph, cycles_dict: dict[int, list[Edge]]) -> list[tuple[int, int]]:
    combinations_of_cycles = list(distinct_combinations(cycles_dict.keys(), 2))

    def get_cycle_nodes(cycle: int, graph: Graph) -> set[Node]:
        # if next edge is free -> cycle is just two edges -> can skip first and last node
        if len(cycles_dict[cycle]) == 2:
            if cycles_dict[cycle][0] != cycles_dict[cycle][1]:
                cycle_or_path = [(cycles_dict[cycle][0][1], cycles_dict[cycle][1][0])]
                assert cycles_dict[cycle][0][1] == cycles_dict[cycle][1][0], (
                    f"cycle is not two edges? Middle node should be the same ({cycles_dict[cycle]})"
                )
                if nx.get_node_attributes(graph, "node_type")[cycles_dict[cycle][0][1]] in {
                    "exit_node",
                    "exit_connection_node",
                }:
                    cycle_or_path = []

            # new if same edge twice is parking edge -> skip completely
            elif get_idx_from_idc(graph.idc_dict, cycles_dict[cycle][0]) in graph.parking_edges_idxs:
                cycle_or_path = []
            else:  # else if path is same edge twice skip (but of course keep first node -> no movement into this edge)
                cycle_or_path = [(cycles_dict[cycle][0][0], cycles_dict[cycle][0][0])]
        # if cycle is real cycle -> need to check all nodes
        elif cycles_dict[cycle][0] == cycles_dict[cycle][-1]:
            cycle_or_path = cycles_dict[cycle]

        # if cycle is only a path -> can skip first and last node
        # extra clause for when there is a stop at the end? -> if last two edges are same -> skip last two edges?
        elif cycles_dict[cycle][-1] == cycles_dict[cycle][-2]:
            cycle_or_path = cycles_dict[cycle][1:-2]
        else:
            cycle_or_path = cycles_dict[cycle][1:-1]
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
        nodes1 = get_cycle_nodes(cycle1, graph)
        nodes2 = get_cycle_nodes(cycle2, graph)

        # new: exclude processing zone node -> if pz node in circles -> can both be executed (TODO check again for moves out of pz)
        # extra: if both end in same edge -> don't execute (scenario where path out of pz ends in same edge as next edge for other)
        # -> new exclude parking edge (can end both in parking edge, since stop moves in parking edge also end in parking edge)
        if len(nodes1.intersection(nodes2)) > 0 or (  # noqa: SIM102
            get_idx_from_idc(graph.idc_dict, cycles_dict[cycle1][-1])
            == (get_idx_from_idc(graph.idc_dict, cycles_dict[cycle2][-1]))
            and (get_idx_from_idc(graph.idc_dict, cycles_dict[cycle1][-1]) not in graph.parking_edges_idxs)
        ):
            # also exclude cases in which a stop move would block a junction (should only block moves into stop move, but not the whole junction)
            # -> do not append if cycle1 is stop move and cycle2 does not move into stop move and vice versa
            if not (
                (
                    get_idx_from_idc(graph.idc_dict, cycles_dict[cycle1][0])
                    == get_idx_from_idc(graph.idc_dict, cycles_dict[cycle1][1])
                    and (
                        get_idx_from_idc(graph.idc_dict, cycles_dict[cycle2][1])
                        != get_idx_from_idc(graph.idc_dict, cycles_dict[cycle1][0])
                    )
                )
                or (
                    get_idx_from_idc(graph.idc_dict, cycles_dict[cycle2][0])
                    == get_idx_from_idc(graph.idc_dict, cycles_dict[cycle2][1])
                    and (
                        get_idx_from_idc(graph.idc_dict, cycles_dict[cycle1][1])
                        != get_idx_from_idc(graph.idc_dict, cycles_dict[cycle2][0])
                    )
                )
            ):
                junction_shared_pairs.append((cycle1, cycle2))

    return junction_shared_pairs
