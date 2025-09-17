from __future__ import annotations

import contextlib
import itertools
from itertools import pairwise
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from z3 import And, AtLeast, AtMost, Bool, Not, Or, Solver, sat

if TYPE_CHECKING:
    from collections.abc import Callable

    from .types import Edge, Graph, Node


# functions
def create_graph(
    m: int,
    n: int,
    ion_chain_size_vertical: int,
    ion_chain_size_horizontal: int,
) -> Graph:
    # m columns (vertical), n rows (horizontal)
    m_extended = m + (ion_chain_size_vertical - 1) * (m - 1)
    n_extended = n + (ion_chain_size_horizontal - 1) * (n - 1)

    networkx_graph: Graph = nx.grid_2d_graph(m_extended, n_extended)
    for node in networkx_graph.nodes():
        networkx_graph.add_node(node, node_type="trap_node", color="b")

    # remove horizontal edges (need vertical variables for that)
    for i in range(0, m_extended - ion_chain_size_vertical, ion_chain_size_vertical):
        for k in range(1, ion_chain_size_vertical):
            for j in range(n_extended - 1):
                networkx_graph.remove_edge((i + k, j), (i + k, j + 1))

    # remove vertical edges (need horizontal variables for that)
    for i in range(0, n_extended - ion_chain_size_horizontal, ion_chain_size_horizontal):
        for k in range(1, ion_chain_size_horizontal):
            for j in range(m_extended - 1):
                networkx_graph.remove_edge((j, i + k), (j + 1, i + k))

    # remove horizontal nodes
    for i in range(0, m_extended - ion_chain_size_vertical, ion_chain_size_vertical):
        for k in range(1, ion_chain_size_vertical):
            for j in range(0, n_extended - ion_chain_size_horizontal, ion_chain_size_horizontal):
                for p in range(1, ion_chain_size_horizontal):
                    networkx_graph.remove_node((i + k, j + p))

    nx.set_edge_attributes(networkx_graph, values=dict.fromkeys(networkx_graph.edges(), "trap"), name="edge_type")

    # add junction nodes
    for i in range(0, m_extended, ion_chain_size_vertical):
        for j in range(0, n_extended, ion_chain_size_horizontal):
            networkx_graph.add_node((i, j), node_type="junction_node", color="g")

    memory_entry = (m_extended - 1, 0)
    memory_exit = (m_extended - 1, n_extended - 1)

    networkx_graph.add_node((memory_exit[0] + 1, memory_exit[1] + 1), node_type="processing_zone_node", color="r")
    networkx_graph.add_edge(memory_exit, (memory_exit[0] + 1, memory_exit[1] + 1), edge_type="exit", color="k")
    networkx_graph.add_edge(
        (memory_exit[0] + 1, memory_exit[1] + 1), (memory_entry[0], memory_entry[1]), edge_type="entry", color="r"
    )

    return networkx_graph


def get_possible_moves_through_node(nx_g: Graph, idc_dict: dict[int, Edge], node: Node) -> list[tuple[int, int]]:
    connected_edges = [get_idx_from_idc(idc_dict, edge_idc) for edge_idc in nx_g.edges(node)]

    possible_moves_through_node = []
    for pairs in itertools.permutations(connected_edges, 2):
        possible_moves_through_node.append(pairs)
    return possible_moves_through_node


# plotting
def plot_state(nx_g: Graph, plot_ions: bool = True) -> None:
    idc_dict = create_idc_dictionary(nx_g)
    pos = {(x, y): (y, -x) for i, (x, y) in enumerate(list(nx_g.nodes()))}
    if plot_ions is True:
        edge_labels = nx.get_edge_attributes(nx_g, "ion_chain")
    else:
        edge_labels = {}
        for idc in nx_g.edges():
            edge_labels[idc] = f"$e_{{{get_idx_from_idc(idc_dict, idc)}}}$"
    edge_color = list(nx.get_edge_attributes(nx_g, "color").values())
    node_color = list(nx.get_node_attributes(nx_g, "color").values())
    nx.draw(
        nx_g,
        pos=pos,
        with_labels=True,
        node_size=200,
        node_color=node_color,
        width=5,
        edge_color=edge_color,
        font_size=5,
    )
    nx.draw_networkx_edge_labels(nx_g, pos, edge_labels)


# create dictionary to swap from idx to idc and vice versa
def create_idc_dictionary(nx_g: Graph) -> dict[int, Edge]:
    edge_dict = {}
    for edge_idx, edge_idc in enumerate(nx_g.edges()):
        edge_dict[edge_idx] = tuple(sorted(edge_idc, key=sum))
    return edge_dict


def get_idx_from_idc(edge_dictionary: dict[int, Edge], idc: Edge) -> int:
    node1, node2 = tuple(sorted(idc, key=sum))
    return list(edge_dictionary.values()).index((node1, node2))


def get_idc_from_idx(edge_dictionary: dict[int, Edge], idx: int) -> Edge:
    return edge_dictionary[idx]


def get_path_to_node(nx_g: Graph, src: Node, tar: Node) -> list[Edge]:
    edge_path = []
    # lambda function to give path over processing zone huge weight -> doesn't take that path if not necessary
    node_path = nx.shortest_path(
        nx_g,
        src,
        tar,
        lambda edge0, edge1, edge_attr_dict: (edge_attr_dict["edge_type"] != "trap") * 1e8 + 1,  # noqa: ARG005
    )
    # shortest path should always be the correct path in a grid -> care for changes
    for edge in pairwise(node_path):
        edge_path.append(edge)
    return edge_path


def get_junctions(
    nx_g: Graph, node: Node, other_node: Node, ion_chain_size_horizontal: int, ion_chain_size_vertical: int
) -> list[Node]:
    assert node in nx_g.nodes, "node not in graph"
    # assert nx_G.nodes[node]['node_type'] != 'junction_node', 'no support for junction nodes yet'

    # new support for solo edge between two junctions (vertical/horizontal = 1)
    if nx_g.nodes[node]["node_type"] == "junction_node" and nx_g.nodes[other_node]["node_type"] == "junction_node":
        junction = [node, other_node]
    else:
        junction = []
        # extra clause for processing zone (does not work in the same way as traps)
        if nx_g.nodes[node]["node_type"] == "processing_zone_node":
            connected_edges = list(nx_g.edges(node))
            for edge in connected_edges:
                for jct_node in edge:
                    if nx_g.nodes[jct_node]["node_type"] == "junction_node":
                        junction.append(jct_node)
        else:
            # right
            for i in range(ion_chain_size_horizontal):
                node_right = (node[0], node[1] + (i + 1))
                if node_right not in list(nx_g.nodes()):
                    break
                if nx_g.nodes[node_right]["node_type"] == "junction_node":
                    junction.append(node_right)
                    break
            # left
            for i in range(ion_chain_size_horizontal):
                node_left = (node[0], node[1] - (i + 1))
                if node_left not in list(nx_g.nodes()):
                    break
                if nx_g.nodes[node_left]["node_type"] == "junction_node":
                    junction.append(node_left)
                    break
            # up
            for j in range(ion_chain_size_vertical):
                node_up = (node[0] + (j + 1), node[1])
                if node_up not in list(nx_g.nodes()):
                    break

                if nx_g.nodes[node_up]["node_type"] == "junction_node":
                    junction.append(node_up)
                    break
            # down
            for j in range(ion_chain_size_vertical):
                node_down = (node[0] - (j + 1), node[1])
                if node_down not in list(nx_g.nodes()):
                    break

                if nx_g.nodes[node_down]["node_type"] == "junction_node":
                    junction.append(node_down)
                    break
    assert junction != [], f"no junction found for node {node} - used exit, entry or processing_zone node?"
    return junction


def get_possible_moves_over_junction(
    nx_g: Graph,
    edge: Edge,
    ion_chain_size_horizontal: int,
    ion_chain_size_vertical: int,
) -> list[Edge]:
    assert len(edge[0]) == 2, "use edge"
    node1 = edge[0]
    node2 = edge[1]
    if nx_g.nodes[node1]["node_type"] != "junction_node":
        junction_nodes = get_junctions(nx_g, node1, node2, ion_chain_size_horizontal, ion_chain_size_vertical)
    else:
        junction_nodes = get_junctions(nx_g, node2, node1, ion_chain_size_horizontal, ion_chain_size_vertical)

    possible_edges = []
    between_edges = []
    for node in junction_nodes:
        for post_jct_node in nx_g.edges(node):
            possible_edges.append(post_jct_node)

    for node in junction_nodes:  # need new loop so possible_edges is finished
        edges_between = get_path_to_node(nx_g, node1, node)
        for edge_betw in edges_between:
            if edge_betw in possible_edges:
                possible_edges.remove(edge_betw)
            node1, node2 = tuple(reversed(edge_betw))
            if (node1, node2) in possible_edges:
                possible_edges.remove((node1, node2))
            between_edges.append(edge_betw)

    return possible_edges


def create_graph_dict(
    nx_g: Graph,
    func: Callable[[Graph, Edge, int, int], list[Edge]],
    ion_chain_size_horizontal: int,
    ion_chain_size_vertical: int,
    edges: list[Edge] | str = "all",
) -> dict[Edge, list[Edge]]:
    return_dict = {}
    if edges == "all":
        edges = list(nx_g.edges())
    assert isinstance(edges, list), "edges need to be list of edges or 'all'"
    for edge in edges:
        return_dict[edge] = func(nx_g, edge, ion_chain_size_horizontal, ion_chain_size_vertical)
        node1, node2 = tuple(reversed(edge))
        return_dict[node1, node2] = func(
            nx_g,
            (node2, node1),
            ion_chain_size_horizontal,
            ion_chain_size_vertical,
        )
    return return_dict


def get_path_between_edges(nx_g: Graph, src_edge: Edge, tar_edge: Edge) -> list[Edge]:
    # care: only works if there is only one junction within path (except start or end node)
    # only edge case that would be a problem is if path is from jct to jct node -> could take wrong path if graph is quadratic
    node1 = src_edge[1] if nx.get_node_attributes(nx_g, "node_type")[src_edge[0]] == "junction_node" else src_edge[0]
    node2 = tar_edge[1] if nx.get_node_attributes(nx_g, "node_type")[tar_edge[0]] == "junction_node" else tar_edge[0]

    path_edges = get_path_to_node(nx_g, node1, node2)
    with contextlib.suppress(ValueError):
        path_edges.remove((src_edge[0], src_edge[1]))
    with contextlib.suppress(ValueError):
        path_edges.remove((src_edge[1], src_edge[0]))
    with contextlib.suppress(ValueError):
        path_edges.remove((tar_edge[0], tar_edge[1]))
    with contextlib.suppress(ValueError):
        path_edges.remove((tar_edge[1], tar_edge[0]))
    return path_edges


def get_possible_previous_edges_from_junction_move(nx_g, edge, ion_chain_size_horizontal, ion_chain_size_vertical):
    # for a junction edge (edge that is connected to a junction node)
    # get all edges that could have been the previous edge of an ion, if that ion moved over the junction
    # -> get all other junction edges of that junction
    # -> for these junction edges get all connecting edges until one reaches another junction (big edge)
    # -> if junction edge is 'entry', only take 'entry' as a possible edge
    assert len(edge[0]) == 2, "use edge"
    node1 = edge[0]
    node2 = edge[1]
    assert nx_g.nodes[node1]["node_type"] == "junction_node" or nx_g.nodes[node2]["node_type"] == "junction_node", (
        "only works for junction edges"
    )
    if nx_g.nodes[node1]["node_type"] == "junction_node":
        # find all other "big" edges around junction
        around_jct = list(nx_g.edges(node1))
        with contextlib.suppress(ValueError):
            around_jct.remove(edge)
        with contextlib.suppress(ValueError):
            around_jct.remove(tuple(reversed(edge)))

        possible_edges = []
        # find next junction in every big edge
        for first_edge in around_jct:
            for node in first_edge:
                if node != node1:
                    other_jct = get_junctions(
                        nx_g,
                        node,
                        node1,
                        ion_chain_size_horizontal,
                        ion_chain_size_vertical,
                    )
            # find all edges between junction and other junction (big edge)
            # if edge around junction is 'entry' -> don't search for edges in processing zone, just take entry (only edge one can move out of processing zone over this junction)
            if nx_g.edges[first_edge]["edge_type"] == "entry":
                big_edge = [first_edge]
            # else find all edges in big edge (path to node with node = other junction)
            else:
                big_edge = get_path_to_node(nx_g, other_jct[0], other_jct[1])
            for within_edge in big_edge:
                possible_edges.append(within_edge)

    elif nx_g.nodes[node2]["node_type"] == "junction_node":
        # find all other "big" edges around junction
        around_jct = list(nx_g.edges(node2))
        with contextlib.suppress(ValueError):
            around_jct.remove(edge)
        with contextlib.suppress(ValueError):
            around_jct.remove(tuple(reversed(edge)))

        possible_edges = []
        # find next junction in every big edge
        for first_edge in around_jct:
            for node in first_edge:
                if node != node2:
                    other_jct = get_junctions(
                        nx_g,
                        node,
                        node2,
                        ion_chain_size_horizontal,
                        ion_chain_size_vertical,
                    )
            # find all edges between junction and other junction (big edge)
            # if edge around junction is 'entry' -> don't search for edges in processing zone, just take entry (only edge one can move out of processing zone over this junction)
            if nx_g.edges[first_edge]["edge_type"] == "entry":
                big_edge = [first_edge]
            # else find all edges in big edge (path to node with node = other junction)
            else:
                big_edge = get_path_to_node(nx_g, other_jct[0], other_jct[1])
            for within_edge in big_edge:
                possible_edges.append(within_edge)

    return possible_edges


class MemorySAT:
    def __init__(
        self,
        graph: Graph,
        ion_chain_size_horizontal: int,
        ion_chain_size_vertical: int,
        ions: list[int],
        timesteps: int,
    ) -> None:
        self.graph = graph
        self.ion_chain_size_horizontal = ion_chain_size_horizontal
        self.ion_chain_size_vertical = ion_chain_size_vertical

        self.idc_dict = create_idc_dictionary(self.graph)

        # find entry and exit of graph
        for edge_idc in graph.edges():
            if nx.get_edge_attributes(graph, "edge_type")[edge_idc] == "entry":
                self.entry = edge_idc
            elif nx.get_edge_attributes(graph, "edge_type")[edge_idc] == "exit":
                self.exit = edge_idc
        self.entry_node = min(self.entry)
        self.exit_node = min(self.exit)
        assert nx.get_node_attributes(graph, "node_type")[max(self.entry)] == "processing_zone_node"
        assert nx.get_node_attributes(graph, "node_type")[max(self.exit)] == "processing_zone_node"

        self.ions = ions
        self.timesteps = timesteps
        assert (
            len([
                node
                for node in graph.nodes()
                if nx.get_node_attributes(graph, "node_type")[node] == "processing_zone_node"
            ])
            == 1
        ), "exactly one processing zone node needed -> so get_junctions() works as intended"
        # set_param(proof=True)

        # Z3
        self.s = Solver()
        # Create Z3 bool variables for self.states
        self.states = [
            [[Bool(f"state_{t}_{edge_idx}_{ion}") for ion in self.ions] for edge_idx in range(len(graph.edges()))]
            for t in range(self.timesteps)
        ]
        # time, trap, ion

    def create_constraints(self, starting_traps: list[Edge]) -> None:
        self.starting_traps = starting_traps

        junction_nodes = [
            node
            for node in self.graph.nodes()
            if nx.get_node_attributes(self.graph, "node_type")[node] == "junction_node"
        ]
        junction_edges = [list(self.graph.edges(node)) for node in junction_nodes]
        # flatten list junction edges: [[1, 3], [4, 2, 5]] -> [1, 3, 4, 2, 5]
        junction_edges_flattened = [
            (sorted(item)[0], sorted(item)[1]) for sublist in junction_edges for item in sublist
        ]
        # junction edges are now edges connected to a junction, but not special edges anymore

        # create lookup dictionary for move constraints
        junction_move_dict = {}

        junction_move_dict = create_graph_dict(
            self.graph,
            get_possible_moves_over_junction,
            self.ion_chain_size_horizontal,
            self.ion_chain_size_vertical,
        )
        previous_junction_move_dict = create_graph_dict(
            self.graph,
            get_possible_previous_edges_from_junction_move,
            self.ion_chain_size_horizontal,
            self.ion_chain_size_vertical,
            edges=junction_edges_flattened,
        )

        # starting configuration:
        for idc in self.graph.edges():
            for ion in self.ions:
                if idc not in self.starting_traps:
                    self.s.add(Not(self.states[0][get_idx_from_idc(self.idc_dict, idc)][ion]))

        for ion, idc in enumerate(self.starting_traps):
            # for simplicity fill first trap with ion1 and second with ion2 and so on:
            ion_of_starting_trap = self.ions[ion]
            self.s.add(self.states[0][get_idx_from_idc(self.idc_dict, idc)][ion_of_starting_trap])

            # also set all other self.ions in starting traps to False (starting trap holds only one ion)
            for other_ion in self.ions:
                if other_ion != ion:
                    self.s.add(Not(self.states[0][get_idx_from_idc(self.idc_dict, idc)][other_ion]))

        # move constraints
        # MV_CONSTR_1: per time, per ion exactly 1 instance can be True of all traps -> amount of ions stays constant
        for t in range(1, self.timesteps):
            for ion in self.ions:
                self.s.add(
                    AtMost(
                        *[
                            self.states[t][get_idx_from_idc(self.idc_dict, edge_idc)][ion]
                            for edge_idc in self.graph.edges()
                        ],
                        1,
                    )
                )
                self.s.add(
                    AtLeast(
                        *[
                            self.states[t][get_idx_from_idc(self.idc_dict, edge_idc)][ion]
                            for edge_idc in self.graph.edges()
                        ],
                        1,
                    )
                )  # changed

        # MV_CONSTR_2: new move constraints
        for t in range(self.timesteps - 1):
            for ion in self.ions:
                for edge_idc in self.graph.edges():
                    possible_edges = junction_move_dict[edge_idc].copy()

                    # MV_CONSTR_2 EXTRA: also add possible move within big edge (old 1 step moves logic)
                    for edge_idc_around in self.graph.edges(edge_idc):
                        possible_edges.append(edge_idc_around)

                    # trap is either True and is in possible edge at the next timestep or is False
                    # also all edges in between have to be empty at this timestep
                    self.s.add(
                        Or(
                            Not(self.states[t][get_idx_from_idc(self.idc_dict, edge_idc)][ion]),
                            And(
                                self.states[t][get_idx_from_idc(self.idc_dict, edge_idc)][ion],
                                Or(*[
                                    And(
                                        self.states[t + 1][get_idx_from_idc(self.idc_dict, poss_edge)][ion],
                                        *[
                                            Not(
                                                self.states[t][
                                                    get_idx_from_idc(
                                                        self.idc_dict,
                                                        path_to_poss_edge,
                                                    )
                                                ][other_ions]
                                            )
                                            for path_to_poss_edge in get_path_between_edges(
                                                self.graph, edge_idc, poss_edge
                                            )
                                            for other_ions in self.ions
                                        ],
                                    )
                                    for poss_edge in possible_edges
                                ]),
                            ),
                        )
                    )
        # MV_CONSTR_2 EXTRA: only one junction move allowed (junction allows only one ion per timestep)
        for t in range(1, self.timesteps):
            for node in junction_nodes:
                self.s.add(
                    AtMost(
                        *[
                            And(
                                self.states[t][get_idx_from_idc(self.idc_dict, junction_edge)][ion],
                                Or(*[
                                    self.states[t - 1][get_idx_from_idc(self.idc_dict, poss_prev_edge)][ion]
                                    for poss_prev_edge in previous_junction_move_dict[junction_edge]
                                ]),
                            )
                            for junction_edge in self.graph.edges(node)
                            for ion in self.ions
                        ],
                        1,
                    )
                )
        # MV_CONSTR_2_EXTRA_EXTRA: need also constraint from prior solution -> only one ion can move through node (not only JUNCTION nodes) per timestep
        for t in range(1, self.timesteps):
            for node in self.graph.nodes():
                self.s.add(
                    AtMost(
                        *[
                            And(
                                self.states[t][edge_moves[0]][ion],
                                self.states[t - 1][edge_moves[1]][ion],
                            )
                            for ion in self.ions
                            for edge_moves in get_possible_moves_through_node(self.graph, self.idc_dict, node)
                        ],
                        1,
                    )
                )

        # MV_CONSTR_3: can't move to occupied trap (changed: exclude processing zone -> 2 register are allowed)
        for t in range(1, self.timesteps):
            for edge_idc in self.graph.edges():
                node1, node2 = tuple(sorted(edge_idc))
                if nx.get_edge_attributes(self.graph, "edge_type")[node1, node2] == "entry":
                    self.s.add(
                        AtMost(
                            *[self.states[t][get_idx_from_idc(self.idc_dict, edge_idc)][ion] for ion in self.ions], 2
                        )
                    )
                else:
                    self.s.add(
                        AtMost(
                            *[self.states[t][get_idx_from_idc(self.idc_dict, edge_idc)][ion] for ion in self.ions], 1
                        )
                    )

        # New exit constraints
        # if in exit -> was in connected edge in graph at t-1 and has to move to entry at t+1
        for t in range(1, self.timesteps - 1):
            for ion in self.ions:
                self.s.add(
                    Or(
                        Not(self.states[t][get_idx_from_idc(self.idc_dict, self.exit)][ion]),
                        And(
                            self.states[t + 1][get_idx_from_idc(self.idc_dict, self.entry)][ion],
                            self.states[t][get_idx_from_idc(self.idc_dict, self.exit)][ion],
                            Or(*[
                                self.states[t - 1][get_idx_from_idc(self.idc_dict, connected_to_exit)][ion]
                                for connected_to_exit in self.graph.edges(self.exit_node)
                            ]),
                        ),
                    )
                )

        # exit constraints
        # if in entry -> was in exit at t-1 (or in entry -> stayed longer in entry)
        for t in range(1, self.timesteps):
            for ion in self.ions:
                self.s.add(
                    Or(
                        Not(self.states[t][get_idx_from_idc(self.idc_dict, self.entry)][ion]),
                        And(
                            self.states[t][get_idx_from_idc(self.idc_dict, self.entry)][ion],
                            Or(
                                self.states[t - 1][get_idx_from_idc(self.idc_dict, self.exit)][ion],
                                self.states[t - 1][get_idx_from_idc(self.idc_dict, self.entry)][ion],
                            ),
                        ),
                    )
                )

        # if in entry -> has to move one past its own junction or stay in entry (otherwise could jump back through processing zone over other junction)
        for t in range(self.timesteps - 1):
            for ion in self.ions:
                self.s.add(
                    Or(
                        Not(self.states[t][get_idx_from_idc(self.idc_dict, self.entry)][ion]),
                        And(
                            self.states[t][get_idx_from_idc(self.idc_dict, self.entry)][ion],
                            Or(
                                self.states[t + 1][get_idx_from_idc(self.idc_dict, self.entry)][ion],
                                *[
                                    self.states[t + 1][get_idx_from_idc(self.idc_dict, connected_to_entry)][ion]
                                    for connected_to_entry in self.graph.edges(self.entry_node)
                                ],
                            ),
                        ),
                    )
                )

        # can't be in exit or entry in the last time step
        for ion in self.ions:
            self.s.add(Not(self.states[-1][get_idx_from_idc(self.idc_dict, self.exit)][ion]))
            self.s.add(Not(self.states[-1][get_idx_from_idc(self.idc_dict, self.entry)][ion]))

    def evaluate(self, sequence: list[int | tuple[int, ...]], num_of_registers: int) -> bool:
        # check that maximum ion index in sequence is also in graph
        # flatten (remove tuples)
        flat_sequence = []
        for sublist in sequence:
            if isinstance(sublist, tuple):
                for item in sublist:
                    flat_sequence.append(item)
            elif isinstance(sublist, int):
                flat_sequence.append(sublist)

        assert num_of_registers > max(flat_sequence), (
            f"numb of registers: {num_of_registers}, max flat sequence: {max(flat_sequence)}"
        )
        assert len(sequence) > 1

        for elem in sequence:
            assert isinstance(elem, tuple | int), f"Element {elem} is not a tuple or int"

        # create sequence states:
        self.seq_index = [
            [Bool(f"seq_tuple_{t}_{tuples}") for tuples in range(len(sequence))] for t in range(self.timesteps)
        ]

        # initialize sequence states -> if tuple -> both tuple ions in entry
        for t in range(self.timesteps):
            for i, tpl in enumerate(sequence):
                if isinstance(tpl, tuple):
                    self.s.add(
                        Or(
                            Not(self.seq_index[t][i]),
                            And(
                                self.seq_index[t][i],
                                self.states[t][get_idx_from_idc(self.idc_dict, self.entry)][tpl[0]],
                                self.states[t][get_idx_from_idc(self.idc_dict, self.entry)][tpl[1]],
                            ),
                        )
                    )
                elif isinstance(tpl, int):
                    self.s.add(
                        Or(
                            Not(self.seq_index[t][i]),
                            And(
                                self.seq_index[t][i],
                                self.states[t][get_idx_from_idc(self.idc_dict, self.entry)][tpl],
                                *[
                                    Not(self.states[t][get_idx_from_idc(self.idc_dict, self.entry)][other_ion])
                                    for other_ion in self.ions
                                    if other_ion != tpl
                                ],
                            ),
                        )
                    )

        # actual sequence constraint
        for i, j in pairwise(range(len(sequence))):
            # considered_ions: next two gates
            considered_ions = []
            considered_ions.append(sequence[i])
            considered_ions.append(sequence[j])

            # flat_cons_ions: flatten list of considered_ions (needed for other_ions below)
            flat_cons_ions = []
            for sublist in considered_ions:
                if isinstance(sublist, tuple):
                    for item in sublist:
                        flat_cons_ions.append(item)
                elif isinstance(sublist, int):
                    flat_cons_ions.append(sublist)

            # other_ions: all ions that are not used in these two gates
            other_ions = self.ions.copy()
            for cons_ion in flat_cons_ions:
                if cons_ion in other_ions:
                    other_ions.remove(cons_ion)

            # sequence constraint: ion(s) of gate1 are in PZ at t -> ion(s) of gate2 are in PZ at some t' later + no other ions in PZ in time between the two gates
            self.s.add(
                Or(*[
                    And(
                        self.seq_index[t][i],
                        Or(*[
                            And(
                                *[
                                    Not(self.states[t_inter][get_idx_from_idc(self.idc_dict, self.entry)][other_ion])
                                    for other_ion in other_ions
                                    for t_inter in range(t, t_next)
                                ],
                                self.seq_index[t_next][j],
                            )
                            for t_next in range(t + 1, self.timesteps)
                        ]),
                    )
                    for t in range(self.timesteps)
                ])
            )

        # every sequence ion is True once
        for seq_ion in range(len(sequence)):
            self.s.add(AtMost(*[self.seq_index[t][seq_ion] for t in range(1, self.timesteps)], 1))
        # only one sequence ion in entry per timestep
        for t in range(1, self.timesteps):
            self.s.add(AtMost(*[self.seq_index[t][seq_ion] for seq_ion in range(len(sequence))], 1))

        self.check = self.s.check()
        if self.check == sat:
            self.model = self.s.model()
        # elif self.check == unsat:
        #    print(self.s.proof())
        print(self.check)

        return bool(self.check == sat)

    def plot(self, show_ions: bool = False) -> None:
        if self.check == sat:
            for t in range(self.timesteps):
                ion_trap = []
                for edge_idc in self.graph.edges():
                    # color all edges black
                    self.graph.add_edge(edge_idc[0], edge_idc[1], color="k")

                    ion_holder = []
                    colors = []
                    np.random.seed(0)
                    for _i in range(len(self.ions)):
                        r = np.round(np.random.rand(), 1)
                        g = np.round(np.random.rand(), 1)
                        b = np.round(np.random.rand(), 1)

                        colors.append((r, g, b))
                    np.random.seed()

                    for i, ion in enumerate(self.ions):
                        if (
                            bool(self.model.evaluate(self.states[t][get_idx_from_idc(self.idc_dict, edge_idc)][ion]))
                            is True
                        ):
                            ion_trap.append(edge_idc)
                            ion_holder.append(ion)
                            self.graph.add_edge(
                                edge_idc[0],
                                edge_idc[1],
                                ion_chain=ion_holder,
                                color=colors[i],
                            )

                        else:
                            # update ion holder (those who were True at t-1 and are False now)
                            self.graph.add_edge(edge_idc[0], edge_idc[1], ion_chain=ion_holder)

                plt.subplot(1, self.timesteps, t + 1)
                plot_state(self.graph, plot_ions=show_ions)
            plt.show()
        else:
            plt.subplot(1, 1, 1)
            plt.show()
