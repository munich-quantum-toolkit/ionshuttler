from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from more_itertools import distinct_combinations, pairwise

from .graph_utils import (
    GraphCreator,
    MZGraphCreator,
    calc_dist_to_pz,
    get_idc_from_idx,
    get_idx_from_idc,
    get_path_to_node,
    order_edges,
)

if TYPE_CHECKING:
    from .types import Edge, Node


class MemoryZone:
    def __init__(
        self,
        m: int,
        n: int,
        v: int,
        h: int,
        starting_config: dict[int, Edge],
        max_timestep: int,
        max_num_parking: int,
        pz: str,
        time_2qubit_gate: int = 2,
        time_1qubit_gate: int = 1,
    ) -> None:
        # new graph MZ
        self.mz_Graph_creator = MZGraphCreator(m, n, v, h, pz)
        self.mz_graph = self.mz_Graph_creator.get_graph()

        self.graph_creator = GraphCreator(m, n, v, h, pz)
        self.graph = self.graph_creator.get_graph()
        self.starting_config = starting_config
        self.max_timestep = max_timestep
        self.max_num_parking = max_num_parking
        self.time_2qubit_gate = time_2qubit_gate
        self.time_1qubit_gate = time_1qubit_gate
        self.num_ion_chains = len(starting_config)
        self.idc_dict = self.graph_creator.idc_dict

        # create dictionary with all distances to entry
        self.dist_dict: dict[Edge, int] = {}
        for edge_idc in self.graph.edges():
            # keep node ordering consistent:
            edge_idx = get_idx_from_idc(self.idc_dict, edge_idc)
            self.dist_dict[get_idc_from_idx(self.idc_dict, edge_idx)] = calc_dist_to_pz(
                self.graph_creator, get_idx_from_idc(self.idc_dict, edge_idc)
            )

        # create dictionary with all distances to entry for all nodes
        self.dist_dict_nodes = {}
        for node in self.graph.nodes():
            self.dist_dict_nodes[node] = len(get_path_to_node(self.graph, node, self.graph_creator.processing_zone))

        # # create dictionary with all paths to entry (TODO artifact?)
        # self.path_dict = {}
        # for edge_idc in self.graph.edges():
        #     self.path_dict[edge_idc] = calc_dist_to_pz(self.graph_creator, get_idx_from_idc(self.idc_dict, edge_idc))

        self.ion_chains = self.starting_config.copy()

        self.junction_nodes = [
            node
            for node in self.graph.nodes()
            if (
                nx.get_node_attributes(self.graph, "node_type")[node]
                in {"junction_node", "exit_node", "exit_connection_node", "entry_node", "entry_connection_node"}
            )
        ]

        self.path_entry_to_exit = get_path_to_node(
            self.graph, self.graph_creator.entry, self.graph_creator.exit, exclude_first_entry_connection=True
        )

        # precalulculate bfs for top left and exit
        # self.bfs_top_left = nx.edge_bfs(self.mz_graph, (0, 0))
        # self.bfs_exit = nx.edge_bfs(self.mz_graph, self.graph_creator.exit)

        def _get_edge_weight(self: MemoryZone, node0: Node, node1: Node) -> int:
            if (
                get_idx_from_idc(self.idc_dict, (node0, node1)) == get_idx_from_idc(self.idc_dict, edge_idc)
                or get_idx_from_idc(self.idc_dict, (node0, node1)) == get_idx_from_idc(self.idc_dict, next_edge)
                or get_idx_from_idc(self.idc_dict, (node0, node1))
                == get_idx_from_idc(self.idc_dict, self.graph_creator.entry_edge)
            ):
                return int(1e8)

            return 1

        # precalc all cycles (shortest paths from outer node to outer node)
        self.node_path_dict = {}
        for edge_idc, next_edge in GraphCreator(m, n, v, h, pz).find_connected_edges():
            self.node_path_dict[edge_idc, next_edge] = nx.shortest_path(
                self.graph, next_edge[1], edge_idc[0], lambda node0, node1, _: _get_edge_weight(self, node0, node1)
            )

    # get edge idxs of ion chains
    def get_state_idxs(self) -> list[int]:
        ion_chains_idx = []
        for chain in self.ion_chains.values():
            ion_chains_idx.append(get_idx_from_idc(self.idc_dict, chain))
        self.state_idxs = ion_chains_idx
        return ion_chains_idx

    # calc distance to parking edge for all ion chains
    def update_distance_map(self) -> dict[int, int]:
        self.distance_map = {}
        for ion_chain, edge_idx in enumerate(self.get_state_idxs()):
            self.distance_map[ion_chain] = self.dist_dict[get_idc_from_idx(self.idc_dict, edge_idx)]
        return self.distance_map

    def count_chains_in_pz(self) -> int:
        return len([chain_idx for chain_idx in self.get_state_idxs() if chain_idx in self.graph_creator.pz_edges_idx])

    def count_chains_in_exit(self) -> int:
        return len([
            chain_idx for chain_idx in self.get_state_idxs() if chain_idx in self.graph_creator.path_to_pz_idxs
        ])

    def count_chains_in_parking(self) -> int:
        return len([
            chain_idx
            for chain_idx in self.get_state_idxs()
            if chain_idx == get_idx_from_idc(self.idc_dict, self.graph_creator.parking_edge)
        ])

    def find_chain_in_edge(self, edge_idc: Edge) -> int | None:
        chains = [
            ion
            for ion, chain_idx in enumerate(self.get_state_idxs())
            if chain_idx == get_idx_from_idc(self.idc_dict, edge_idc)
        ]
        assert len(chains) <= 1, (
            f"more than one chain ({chains}) in edge {edge_idc}, if parking edge -> use find_chains_in_parking()"
        )
        if len(chains) == 0:
            return None
        return chains[0]

    def find_chains_in_parking(self) -> list[int]:
        return [
            ion
            for ion, chain_idx in enumerate(self.get_state_idxs())
            if chain_idx == get_idx_from_idc(self.idc_dict, self.graph_creator.parking_edge)
        ]

    def find_next_edge(self, edge_idc: Edge, towards: Node | str = (0, 0)) -> Edge:
        # find next edge given edge_idc of ion chain

        # if in entry connection -> move to next edge
        for i, edge_idx in enumerate(self.graph_creator.path_from_pz_idxs[:-1]):
            if get_idx_from_idc(self.idc_dict, edge_idc) == edge_idx:
                return get_idc_from_idx(self.idc_dict, self.graph_creator.path_from_pz_idxs[i + 1])

        if get_idx_from_idc(self.idc_dict, edge_idc) == get_idx_from_idc(self.idc_dict, self.graph_creator.entry_edge):
            if towards == (0, 0):
                next_edge = next(
                    edge
                    for edge in self.graph.edges(self.graph_creator.entry)
                    if edge not in {self.graph_creator.entry_edge, self.path_entry_to_exit[0]}
                )
            elif towards == "exit":
                next_edge = self.path_entry_to_exit[0]
            else:
                msg = "towards must be (0, 0) or 'exit'"
                raise ValueError(msg)

            # assert that next edge after entry is not entry or exit
            assert get_idx_from_idc(self.idc_dict, next_edge) != get_idx_from_idc(
                self.idc_dict, self.graph_creator.exit_edge
            )
            assert get_idx_from_idc(self.idc_dict, next_edge) != get_idx_from_idc(
                self.idc_dict, self.graph_creator.entry_edge
            )
            return next_edge

        # if in parking space
        if get_idx_from_idc(self.idc_dict, edge_idc) == get_idx_from_idc(
            self.idc_dict, self.graph_creator.parking_edge
        ):
            return self.graph_creator.parking_edge

        # find shortest path from both sides for all other edges
        path0 = nx.shortest_path(
            self.graph,
            edge_idc[0],
            self.graph_creator.parking_node,
            lambda _, __, edge_attr_dict: (edge_attr_dict["edge_type"] == "first_entry_connection") * 1e8 + 1,
        )
        path1 = nx.shortest_path(
            self.graph,
            edge_idc[1],
            self.graph_creator.parking_node,
            lambda _, __, edge_attr_dict: (edge_attr_dict["edge_type"] == "first_entry_connection") * 1e8 + 1,
        )

        # create chains in correct order -> chains are path from outer node to other outer node
        # start with outer node that is farther away from processing zone (longer path)
        # end with closer outer node (shorter path)
        if len(path1) < len(path0):
            next_edge = (path1[0], path1[1])
            # make next_edge_idc consistent
            next_edge_idx = get_idx_from_idc(self.idc_dict, next_edge)
            return get_idc_from_idx(self.idc_dict, next_edge_idx)

        next_edge = (path0[0], path0[1])
        # make next_edge_idc consistent
        next_edge_idx = get_idx_from_idc(self.idc_dict, next_edge)
        return get_idc_from_idx(self.idc_dict, next_edge_idx)

    def find_ordered_edges(self, edge1: Edge, edge2: Edge) -> tuple[Edge, Edge]:
        edge1_in_order, edge2_in_order = order_edges(edge1, edge2)

        # new if same edge twice don't change order
        if get_idx_from_idc(self.idc_dict, edge1_in_order) == get_idx_from_idc(self.idc_dict, edge2_in_order):
            edge2_in_order = edge1_in_order

        return edge1_in_order, edge2_in_order

    def have_common_junction_node(self, edge1: Edge, edge2: Edge) -> bool:
        # Extract nodes from the edges
        nodes_edge1 = set(edge1)
        nodes_edge2 = set(edge2)
        # TODO can change to self.junction_nodes?
        all_junctions = [
            *self.junction_nodes,
            self.graph_creator.processing_zone,
            self.graph_creator.entry,
            self.graph_creator.exit,
        ]

        # Check if the edges have any common junction nodes
        common_junction_nodes = nodes_edge1.intersection(nodes_edge2).intersection(all_junctions)

        return len(common_junction_nodes) == 1

    def create_outer_circle(
        self,
        edge_idc: Edge,
        next_edge: Edge,
        other_next_edges: list[Edge],
        towards: Node = (0, 0),
    ) -> list[Edge]:
        if towards == (0, 0):
            # towards is first edge in graph (can't be (0, 0) because it may be deleted)
            towards = next(iter(self.graph.edges()))[0]

        # move from entry to memory zone
        if get_idx_from_idc(self.idc_dict, edge_idc) == get_idx_from_idc(
            self.idc_dict, self.graph_creator.entry_edge
        ):  # in self.graph_creator.path_from_pz_idxs:
            target_edge = self.bfs_free_edge(towards, other_next_edges)
            # calc path to target edge
            path0 = get_path_to_node(
                self.graph,
                self.graph_creator.processing_zone,
                target_edge[0],
                exclude_exit=True,
                exclude_first_entry_connection=False,
            )
            path1 = get_path_to_node(
                self.graph,
                self.graph_creator.processing_zone,
                target_edge[1],
                exclude_exit=True,
                exclude_first_entry_connection=False,
            )
            if len(path1) > len(path0):
                edge_path = [*path0, (target_edge[0], target_edge[1])]
            else:
                edge_path = [*path1, (target_edge[1], target_edge[0])]
            return edge_path

        # circles within memory zone
        node_path = self.node_path_dict[edge_idc, next_edge]
        # node_path = nx.shortest_path(
        #     self.graph,
        #     next_edge[1],
        #     edge_idc[0],
        #     lambda node0, node1, _: [
        #         1e8
        #         if (
        #             get_idx_from_idc(self.idc_dict, (node0, node1)) == get_idx_from_idc(self.idc_dict, edge_idc)
        #             or get_idx_from_idc(self.idc_dict, (node0, node1)) == get_idx_from_idc(self.idc_dict, next_edge)
        #             or get_idx_from_idc(self.idc_dict, (node0, node1))
        #             == get_idx_from_idc(self.idc_dict, self.graph_creator.entry_edge)
        #         )
        #         else 1
        #     ][0],
        # )
        edge_path = []
        for edge in pairwise(node_path):
            edge_path.append(edge)

        return [edge_idc, next_edge, *edge_path, edge_idc]

    def check_if_edge_is_filled(self, edge_idc: Edge) -> bool:
        return get_idx_from_idc(self.idc_dict, edge_idc) in self.get_state_idxs()

    def find_nonfree_and_free_circle_idxs(self, circles_dict: dict[int, list[Edge]]) -> list[Node]:
        # careful! If only two edges -> second edge must always be free! (could also be parking edge, because PE can store multiple)
        junction_nodes = [*self.junction_nodes, self.graph_creator.processing_zone]
        combinations_of_circles = list(distinct_combinations(circles_dict.keys(), 2))

        def get_circle_nodes(circle: int) -> set[Node]:
            circle_or_path: list[Edge]
            # if next edge is free -> circle is just two edges -> can skip first and last node
            if len(circles_dict[circle]) == 2:
                if circles_dict[circle][0] != circles_dict[circle][1]:
                    circle_or_path = [(circles_dict[circle][0][1], circles_dict[circle][1][0])]
                    assert circles_dict[circle][0][1] == circles_dict[circle][1][0], (
                        "circle is not two edges? Middle node should be the same"
                    )
                    # if middle node is exit or exit connection -> skip also middle node -> can always push through to parking edge
                    # TODO unskip
                    # if (
                    #     nx.get_node_attributes(self.graph, "node_type")[circles_dict[circle][0][1]] in ("exit_node", "exit_connection_node")
                    # ):
                    #     circle_or_path = []

                # new if same edge twice is parking edge -> skip completely
                elif get_idx_from_idc(self.idc_dict, circles_dict[circle][0]) == get_idx_from_idc(
                    self.idc_dict, self.graph_creator.parking_edge
                ):
                    # if self.count_chains_in_parking() >= self.max_num_parking:
                    #     circle_or_path = [(circles_dict[circle][0][0], circles_dict[circle][0][0])]
                    # else:
                    circle_or_path = []
                else:  # else if path is same edge twice skip (but of course keep first node -> no movement into this edge)
                    circle_or_path = [(circles_dict[circle][0][0], circles_dict[circle][0][0])]
            # if circle is real circle -> need to check all nodes
            elif circles_dict[circle][0] == circles_dict[circle][-1]:
                circle_or_path = circles_dict[circle]

            # if circle is only a path -> can skip first and last node
            # extra clause for when there is a stop at the end? -> if last two edges are same -> skip last two edges?
            elif circles_dict[circle][-1] == circles_dict[circle][-2]:
                circle_or_path = circles_dict[circle][1:-2]
            else:
                circle_or_path = circles_dict[circle][1:-1]
            nodes = set()
            for edge in circle_or_path:
                node1, node2 = edge
                if node1 == node2:
                    nodes.add(node1)
                else:
                    nodes.add(node1)
                    nodes.add(node2)
            return nodes

        junction_shared_pairs = []
        for circle1, circle2 in combinations_of_circles:
            nodes1 = get_circle_nodes(circle1)
            nodes2 = get_circle_nodes(circle2)

            # following clause is only to allow some cases that the assert would stop
            # if (
            #     len(nodes1) != 0
            #     and len(nodes2) != 0
            #     and get_idx_from_idc(self.idc_dict, circles_dict[circle1][-1])
            #     != get_idx_from_idc(self.idc_dict, self.graph_creator.parking_edge)
            #     and not (
            #         len(nodes1.intersection(nodes2).intersection(junction_nodes)) > 0
            #         and self.graph_creator.processing_zone not in nodes1.intersection(nodes2)
            #     )
            #     # added new clause that it is allowed if they block each other (this is the if statement below, that adds the circles to junction_shared_pairs and thus the first blocks the second)
            # ):
            #     # assert that circles don't end in same edge (just sanity check at this point - exceptions are added in if clause above after being checked)
            #     assert get_idx_from_idc(self.idc_dict, circles_dict[circle1][-1]) != (
            #         get_idx_from_idc(self.idc_dict, circles_dict[circle2][-1])
            #     ), "circles end in same edge, was in if statement below as: or (get_idx_from_idc(self.idc_dict, circles_dict[circle1][-1]) == (get_idx_from_idc(self.idc_dict, circles_dict[circle2][-1]))), -> problem with circles: {}, {}".format(
            #         circles_dict[circle1], circles_dict[circle2]
            #     )

            # new: exclude processing zone node -> if pz node in circles -> can both be executed (TODO check again for moves out of pz)
            # extra: if both end in same edge -> don't execute (scenario where path out of pz ends in same edge as next edge for other)
            if (
                len(nodes1.intersection(nodes2).intersection(junction_nodes)) > 0
                and self.graph_creator.processing_zone not in nodes1.intersection(nodes2)
            ) or (
                get_idx_from_idc(self.idc_dict, circles_dict[circle1][-1])
                == (get_idx_from_idc(self.idc_dict, circles_dict[circle2][-1]))
            ):
                junction_shared_pairs.append((circle1, circle2))

        # free_circle_combs = [
        #     circle_idx_pair
        #     for circle_idx_pair in combinations_of_circles
        #     if (circle_idx_pair not in junction_shared_pairs and (circle_idx_pair[1], circle_idx_pair[0]) not in junction_shared_pairs)
        # ]
        return junction_shared_pairs  # , free_circle_combs

    # change: if list in other list -> take longer list, delete other
    # if list can be connected to other list -> combine and delete both

    def rotate(self, full_circle_idxs: list[int]) -> dict[int, int]:
        # create dictionary of state
        # convert keys to values and vice versa, so one can iterate over the path (positions need to be keys here)
        edge_state_dict = {}
        for ion, edge in self.ion_chains.items():
            edge_state_dict[get_idx_from_idc(self.idc_dict, edge)] = ion

        new_edge_state_dict = {}
        for edge_bef, edge_aft in pairwise(full_circle_idxs):
            try:
                new_edge_state_dict[edge_aft] = edge_state_dict[edge_bef]
                del edge_state_dict[edge_bef]
            except KeyError:
                continue

        # change ion chains
        for idx, ion in new_edge_state_dict.items():
            self.ion_chains[ion] = get_idc_from_idx(self.idc_dict, idx)

        return new_edge_state_dict

    def bfs_free_edge(self, node: Node, other_next_edges: list[Edge]) -> Edge:
        state_idxs = self.get_state_idxs()
        other_next_edges_idxs = [get_idx_from_idc(self.idc_dict, next_edge) for next_edge in other_next_edges]
        for edge_idc in nx.edge_bfs(self.mz_graph, node):
            if (
                get_idx_from_idc(self.idc_dict, edge_idc) not in state_idxs
                and get_idx_from_idc(self.idc_dict, edge_idc) not in other_next_edges_idxs
            ):
                return edge_idc
        msg = f"No free edge found for node {node}"
        raise RuntimeError(msg)

    def find_least_import_chain_in_parking(self, seq: list[int], ions_in_parking: list[int]) -> int:
        for num in seq:
            if num in ions_in_parking:
                ions_in_parking.remove(num)
                if len(ions_in_parking) == 1:
                    return ions_in_parking[0]
        return ions_in_parking[-1]
