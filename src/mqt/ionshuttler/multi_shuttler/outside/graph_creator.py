from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import networkx as nx

from .graph import Graph
from .graph_utils import convert_nodes_to_float, create_idc_dictionary, get_idx_from_idc

if TYPE_CHECKING:
    from .processing_zone import ProcessingZone
    from .types import Edge, Node


class GraphCreator:
    def __init__(
        self,
        m: int,
        n: int,
        ion_chain_size_vertical: int,
        ion_chain_size_horizontal: int,
        failing_junctions: int,
        pz_info: list[ProcessingZone],
    ):
        self.m = m
        self.n = n
        self.ion_chain_size_vertical = ion_chain_size_vertical
        self.ion_chain_size_horizontal = ion_chain_size_horizontal
        self.failing_junctions = failing_junctions
        self.pz_info = pz_info
        self.m_extended = self.m + (self.ion_chain_size_vertical - 1) * (self.m - 1)
        self.n_extended = self.n + (self.ion_chain_size_horizontal - 1) * (self.n - 1)
        self.networkx_graph = self.create_graph()

    def create_graph(self) -> Graph:
        networkx_graph = nx.grid_2d_graph(self.m_extended, self.n_extended, create_using=Graph)
        # Convert nodes to float
        networkx_graph = convert_nodes_to_float(networkx_graph)
        # color all edges black
        nx.set_edge_attributes(networkx_graph, values=dict.fromkeys(networkx_graph.edges(), "k"), name="color")
        # num_edges needed for outer pz (length of one-way connection - exit/entry)
        self._set_trap_nodes(networkx_graph)
        self._remove_edges(networkx_graph)
        self._remove_nodes(networkx_graph)
        networkx_graph.junction_nodes = []
        self._set_junction_nodes(networkx_graph)
        # if self.pz == 'mid':
        #     self._remove_mid_part(networkx_graph)
        self._remove_junctions(networkx_graph, self.failing_junctions)
        nx.set_edge_attributes(networkx_graph, values=dict.fromkeys(networkx_graph.edges(), "trap"), name="edge_type")
        nx.set_edge_attributes(networkx_graph, values=dict.fromkeys(networkx_graph.edges(), 1), name="weight")

        return networkx_graph

    def _set_trap_nodes(self, networkx_graph: Graph) -> None:
        for node in networkx_graph.nodes():
            float_node = (float(node[0]), float(node[1]))
            networkx_graph.add_node(float_node, node_type="trap_node", color="k", node_size=100)

    def _remove_edges(self, networkx_graph: Graph) -> None:
        self._remove_horizontal_edges(networkx_graph)
        self._remove_vertical_edges(networkx_graph)

    def _remove_nodes(self, networkx_graph: Graph) -> None:
        self._remove_horizontal_nodes(networkx_graph)

    def _remove_horizontal_edges(self, networkx_graph: Graph) -> None:
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(self.n_extended - 1):
                    node1 = (float(i + k), float(j))
                    node2 = (float(i + k), float(j + 1))
                    networkx_graph.remove_edge(node1, node2)

    def _remove_vertical_edges(self, networkx_graph: Graph) -> None:
        for i in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
            for k in range(1, self.ion_chain_size_horizontal):
                for j in range(self.m_extended - 1):
                    node1 = (float(j), float(i + k))
                    node2 = (float(j + 1), float(i + k))
                    networkx_graph.remove_edge(node1, node2)

    def _remove_horizontal_nodes(self, networkx_graph: Graph) -> None:
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
                    for s in range(1, self.ion_chain_size_horizontal):
                        node = (float(i + k), float(j + s))
                        networkx_graph.remove_node(node)

    def _set_junction_nodes(self, networkx_graph: Graph) -> None:
        for i in range(0, self.m_extended, self.ion_chain_size_vertical):
            for j in range(0, self.n_extended, self.ion_chain_size_horizontal):
                float_node = (float(i), float(j))
                networkx_graph.add_node(float_node, node_type="junction_node", color="g", node_size=200)
                networkx_graph.junction_nodes.append(float_node)

    def _remove_junctions(self, networkx_graph: Graph, num_nodes_to_remove: int) -> None:
        """
        Removes a specified number of nodes from the graph, excluding nodes of type 'exit_node' or 'entry_node'.
        """
        #  Filter out nodes that are of type 'exit_node' or 'entry_node'
        nodes_to_remove: list[Node] = [
            node
            for node, data in networkx_graph.nodes(data=True)
            if data.get("node_type") not in {"exit_node", "entry_node", "exit_connection_node", "entry_connection_node"}
        ]

        # Shuffle the list of nodes to remove
        random.seed(0)
        random.shuffle(nodes_to_remove)

        # Remove the specified number of nodes
        for node in nodes_to_remove[:num_nodes_to_remove]:
            networkx_graph.remove_node(node)

        random.seed()

    def get_graph(self) -> Graph:
        return self.networkx_graph


class PZCreator(GraphCreator):
    def __init__(
        self,
        m: int,
        n: int,
        ion_chain_size_vertical: int,
        ion_chain_size_horizontal: int,
        failing_junctions: int,
        pzs: list[ProcessingZone],
    ):
        super().__init__(m, n, ion_chain_size_vertical, ion_chain_size_horizontal, failing_junctions, pzs)
        self.pzs = pzs

        for pz in pzs:
            self._set_processing_zone(self.networkx_graph, pz)

        self.idc_dict = create_idc_dictionary(self.networkx_graph)
        self.get_pz_from_edge = {}
        self.parking_edges_of_pz = {}
        self.processing_zone_nodes_of_pz = {}
        for pz in self.pzs:
            self.parking_edges_of_pz[pz] = get_idx_from_idc(self.idc_dict, pz.parking_edge)
            self.processing_zone_nodes_of_pz[pz] = pz.processing_zone
            pz.path_to_pz_idxs = [get_idx_from_idc(self.idc_dict, edge) for edge in pz.path_to_pz]
            pz.path_from_pz_idxs = [get_idx_from_idc(self.idc_dict, edge) for edge in pz.path_from_pz]
            pz.rest_of_path_to_pz = {edge: pz.path_to_pz[i + 1 :] for i, edge in enumerate(pz.path_to_pz)}
            pz.rest_of_path_from_pz = {edge: pz.path_from_pz[i + 1 :] for i, edge in enumerate(pz.path_from_pz)}
            pz.pz_edges_idx = [
                *pz.path_to_pz_idxs,
                get_idx_from_idc(self.idc_dict, pz.parking_edge),
                *pz.path_from_pz_idxs,
            ]
            for edge in pz.pz_edges_idx:
                self.get_pz_from_edge[edge] = pz

    def find_shared_border(self, node1: Node, node2: Node) -> str | None:
        x1, y1 = node1
        x2, y2 = node2

        # Check for shared row (Top or Bottom border)
        if x1 == x2:
            if x1 == 0:
                return "top"
            if x1 == self.m_extended - 1:
                return "bottom"

        # Check for shared column (Left or Right border)
        if y1 == y2:
            if y1 == 0:
                return "left"
            if y1 == self.n_extended - 1:
                return "right"

        return None

    def _set_processing_zone(self, networkx_graph: Graph, pz: ProcessingZone) -> Graph:
        border = self.find_shared_border(pz.exit_node, pz.entry_node)

        # Define the parking edge (edge between processing zone and parking node)
        if border == "top":
            pz.parking_node = (pz.processing_zone[0] - 2, pz.processing_zone[1])  # Above processing zone
        elif border == "bottom":
            pz.parking_node = (pz.processing_zone[0] + 2, pz.processing_zone[1])  # Below processing zone
        elif border == "left":
            pz.parking_node = (pz.processing_zone[0], pz.processing_zone[1] - 2)  # Left of processing zone
        elif border == "right":
            pz.parking_node = (pz.processing_zone[0], pz.processing_zone[1] + 2)  # Right of processing zone
        pz.parking_edge = (pz.processing_zone, pz.parking_node)

        # Number of edges between exit/entry and processing zone (size of one-way connection)
        if border in {"top", "bottom"}:
            pz.num_edges = math.ceil(
                math.ceil(abs(pz.entry_node[1] - pz.exit_node[1]) / self.ion_chain_size_horizontal) / 2
            )  # Number of edges between exit/entry and processing zone
        elif border in {"left", "right"}:
            pz.num_edges = math.ceil(
                math.ceil(abs(pz.entry_node[0] - pz.exit_node[0]) / self.ion_chain_size_vertical) / 2
            )  # Number of edges between exit/entry and processing zone

        # differences
        dx_exit = pz.processing_zone[0] - pz.exit_node[0]
        dx_entry = pz.entry_node[0] - pz.processing_zone[0]
        dy_exit = pz.exit_node[1] - pz.processing_zone[1]
        dy_entry = pz.processing_zone[1] - pz.entry_node[1]

        pz.path_to_pz = []
        pz.path_from_pz = []

        # Add exit edges
        for i in range(pz.num_edges):
            exit_node = (
                float(pz.exit_node[0] + (i + 1) * dx_exit / pz.num_edges),
                float(pz.exit_node[1] - (i + 1) * dy_exit / pz.num_edges),
            )

            if i == 0:
                # networkx_graph.add_node(exit_node, node_type="exit_node", color="y") # will get overwritten by exit_connection_node
                previous_exit_node = pz.exit_node
                pz.exit_edge = (previous_exit_node, exit_node)

            networkx_graph.add_node(exit_node, node_type="exit_connection_node", color="g", node_size=200)
            networkx_graph.junction_nodes.append(exit_node)
            networkx_graph.add_edge(previous_exit_node, exit_node, edge_type="exit", color="g")
            pz.path_to_pz.append((previous_exit_node, exit_node))
            previous_exit_node = exit_node

        # Add entry edges
        for i in range(pz.num_edges):
            entry_node = (
                float(pz.entry_node[0] - (i + 1) * dx_entry / pz.num_edges),
                float(pz.entry_node[1] + (i + 1) * dy_entry / pz.num_edges),
            )
            if i == 0:
                # networkx_graph.add_node(entry_node, node_type="entry_node", color="orange")
                previous_entry_node = pz.entry_node
                pz.entry_edge = (previous_entry_node, entry_node)

            networkx_graph.add_node(entry_node, node_type="entry_connection_node", color="g", node_size=200)
            networkx_graph.junction_nodes.append(entry_node)
            if entry_node == pz.processing_zone:
                pz.first_entry_connection_from_pz = (entry_node, previous_entry_node)
                networkx_graph.add_edge(previous_entry_node, entry_node, edge_type="first_entry_connection", color="g")
            else:
                networkx_graph.add_edge(previous_entry_node, entry_node, edge_type="entry", color="g")
            pz.path_from_pz.insert(0, (entry_node, previous_entry_node))

            previous_entry_node = entry_node

        assert exit_node == entry_node, "Exit and entry do not end in same node"
        assert exit_node == pz.processing_zone, "Exit and entry do not end in processing zone"

        # Add the processing zone node
        networkx_graph.add_node(pz.processing_zone, node_type="processing_zone_node", color="r", node_size=100)

        # new: add exit and entry node
        networkx_graph.add_node(pz.exit_node, node_type="exit_node", color="g", node_size=200)
        networkx_graph.add_node(pz.entry_node, node_type="entry_node", color="g")
        networkx_graph.junction_nodes.append(pz.exit_node)
        networkx_graph.junction_nodes.append(pz.entry_node)

        # Add new parking edge
        networkx_graph.add_node(pz.parking_node, node_type="parking_node", color="r", node_size=200)
        networkx_graph.add_edge(pz.parking_edge[0], pz.parking_edge[1], edge_type="parking_edge", color="r")
        networkx_graph.junction_nodes.append(pz.parking_node)
        # add new info to pz
        # not needed? already done above? pz.parking_node =

        return networkx_graph

    def order_edges(self, edge1: Edge, edge2: Edge) -> tuple[Edge, Edge]:
        # Find the common node shared between the two edges
        common_nodes = set(edge1).intersection(set(edge2))

        if len(common_nodes) != 1 and edge1 != edge2:
            msg = f"The input edges are not connected. Edges: {edge1}, {edge2}"
            raise ValueError(msg)

        common_node = common_nodes.pop()
        if edge1[0] == common_node:
            edge1_in_order = (edge1[1], common_node)
            edge2_in_order = (common_node, edge2[1]) if edge2[0] == common_node else (common_node, edge2[0])
        else:
            edge1_in_order = (edge1[0], common_node)
            edge2_in_order = (common_node, edge2[1]) if edge2[0] == common_node else (common_node, edge2[0])

        return edge1_in_order, edge2_in_order

    def find_connected_edges(self) -> list[list[Edge]]:
        connected_edge_pairs = set()
        for edge in self.networkx_graph.edges():
            node1, node2 = edge
            # Find edges connected to node1
            for neighbor in self.networkx_graph.neighbors(node1):
                if neighbor != node2:  # avoid the original edge
                    edge_pair = tuple(sorted([edge, (node1, neighbor)]))
                    connected_edge_pairs.add(edge_pair)
            # Find edges connected to node2
            for neighbor in self.networkx_graph.neighbors(node2):
                if neighbor != node1:  # avoid the original edge
                    edge_pair = tuple(sorted([edge, (node2, neighbor)]))
                    connected_edge_pairs.add(edge_pair)
        # order edges (also include reverse order -> opposite direction moves are now needed if a junction fails)
        connected_edge_pair_list = [
            self.order_edges(edge_pair[0], edge_pair[1]) for edge_pair in connected_edge_pairs
        ] + [self.order_edges(edge_pair[1], edge_pair[0]) for edge_pair in connected_edge_pairs]
        # Convert set of tuples to a list of lists
        return [list(pair) for pair in connected_edge_pair_list]
