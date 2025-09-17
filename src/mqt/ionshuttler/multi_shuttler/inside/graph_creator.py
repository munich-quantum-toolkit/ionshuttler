from __future__ import annotations

import random
from typing import TYPE_CHECKING

import networkx as nx

from .graph import Graph

if TYPE_CHECKING:
    from .processing_zone import ProcessingZone


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
        networkx_graph: Graph = nx.grid_2d_graph(self.m_extended, self.n_extended, create_using=Graph)
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
        nx.set_node_attributes(networkx_graph, values=dict.fromkeys(networkx_graph.nodes(), 1), name="weight")

        return networkx_graph

    def _set_trap_nodes(self, networkx_graph: Graph) -> None:
        for node in networkx_graph.nodes():
            networkx_graph.add_node(node, node_type="trap_node", color="b")

    def _remove_edges(self, networkx_graph: Graph) -> None:
        self._remove_horizontal_edges(networkx_graph)
        self._remove_vertical_edges(networkx_graph)

    def _remove_nodes(self, networkx_graph: Graph) -> None:
        self._remove_horizontal_nodes(networkx_graph)

    def _remove_horizontal_edges(self, networkx_graph: Graph) -> None:
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(self.n_extended - 1):
                    networkx_graph.remove_edge((i + k, j), (i + k, j + 1))

    def _remove_vertical_edges(self, networkx_graph: Graph) -> None:
        for i in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
            for k in range(1, self.ion_chain_size_horizontal):
                for j in range(self.m_extended - 1):
                    networkx_graph.remove_edge((j, i + k), (j + 1, i + k))

    def _remove_horizontal_nodes(self, networkx_graph: Graph) -> None:
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
                    for s in range(1, self.ion_chain_size_horizontal):
                        networkx_graph.remove_node((i + k, j + s))

    def _set_junction_nodes(self, networkx_graph: Graph) -> None:
        for i in range(0, self.m_extended, self.ion_chain_size_vertical):
            for j in range(0, self.n_extended, self.ion_chain_size_horizontal):
                networkx_graph.add_node((i, j), node_type="junction_node", color="g")
                networkx_graph.junction_nodes.append((i, j))

    def _remove_junctions(self, networkx_graph: Graph, num_nodes_to_remove: int) -> None:
        """
        Removes a specified number of nodes from the graph, excluding nodes of type 'exit_node' or 'entry_node'.
        """
        #  Filter out nodes that are of type 'exit_node' or 'entry_node'
        nodes_to_remove = [
            node
            for node, data in networkx_graph.nodes(data=True)
            if data.get("node_type") not in {"exit_node", "entry_node"}
        ]

        # Shuffle the list of nodes to remove
        random.seed(0)
        random.shuffle(nodes_to_remove)

        # Remove the specified number of nodes
        for node in nodes_to_remove[:num_nodes_to_remove]:
            networkx_graph.remove_node(node)

    def get_graph(self) -> Graph:
        return self.networkx_graph
