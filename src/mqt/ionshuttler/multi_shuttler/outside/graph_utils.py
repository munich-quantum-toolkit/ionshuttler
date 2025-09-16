from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGDepNode

    from .types import Edge, Node


class Graph(nx.Graph):  # type: ignore [type-arg]
    @property
    def mz_graph(self) -> Graph:
        return self._mz_graph

    @mz_graph.setter
    def mz_graph(self, value: Graph) -> None:
        self._mz_graph = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value

    @property
    def idc_dict(self) -> dict[Edge, int]:
        if not hasattr(self, "_idc_dict"):
            self._idc_dict = create_idc_dictionary(self)
        return self._idc_dict

    @property
    def max_num_parking(self) -> int:
        return self._max_num_parking

    @max_num_parking.setter
    def max_num_parking(self, value: int) -> None:
        self._max_num_parking = value

    @property
    def pzs(self) -> list[ProcessingZone]:
        return self._pzs

    @pzs.setter
    def pzs(self, value: list[ProcessingZone]) -> None:
        parking_edges_idxs = []
        pzs_name_map = {}
        edge_to_pz_map = {}

        for pz in value:
            pz.max_num_parking = self.max_num_parking
            parking_idx = get_idx_from_idc(self.idc_dict, pz.parking_edge)
            parking_edges_idxs.append(parking_idx)
            pzs_name_map[pz.name] = pz
            # Populate edge_to_pz_map for edges belonging to this PZ's structure
            for edge_idx in pz.pz_edges_idx:
                edge_to_pz_map[edge_idx] = pz

        self._parking_edges_idxs = parking_edges_idxs
        self._pzs_name_map = pzs_name_map
        self._edge_to_pz_map = edge_to_pz_map
        self._pzs = value

    @property
    def parking_edges_idxs(self) -> list[int]:
        return self._parking_edges_idxs

    @property
    def pzs_name_map(self) -> dict[str, ProcessingZone]:
        return self._pzs_name_map

    @property
    def edge_to_pz_map(self) -> dict[int, ProcessingZone]:
        return self._edge_to_pz_map

    @property
    def plot(self) -> bool:
        return self._plot

    @plot.setter
    def plot(self, value: bool) -> None:
        self._plot = value

    @property
    def save(self) -> bool:
        return self._save

    @save.setter
    def save(self, value: bool) -> None:
        self._save = value

    @property
    def state(self) -> dict[int, Edge]:
        return self._state

    @state.setter
    def state(self, value: dict[int, Edge]) -> None:
        self._state = value

    @property
    def sequence(self) -> list[tuple[int, ...]]:
        return self._sequence

    @sequence.setter
    def sequence(self, value: list[tuple[int, ...]]) -> None:
        self._sequence = value

    @property
    def locked_gates(self) -> dict[tuple[int, ...], str]:
        return self._locked_gates

    @locked_gates.setter
    def locked_gates(self, value: dict[tuple[int, ...], str]) -> None:
        self._locked_gates = value

    @property
    def in_process(self) -> list[int]:
        return self._in_process

    @in_process.setter
    def in_process(self, value: list[int]) -> None:
        self._in_process = value

    @property
    def arch(self) -> str:
        return self._arch

    @arch.setter
    def arch(self, value: str) -> None:
        self._arch = value

    @property
    def map_to_pz(self) -> dict[int, str]:
        return self._map_to_pz

    @map_to_pz.setter
    def map_to_pz(self, value: dict[int, str]) -> None:
        self._map_to_pz = value

    @property
    def next_gate_at_pz(self) -> dict[str, tuple[int, ...]]:
        return self._next_gate_at_pz

    @next_gate_at_pz.setter
    def next_gate_at_pz(self, value: dict[str, tuple[int, ...]]) -> None:
        self._next_gate_at_pz = value

    @property
    def dist_dict(self) -> dict[str, dict[Edge, list[Node]]]:
        if not hasattr(self, "_dist_dict"):
            self._dist_dict = create_dist_dict(self)
        return self._dist_dict

    @dist_dict.setter
    def dist_dict(self, value: dict[str, dict[Edge, list[Node]]]) -> None:
        self._dist_dict = value

    @property
    def junction_nodes(self) -> list[Node]:
        return self._junction_nodes

    @junction_nodes.setter
    def junction_nodes(self, value: list[Node]) -> None:
        self._junction_nodes = value


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
    from .cycles import find_path_edge_to_edge  # noqa: PLC0415

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


class ProcessingZone:
    def __init__(self, name, info):
        self.name = name
        self.pz_info = info
        self.exit_node = info[0]
        self.entry_node = info[1]
        self.processing_zone = info[2]

    @property
    def parking_node(self) -> Node:
        return self._parking_node

    @parking_node.setter
    def parking_node(self, value: Node) -> None:
        self._parking_node = value

    @property
    def parking_edge(self) -> Edge:
        return self._parking_edge

    @parking_edge.setter
    def parking_edge(self, value: Edge) -> None:
        self._parking_edge = value

    @property
    def time_in_pz_counter(self) -> int:
        return self._time_in_pz_counter

    @time_in_pz_counter.setter
    def time_in_pz_counter(self, value: int) -> None:
        self._time_in_pz_counter = value

    @property
    def gate_execution_finished(self) -> bool:
        return self._gate_execution_finished

    @gate_execution_finished.setter
    def gate_execution_finished(self, value: bool) -> None:
        self._gate_execution_finished = value

    @property
    def getting_processed(self) -> list[DAGDepNode]:
        return self._getting_processed

    @getting_processed.setter
    def getting_processed(self, value: list[DAGDepNode]) -> None:
        self._getting_processed = value

    @property
    def rotate_entry(self) -> bool:
        return self._rotate_entry

    @rotate_entry.setter
    def rotate_entry(self, value: bool) -> None:
        self._rotate_entry = value

    @property
    def out_of_parking_cycle(self) -> int | None:
        return self._out_of_parking_cycle

    @out_of_parking_cycle.setter
    def out_of_parking_cycle(self, value: int | None) -> None:
        self._out_of_parking_cycle = value

    @property
    def out_of_parking_move(self) -> int | None:
        return self._out_of_parking_move

    @out_of_parking_move.setter
    def out_of_parking_move(self, value: int | None) -> None:
        self._out_of_parking_move = value

    @property
    def entry_edge(self) -> Edge:
        return self._entry_edge

    @entry_edge.setter
    def entry_edge(self, value: Edge) -> None:
        self._entry_edge = value

    @property
    def exit_edge(self) -> Edge:
        return self._exit_edge

    @exit_edge.setter
    def exit_edge(self, value: Edge) -> None:
        self._exit_edge = value

    @property
    def ion_to_move_out_of_pz(self) -> int | None:
        return self._ion_to_move_out_of_pz

    @ion_to_move_out_of_pz.setter
    def ion_to_move_out_of_pz(self, value: int | None) -> None:
        self._ion_to_move_out_of_pz = value

    @property
    def path_from_pz(self) -> list[Edge]:
        return self._path_from_pz

    @path_from_pz.setter
    def path_from_pz(self, value: list[Edge]) -> None:
        self._path_from_pz = value

    @property
    def rest_of_path_from_pz(self) -> dict[Edge, list[Edge]]:
        return self._rest_of_path_from_pz

    @rest_of_path_from_pz.setter
    def rest_of_path_from_pz(self, value: dict[Edge, list[Edge]]) -> None:
        self._rest_of_path_from_pz = value

    @property
    def path_to_pz(self) -> list[Edge]:
        return self._path_to_pz

    @path_to_pz.setter
    def path_to_pz(self, value: list[Edge]) -> None:
        self._path_to_pz = value

    @property
    def rest_of_path_to_pz(self) -> dict[Edge, list[Edge]]:
        return self._rest_of_path_to_pz

    @rest_of_path_to_pz.setter
    def rest_of_path_to_pz(self, value: dict[Edge, list[Edge]]) -> None:
        self._rest_of_path_to_pz = value

    @property
    def first_entry_connection_from_pz(self) -> Edge:
        return self._first_entry_connection_from_pz

    @first_entry_connection_from_pz.setter
    def first_entry_connection_from_pz(self, value: Edge) -> None:
        self._first_entry_connection_from_pz = value

    @property
    def ion_to_park(self) -> int | None:
        return self._ion_to_park

    @ion_to_park.setter
    def ion_to_park(self, value: int | None) -> None:
        self._ion_to_park = value

    @property
    def max_num_parking(self) -> int:
        return self._max_num_parking

    @max_num_parking.setter
    def max_num_parking(self, value: int) -> None:
        self._max_num_parking = value

    @property
    def path_to_pz_idxs(self) -> list[int]:
        return self._path_to_pz_idxs

    @path_to_pz_idxs.setter
    def path_to_pz_idxs(self, value: list[int]) -> None:
        self._path_to_pz_idxs = value

    @property
    def path_from_pz_idxs(self) -> list[int]:
        return self._path_from_pz_idxs

    @path_from_pz_idxs.setter
    def path_from_pz_idxs(self, value: list[int]) -> None:
        self._path_from_pz_idxs = value

    @property
    def pz_edges_idx(self) -> list[int]:
        return self._pz_edges_idx

    @pz_edges_idx.setter
    def pz_edges_idx(self, value: list[int]) -> None:
        self._pz_edges_idx = value

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @num_edges.setter
    def num_edges(self, value: int) -> None:
        self._num_edges = value


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
