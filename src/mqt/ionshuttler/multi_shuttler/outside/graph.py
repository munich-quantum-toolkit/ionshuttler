from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from .graph_utils import create_dist_dict, create_idc_dictionary, get_idx_from_idc

if TYPE_CHECKING:
    from .processing_zone import ProcessingZone
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
