from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import networkx as nx

from .graph_utils import create_dist_dict, create_idc_dictionary, get_idx_from_idc

if TYPE_CHECKING:
    from .ion_types import Edge, Node
    from .processing_zone import ProcessingZone


@dataclass
class RunStats:
    pre_selected_cycles_total: int = 0
    pre_selected_paths_total: int = 0
    selected_cycles_total: int = 0
    selected_paths_total: int = 0
    per_timestep: list[dict[str, int]] = field(default_factory=list)

    def record_selection_stats(self, timestep: int, cycles: int, paths: int) -> None:
        self.selected_cycles_total += cycles
        self.selected_paths_total += paths
        self.per_timestep.append({"timestep": timestep, "cycles": cycles, "paths": paths})

    def record_move_stats(self, timestep: int, cycles: int, paths: int) -> None:
        self.pre_selected_cycles_total += cycles
        self.pre_selected_paths_total += paths
        self.per_timestep.append({"timestep": timestep, "cycles": cycles, "paths": paths})


class Graph(nx.Graph):  # type: ignore [type-arg]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.executed_gates_next: list[dict[str, object]] = []
        self._pz_assignment_policy: str = "legacy"  # default
        self.run_stats: RunStats = RunStats()
        self.path_cache: dict[Any, Any] = {}
        self.max_timesteps: int = 1_000_000
        self.parameter: float = 1.0

        # typed dynamic attrs used across scheduling/cycles/shuttle
        self.pre_last_selected_move_stats: dict[str, int] = {"cycles": 0, "paths": 0}
        self.last_selected_move_stats: dict[str, int] = {"cycles": 0, "paths": 0}
        self.rotated_ions: list[int] = []
        self._bridge_set_cache: Any = None
        self.in_process: dict[str, list[int]] = {}

    @property
    def pz_assignment_policy(self) -> str:
        return self._pz_assignment_policy

    @pz_assignment_policy.setter
    def pz_assignment_policy(self, value: str) -> None:
        self._pz_assignment_policy = value

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
    def in_process(self) -> dict[str, list[int]]:
        return self._in_process

    @in_process.setter
    def in_process(self, value: dict[str, list[int]]) -> None:
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
