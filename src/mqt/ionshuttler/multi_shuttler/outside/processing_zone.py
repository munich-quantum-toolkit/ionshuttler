from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGDepNode

    from .types import Edge, Node


class ProcessingZone:
    def __init__(self, name: str, info: list[tuple[float, float]]) -> None:
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
