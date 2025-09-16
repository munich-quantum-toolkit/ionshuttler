from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from .graph_utils import create_idc_dictionary

if TYPE_CHECKING:
    from .processing_zone import ProcessingZone
    from .types import Edge, Node


class Graph(nx.Graph):  # type: ignore [type-arg]
    @property
    def junction_nodes(self) -> list[Node]:
        return self._junction_nodes

    @junction_nodes.setter
    def junction_nodes(self, value: list[Node]) -> None:
        self._junction_nodes = value

    @property
    def pzs(self) -> list[ProcessingZone]:
        return self._pzs

    @pzs.setter
    def pzs(self, value: list[ProcessingZone]) -> None:
        self._pzs = value

    @property
    def locked_gates(self) -> dict[tuple[int, ...], str]:
        return self._locked_gates

    @locked_gates.setter
    def locked_gates(self, value: dict[tuple[int, ...], str]) -> None:
        self._locked_gates = value

    @property
    def state(self) -> dict[int, Edge]:
        return self._state

    @state.setter
    def state(self, value: dict[int, Edge]) -> None:
        self._state = value

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
    def sequence(self) -> list[tuple[int, ...]]:
        return self._sequence

    @sequence.setter
    def sequence(self, value: list[tuple[int, ...]]) -> None:
        self._sequence = value

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
    def stop_moves(self) -> list[int]:
        return self._stop_moves

    @stop_moves.setter
    def stop_moves(self, value: list[int]) -> None:
        self._stop_moves = value

    @property
    def idc_dict(self) -> dict[int, Edge]:
        if not hasattr(self, "_idc_dict"):
            self._idc_dict = create_idc_dictionary(self)
        return self._idc_dict

    @property
    def map_to_pz(self) -> dict[int, str]:
        return self._map_to_pz

    @map_to_pz.setter
    def map_to_pz(self, value: dict[int, str]) -> None:
        self._map_to_pz = value
