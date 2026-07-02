from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

from .graph_utils import create_idc_dictionary

if TYPE_CHECKING:
    from ..circuit_types import GateInfo
    from .ion_types import Edge, Node
    from .processing_zone import ProcessingZone


GateRef = int | tuple[int, ...]


class Graph(nx.Graph):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._executed_gates_next = []
        self._locked_gates = {}
        self._gate_info = {}
        self._gate_pz_assignment = {}
        self._in_process = []

    @property
    def executed_gates_next(self) -> list[dict[str, Any]]:
        return self._executed_gates_next

    @executed_gates_next.setter
    def executed_gates_next(self, value: list[dict[str, Any]]) -> None:
        self._executed_gates_next = value

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
    def locked_gates(self) -> dict[int, str]:
        return self._locked_gates

    @locked_gates.setter
    def locked_gates(self, value: dict[int, str]) -> None:
        self._locked_gates = value

    @property
    def gate_info(self) -> dict[int, GateInfo]:
        return self._gate_info

    @gate_info.setter
    def gate_info(self, value: dict[int, GateInfo]) -> None:
        self._gate_info = value

    def gate_qubits(self, gate: GateRef) -> tuple[int, ...]:
        if isinstance(gate, int):
            return self.gate_info[gate].qubits
        return gate

    @property
    def gate_pz_assignment(self) -> dict[int, str]:
        return self._gate_pz_assignment

    @gate_pz_assignment.setter
    def gate_pz_assignment(self, value: dict[int, str]) -> None:
        self._gate_pz_assignment = value

    def preferred_pz_for_gate(self, gate_id: int) -> str | None:
        return self.gate_pz_assignment.get(gate_id)

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
    def sequence(self) -> list[int]:
        return self._sequence

    @sequence.setter
    def sequence(self, value: list[int]) -> None:
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
