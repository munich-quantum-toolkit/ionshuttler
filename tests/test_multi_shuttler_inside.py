"""Tests for the multi_shuttler (heuristic solver) subpackage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from mqt.ionshuttler.multi_shuttler.circuit_types import GateInfo
from mqt.ionshuttler.multi_shuttler.inside.scheduling import create_priority_queue

if TYPE_CHECKING:
    from mqt.ionshuttler.multi_shuttler.inside.graph import Graph


class TestMultiCompilation:
    """Tests for multi_shuttler.inside.compilation."""

    def test_inside_priority_queue_accepts_gate_ids(self):
        """The inside scheduler should accept gate-id sequences backed by metadata."""
        gate_info = {
            0: GateInfo(qubits=(0,), qasm="x q[0];"),
            1: GateInfo(qubits=(1, 2), qasm="cx q[1],q[2];"),
        }
        graph = SimpleNamespace(
            sequence=[0, 1],
            gate_info=gate_info,
            gate_qubits=lambda gate: gate_info[gate].qubits if isinstance(gate, int) else gate,
            preferred_pz_for_gate=lambda _gate_id: None,
            map_to_pz={0: "pz1", 1: "pz1", 2: "pz2"},
            locked_gates={},
            pzs=[SimpleNamespace(name="pz1"), SimpleNamespace(name="pz2")],
        )

        with patch(
            "mqt.ionshuttler.multi_shuttler.inside.scheduling.pick_pz_for_2_q_gate",
            return_value="pz2",
        ):
            priority_queue, next_gate_at_pz = create_priority_queue(cast("Graph", graph))

        assert priority_queue == {0: "pz1", 1: "pz2", 2: "pz2"}
        assert next_gate_at_pz == {"pz1": 0, "pz2": 1}

    def test_inside_priority_queue_prefers_explicit_gate_assignment(self):
        """The inside scheduler should honor explicit gate-to-PZ overrides."""
        gate_info = {
            0: GateInfo(qubits=(0,), qasm="x q[0];"),
            1: GateInfo(qubits=(1, 2), qasm="cx q[1],q[2];"),
        }
        graph = SimpleNamespace(
            sequence=[0, 1],
            gate_info=gate_info,
            gate_qubits=lambda gate: gate_info[gate].qubits if isinstance(gate, int) else gate,
            gate_pz_assignment={0: "pz2", 1: "pz1"},
            preferred_pz_for_gate={0: "pz2", 1: "pz1"}.get,
            map_to_pz={0: "pz1", 1: "pz1", 2: "pz2"},
            locked_gates={},
            pzs=[SimpleNamespace(name="pz1"), SimpleNamespace(name="pz2")],
        )

        priority_queue, next_gate_at_pz = create_priority_queue(cast("Graph", graph))

        assert priority_queue == {0: "pz2", 1: "pz1", 2: "pz1"}
        assert next_gate_at_pz == {"pz1": 1, "pz2": 0}
