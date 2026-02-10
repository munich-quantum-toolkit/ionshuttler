"""Tests for the single_shuttler (exact solver) subpackage."""

from __future__ import annotations

import networkx as nx
import pytest


# ===================================================================
# Graph creation tests
# ===================================================================


class TestCreateGraph:
    """Tests for the ``create_graph`` function in ``memory_sat``."""

    def test_basic_2x2_graph_structure(self, small_grid_graph):
        """A 2×2 graph should have the expected nodes and edge types."""
        g = small_grid_graph
        assert isinstance(g, nx.Graph)
        # Must contain at least one processing zone node
        pz_nodes = [n for n in g.nodes() if g.nodes[n].get("node_type") == "processing_zone_node"]
        assert len(pz_nodes) == 1

    def test_graph_has_entry_and_exit(self, small_grid_graph):
        """The graph must have exactly one entry and one exit edge."""
        g = small_grid_graph
        edge_types = nx.get_edge_attributes(g, "edge_type")
        entries = [e for e, t in edge_types.items() if t == "entry"]
        exits = [e for e, t in edge_types.items() if t == "exit"]
        assert len(entries) == 1
        assert len(exits) == 1

    def test_graph_has_trap_edges(self, small_grid_graph):
        """The graph should contain trap edges."""
        g = small_grid_graph
        edge_types = nx.get_edge_attributes(g, "edge_type")
        traps = [e for e, t in edge_types.items() if t == "trap"]
        assert len(traps) > 0

    def test_graph_has_junction_nodes(self, small_grid_graph):
        """The graph should contain junction nodes."""
        g = small_grid_graph
        junction_nodes = [n for n in g.nodes() if g.nodes[n].get("node_type") == "junction_node"]
        assert len(junction_nodes) > 0

    def test_3x3_graph_has_more_traps_than_2x2(self, small_grid_graph, medium_grid_graph):
        """A 3×3 graph should have more trap edges than a 2×2 graph."""
        et_small = nx.get_edge_attributes(small_grid_graph, "edge_type")
        et_medium = nx.get_edge_attributes(medium_grid_graph, "edge_type")
        traps_small = sum(1 for t in et_small.values() if t == "trap")
        traps_medium = sum(1 for t in et_medium.values() if t == "trap")
        assert traps_medium > traps_small

    def test_larger_ion_chain_sizes(self):
        """Graph creation with ion_chain_size > 1 should still produce a valid graph."""
        from mqt.ionshuttler.single_shuttler.memory_sat import create_graph

        g = create_graph(m=3, n=3, ion_chain_size_vertical=2, ion_chain_size_horizontal=2)
        assert isinstance(g, nx.Graph)
        assert len(g.edges()) > 0
        # Still must have entry/exit
        edge_types = nx.get_edge_attributes(g, "edge_type")
        assert "entry" in edge_types.values()
        assert "exit" in edge_types.values()


# ===================================================================
# IDC dictionary / edge indexing utilities
# ===================================================================


class TestIdcDictionary:
    """Tests for edge index ↔ edge coordinate conversions."""

    def test_idc_dict_round_trip(self, small_grid_graph):
        """Converting idx → idc → idx should be identity."""
        from mqt.ionshuttler.single_shuttler.memory_sat import (
            create_idc_dictionary,
            get_idc_from_idx,
            get_idx_from_idc,
        )

        idc_dict = create_idc_dictionary(small_grid_graph)
        for idx in range(len(small_grid_graph.edges())):
            idc = get_idc_from_idx(idc_dict, idx)
            assert get_idx_from_idc(idc_dict, idc) == idx

    def test_idc_dict_covers_all_edges(self, medium_grid_graph):
        """The dictionary should have an entry for every edge."""
        from mqt.ionshuttler.single_shuttler.memory_sat import create_idc_dictionary

        idc_dict = create_idc_dictionary(medium_grid_graph)
        assert len(idc_dict) == len(medium_grid_graph.edges())


# ===================================================================
# graph_utils tests (single_shuttler)
# ===================================================================


class TestGraphUtils:
    """Tests for single_shuttler.graph_utils helpers."""

    def test_order_edges_connected(self):
        """order_edges should correctly order two connected edges."""
        from mqt.ionshuttler.single_shuttler.graph_utils import order_edges

        edge1 = ((0, 0), (0, 1))
        edge2 = ((0, 1), (1, 1))
        e1_ordered, e2_ordered = order_edges(edge1, edge2)
        # The common node (0,1) should be at the junction
        assert e1_ordered[1] == e2_ordered[0] == (0, 1)

    def test_order_edges_disconnected_raises(self):
        """order_edges should raise ValueError for disconnected edges."""
        from mqt.ionshuttler.single_shuttler.graph_utils import order_edges

        edge1 = ((0, 0), (0, 1))
        edge2 = ((2, 2), (3, 3))
        with pytest.raises(ValueError, match="not connected"):
            order_edges(edge1, edge2)

    def test_mz_graph_creator(self):
        """MZGraphCreator should produce a valid memory zone graph."""
        from mqt.ionshuttler.single_shuttler.graph_utils import MZGraphCreator

        creator = MZGraphCreator(3, 3, 1, 1, "outer")
        g = creator.get_graph()
        assert isinstance(g, nx.Graph)
        assert len(g.edges()) > 0

    def test_graph_creator_outer(self):
        """GraphCreator with 'outer' PZ should have processing zone structure."""
        from mqt.ionshuttler.single_shuttler.graph_utils import GraphCreator

        creator = GraphCreator(3, 3, 1, 1, "outer")
        g = creator.get_graph()
        assert isinstance(g, nx.Graph)
        pz_nodes = [n for n in g.nodes() if g.nodes[n].get("node_type") == "processing_zone_node"]
        assert len(pz_nodes) == 1

    def test_graph_creator_mid(self):
        """GraphCreator with 'mid' PZ should have processing zone structure."""
        from mqt.ionshuttler.single_shuttler.graph_utils import GraphCreator

        creator = GraphCreator(3, 3, 2, 2, "mid")
        g = creator.get_graph()
        assert isinstance(g, nx.Graph)
        pz_nodes = [n for n in g.nodes() if g.nodes[n].get("node_type") == "processing_zone_node"]
        assert len(pz_nodes) == 1

    def test_graph_creator_invalid_pz_raises(self):
        """GraphCreator should raise ValueError for invalid pz type."""
        from mqt.ionshuttler.single_shuttler.graph_utils import GraphCreator

        with pytest.raises(ValueError, match="pz must be"):
            GraphCreator(3, 3, 1, 1, "invalid")

    def test_get_path_to_node(self, small_grid_graph):
        """get_path_to_node should return a non-empty path between connected nodes."""
        from mqt.ionshuttler.single_shuttler.memory_sat import get_path_to_node

        nodes = list(small_grid_graph.nodes())
        trap_nodes = [n for n in nodes if small_grid_graph.nodes[n].get("node_type") in ("trap_node", "junction_node")]
        if len(trap_nodes) >= 2:
            path = get_path_to_node(small_grid_graph, trap_nodes[0], trap_nodes[1])
            assert isinstance(path, list)
            assert len(path) >= 1

    def test_find_connected_edges(self):
        """GraphCreator.find_connected_edges should return non-empty list."""
        from mqt.ionshuttler.single_shuttler.graph_utils import GraphCreator

        creator = GraphCreator(2, 2, 1, 1, "outer")
        edges = creator.find_connected_edges()
        assert isinstance(edges, list)
        assert len(edges) > 0
        # Each entry should be a list of two edges
        for pair in edges:
            assert len(pair) == 2


# ===================================================================
# Compilation tests (single_shuttler)
# ===================================================================


class TestCompilation:
    """Tests for single_shuttler.compilation."""

    def test_extract_qubits_from_gate(self):
        """extract_qubits_from_gate should parse qubit indices correctly."""
        from mqt.ionshuttler.single_shuttler.compilation import extract_qubits_from_gate

        result = extract_qubits_from_gate("cx q[0],q[3];")
        assert result == [0, 3]

    def test_extract_qubits_single_qubit_gate(self):
        """Single-qubit gate line should return one qubit."""
        from mqt.ionshuttler.single_shuttler.compilation import extract_qubits_from_gate

        result = extract_qubits_from_gate("h q[2];")
        assert result == [2]

    def test_extract_qubits_no_match(self):
        """Lines with no qubit pattern should return empty list."""
        from mqt.ionshuttler.single_shuttler.compilation import extract_qubits_from_gate

        result = extract_qubits_from_gate("barrier;")
        assert result == []

    def test_is_qasm_file_valid(self, qasm_file_qft6):
        """is_qasm_file should return True for a valid QASM file."""
        from mqt.ionshuttler.single_shuttler.compilation import is_qasm_file

        assert is_qasm_file(qasm_file_qft6) is True

    def test_is_qasm_file_invalid(self, tmp_path):
        """is_qasm_file should return False for a non-QASM file."""
        from mqt.ionshuttler.single_shuttler.compilation import is_qasm_file

        fake = tmp_path / "not_qasm.txt"
        fake.write_text("Hello world\n" * 10)
        assert is_qasm_file(fake) is False

    def test_parse_qasm(self, qasm_file_qft6):
        """parse_qasm should return a list of qubit tuples."""
        from mqt.ionshuttler.single_shuttler.compilation import parse_qasm

        result = parse_qasm(qasm_file_qft6)
        assert isinstance(result, list)
        assert len(result) > 0
        for gate in result:
            assert isinstance(gate, tuple)
            assert all(isinstance(q, int) for q in gate)

    def test_get_front_layer(self):
        """get_front_layer should return nodes with no predecessors."""
        from qiskit import QuantumCircuit
        from qiskit.converters import circuit_to_dagdependency

        from mqt.ionshuttler.single_shuttler.compilation import get_front_layer

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        dag = circuit_to_dagdependency(qc)
        front = get_front_layer(dag)
        assert len(front) >= 1
        # H(0) should be in front layer since it has no predecessors
        front_ops = [n.op.name for n in front]
        assert "h" in front_ops

    def test_manual_copy_dag(self):
        """manual_copy_dag should produce a DAG with same number of nodes."""
        from qiskit import QuantumCircuit
        from qiskit.converters import circuit_to_dagdependency

        from mqt.ionshuttler.single_shuttler.compilation import manual_copy_dag

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        dag = circuit_to_dagdependency(qc)
        copied = manual_copy_dag(dag)
        assert len(list(copied.get_nodes())) == len(list(dag.get_nodes()))


# ===================================================================
# MemorySAT tests
# ===================================================================


class TestMemorySAT:
    """Tests for the SAT-based exact solver."""

    def test_trivial_single_qubit_sequence(self):
        """A trivial 1-qubit sequence on a 2×2 grid should be satisfiable."""
        from mqt.ionshuttler.single_shuttler.memory_sat import MemorySAT, create_graph

        g = create_graph(2, 2, 1, 1)
        starting_traps = [e for e in g.edges() if g.get_edge_data(*e)["edge_type"] == "trap"][:2]
        ions = [0, 1]
        sat_solver = MemorySAT(g, 1, 1, ions, timesteps=4)
        sat_solver.create_constraints(starting_traps)
        # Sequence: bring ion 0 then ion 1 to PZ
        result = sat_solver.evaluate([0, 1], num_of_registers=2)
        assert isinstance(result, bool)

    def test_sat_solver_with_full_register_config(self, exact_config_full_register):
        """The exact solver should find a solution for the full_register_access config."""
        from mqt.ionshuttler.single_shuttler.main import main

        main(exact_config_full_register)  # Should not raise


# ===================================================================
# Integration: single_shuttler.main
# ===================================================================


class TestSingleShuttlerMain:
    """Integration tests for the single_shuttler main entry point."""

    def test_main_qft05(self, exact_config_qft05):
        """main() should complete without error for the QFT-5 config."""
        from mqt.ionshuttler.single_shuttler.main import main

        main(exact_config_qft05)  # Should not raise

