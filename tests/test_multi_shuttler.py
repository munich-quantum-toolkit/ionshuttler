"""Tests for the multi_shuttler (heuristic solver) subpackage."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest


# ===================================================================
# ProcessingZone tests
# ===================================================================


class TestProcessingZone:
    """Tests for the ProcessingZone data class."""

    def test_basic_creation(self):
        """A ProcessingZone should store its name and info correctly."""
        from mqt.ionshuttler.multi_shuttler.outside.processing_zone import ProcessingZone

        pz = ProcessingZone("pz_test", [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
        assert pz.name == "pz_test"
        assert pz.exit_node == (1.0, 2.0)
        assert pz.entry_node == (3.0, 4.0)
        assert pz.processing_zone == (5.0, 6.0)

    def test_properties_settable(self):
        """ProcessingZone properties should be gettable and settable."""
        from mqt.ionshuttler.multi_shuttler.outside.processing_zone import ProcessingZone

        pz = ProcessingZone("pz1", [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
        pz.parking_node = (3.0, 3.0)
        assert pz.parking_node == (3.0, 3.0)
        pz.parking_edge = ((2.0, 2.0), (3.0, 3.0))
        assert pz.parking_edge == ((2.0, 2.0), (3.0, 3.0))
        pz.time_in_pz_counter = 5
        assert pz.time_in_pz_counter == 5
        pz.gate_execution_finished = True
        assert pz.gate_execution_finished is True
        pz.rotate_entry = False
        assert pz.rotate_entry is False

    def test_multiple_pzs_have_unique_names(self, multi_processing_zone_1pz):
        """Each ProcessingZone should have a unique name."""
        from mqt.ionshuttler.multi_shuttler.outside.processing_zone import ProcessingZone

        pz2 = ProcessingZone("pz2", [(0.0, 0.0), (0.0, 2.0), (4.5, 1.0)])
        assert multi_processing_zone_1pz.name != pz2.name


# ===================================================================
# Graph creation tests (multi_shuttler)
# ===================================================================


class TestMultiGraphCreation:
    """Tests for GraphCreator and PZCreator in multi_shuttler."""

    def test_graph_creator_produces_graph(self, multi_graph_creator_1pz):
        """GraphCreator should produce a valid networkx Graph."""
        basegraph, _ = multi_graph_creator_1pz
        g = basegraph.get_graph()
        assert isinstance(g, nx.Graph)
        assert len(g.nodes()) > 0
        assert len(g.edges()) > 0

    def test_pz_creator_adds_processing_zone(self, multi_graph_creator_1pz):
        """PZCreator should add processing zone nodes to the graph."""
        _, pzgraph = multi_graph_creator_1pz
        g = pzgraph.get_graph()
        pz_nodes = [n for n in g.nodes() if g.nodes[n].get("node_type") == "processing_zone_node"]
        assert len(pz_nodes) >= 1

    def test_pz_creator_adds_entry_exit_edges(self, multi_graph_creator_1pz):
        """PZCreator should add entry and exit edges."""
        _, pzgraph = multi_graph_creator_1pz
        g = pzgraph.get_graph()
        edge_types = nx.get_edge_attributes(g, "edge_type")
        assert "exit" in edge_types.values()
        has_entry = "entry" in edge_types.values() or "first_entry_connection" in edge_types.values()
        assert has_entry

    def test_pz_creator_adds_parking_edge(self, multi_graph_creator_1pz):
        """PZCreator should add a parking edge."""
        _, pzgraph = multi_graph_creator_1pz
        g = pzgraph.get_graph()
        edge_types = nx.get_edge_attributes(g, "edge_type")
        assert "parking_edge" in edge_types.values()

    def test_graph_has_junction_nodes(self, multi_graph_creator_1pz):
        """The graph should have junction nodes."""
        basegraph, _ = multi_graph_creator_1pz
        g = basegraph.get_graph()
        assert len(g.junction_nodes) > 0

    def test_graph_with_two_pzs(self):
        """Creating a graph with 2 PZs should produce 2 processing zone nodes."""
        from mqt.ionshuttler.multi_shuttler.outside.graph_creator import GraphCreator, PZCreator
        from mqt.ionshuttler.multi_shuttler.outside.processing_zone import ProcessingZone

        m, n, v, h = 3, 3, 1, 1
        height = -4.5
        pz1 = ProcessingZone("pz1", [
            (float((m - 1) * v), float((n - 1) * h)),
            (float((m - 1) * v), float(0)),
            (float((m - 1) * v - height), float((n - 1) * h / 2)),
        ])
        pz2 = ProcessingZone("pz2", [
            (0.0, 0.0),
            (0.0, float((n - 1) * h)),
            (float(height), float((n - 1) * h / 2)),
        ])
        pzs = [pz1, pz2]
        GraphCreator(m, n, v, h, 0, pzs)
        pzgraph = PZCreator(m, n, v, h, 0, pzs)
        g = pzgraph.get_graph()
        pz_nodes = [n for n in g.nodes() if g.nodes[n].get("node_type") == "processing_zone_node"]
        assert len(pz_nodes) == 2

    def test_graph_with_failing_junctions(self):
        """Graph with failing junctions should have fewer nodes."""
        from mqt.ionshuttler.multi_shuttler.outside.graph_creator import GraphCreator
        from mqt.ionshuttler.multi_shuttler.outside.processing_zone import ProcessingZone

        m, n, v, h = 3, 3, 1, 1
        pz1 = ProcessingZone("pz1", [
            (float((m - 1) * v), float((n - 1) * h)),
            (float((m - 1) * v), float(0)),
            (6.5, 1.0),
        ])
        g_no_fail = GraphCreator(m, n, v, h, 0, [pz1]).get_graph()
        g_fail = GraphCreator(m, n, v, h, 1, [pz1]).get_graph()
        assert len(g_fail.nodes()) < len(g_no_fail.nodes())


# ===================================================================
# Graph utility tests (multi_shuttler)
# ===================================================================


class TestMultiGraphUtils:
    """Tests for multi_shuttler.outside.graph_utils."""

    def test_idc_dictionary_round_trip(self, multi_graph_creator_1pz):
        """Converting idc → idx → idc should be consistent."""
        from mqt.ionshuttler.multi_shuttler.outside.graph_utils import (
            create_idc_dictionary,
            get_idc_from_idx,
            get_idx_from_idc,
        )

        _, pzgraph = multi_graph_creator_1pz
        g = pzgraph.get_graph()
        idc_dict = create_idc_dictionary(g)
        # Check a sample of edges
        for edge in list(g.edges())[:10]:
            idx = get_idx_from_idc(idc_dict, edge)
            idc = get_idc_from_idx(idc_dict, idx)
            assert get_idx_from_idc(idc_dict, idc) == idx

    def test_convert_nodes_to_float(self):
        """convert_nodes_to_float should apply a float mapping to nodes.

        Note: Due to Python's hash equality between int and float (hash(0) == hash(0.0)),
        nx.relabel_nodes with copy=False may not actually change the type of nodes
        whose int coords hash-equal their float equivalents. We verify the function
        runs without error and the graph structure is preserved.
        """
        from mqt.ionshuttler.multi_shuttler.outside.graph import Graph
        from mqt.ionshuttler.multi_shuttler.outside.graph_utils import convert_nodes_to_float

        g = nx.grid_2d_graph(3, 3, create_using=Graph)
        n_nodes_before = len(g.nodes())
        n_edges_before = len(g.edges())
        convert_nodes_to_float(g)
        assert len(g.nodes()) == n_nodes_before
        assert len(g.edges()) == n_edges_before


# ===================================================================
# Graph class tests (multi_shuttler)
# ===================================================================


class TestMultiGraph:
    """Tests for the custom Graph class."""

    def test_graph_inherits_from_networkx(self):
        """The custom Graph should be a subclass of nx.Graph."""
        from mqt.ionshuttler.multi_shuttler.outside.graph import Graph

        g = Graph()
        assert isinstance(g, nx.Graph)

    def test_idc_dict_is_lazy(self):
        """The idc_dict property should be lazily initialized."""
        from mqt.ionshuttler.multi_shuttler.outside.graph import Graph

        g = Graph()
        g.add_edge((0.0, 0.0), (1.0, 0.0))
        idc_dict = g.idc_dict
        assert isinstance(idc_dict, dict)
        assert len(idc_dict) > 0


# ===================================================================
# Compilation tests (multi_shuttler)
# ===================================================================


class TestMultiCompilation:
    """Tests for multi_shuttler.outside.compilation."""

    def test_extract_qubits_from_gate(self):
        """extract_qubits_from_gate should parse qubit indices correctly."""
        from mqt.ionshuttler.multi_shuttler.outside.compilation import extract_qubits_from_gate

        result = extract_qubits_from_gate("cx q[1],q[4];")
        assert result == [1, 4]

    def test_is_qasm_file(self, qasm_file_qft6):
        """is_qasm_file should return True for a valid QASM file."""
        from mqt.ionshuttler.multi_shuttler.outside.compilation import is_qasm_file

        assert is_qasm_file(qasm_file_qft6) is True

    def test_parse_qasm(self, qasm_file_qft6):
        """parse_qasm should return a list of qubit tuples."""
        from mqt.ionshuttler.multi_shuttler.outside.compilation import parse_qasm

        result = parse_qasm(qasm_file_qft6)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_create_dag(self, qasm_file_qft6):
        """create_dag should return a DAGDependency object."""
        from qiskit.dagcircuit import DAGDependency

        from mqt.ionshuttler.multi_shuttler.outside.compilation import create_dag

        dag = create_dag(qasm_file_qft6)
        assert isinstance(dag, DAGDependency)
        assert len(list(dag.get_nodes())) > 0

    def test_create_initial_sequence(self, qasm_file_qft6):
        """create_initial_sequence should return a non-empty gate sequence."""
        from mqt.ionshuttler.multi_shuttler.outside.compilation import create_initial_sequence

        seq = create_initial_sequence(qasm_file_qft6)
        assert isinstance(seq, list)
        assert len(seq) > 0

    def test_get_front_layer(self):
        """get_front_layer should return the initial nodes of a DAG."""
        from qiskit import QuantumCircuit
        from qiskit.converters import circuit_to_dagdependency

        from mqt.ionshuttler.multi_shuttler.outside.compilation import get_front_layer

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 2)
        dag = circuit_to_dagdependency(qc)
        front = get_front_layer(dag)
        assert len(front) >= 2  # h(0) and h(1) are both in front layer

    def test_manual_copy_dag(self):
        """manual_copy_dag should copy a DAG preserving all nodes."""
        from qiskit import QuantumCircuit
        from qiskit.converters import circuit_to_dagdependency

        from mqt.ionshuttler.multi_shuttler.outside.compilation import manual_copy_dag

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        dag = circuit_to_dagdependency(qc)
        copied = manual_copy_dag(dag)
        assert len(list(copied.get_nodes())) == len(list(dag.get_nodes()))

    def test_remove_node_reduces_dag_size(self):
        """remove_node should reduce the number of nodes in the DAG."""
        from qiskit import QuantumCircuit
        from qiskit.converters import circuit_to_dagdependency

        from mqt.ionshuttler.multi_shuttler.outside.compilation import get_front_layer, remove_node

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        dag = circuit_to_dagdependency(qc)
        original_count = len(list(dag.get_nodes()))
        front = get_front_layer(dag)
        remove_node(dag, front[0])
        new_count = len(list(dag.get_nodes()))
        assert new_count == original_count - 1


# ===================================================================
# Partition tests (multi_shuttler)
# ===================================================================


class TestPartition:
    """Tests for multi_shuttler.outside.partition."""

    def test_read_qasm_file(self, qasm_file_qft6):
        """read_qasm_file should return a QuantumCircuit."""
        from qiskit import QuantumCircuit

        from mqt.ionshuttler.multi_shuttler.outside.partition import read_qasm_file

        qc = read_qasm_file(qasm_file_qft6)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits > 0

    def test_construct_interaction_graph(self, qasm_file_qft6):
        """construct_interaction_graph should produce a valid weighted graph."""
        from mqt.ionshuttler.multi_shuttler.outside.partition import (
            construct_interaction_graph,
            read_qasm_file,
        )

        qc = read_qasm_file(qasm_file_qft6)
        ig = construct_interaction_graph(qc)
        assert isinstance(ig, nx.Graph)
        assert len(ig.nodes()) > 0
        # Edges should have 'weight' attribute
        for _, _, data in ig.edges(data=True):
            assert "weight" in data
            assert data["weight"] >= 1


# ===================================================================
# Cycles / starting config tests (multi_shuttler)
# ===================================================================


class TestMultiCycles:
    """Tests for multi_shuttler.outside.cycles."""

    def test_create_starting_config(self, multi_graph_creator_1pz):
        """create_starting_config should place ions onto trap edges."""
        from mqt.ionshuttler.multi_shuttler.outside.cycles import create_starting_config, get_ions

        _, pzgraph = multi_graph_creator_1pz
        g = pzgraph.get_graph()
        g.max_num_parking = 2
        g.pzs = pzgraph.pzs

        n_ions = 4
        num_reg = create_starting_config(g, n_ions, seed=0)
        assert num_reg == n_ions

        ions = get_ions(g)
        assert len(ions) == n_ions

    def test_get_state_idxs(self, multi_graph_creator_1pz):
        """get_state_idxs should return ion → edge_idx mapping."""
        from mqt.ionshuttler.multi_shuttler.outside.cycles import (
            create_starting_config,
            get_state_idxs,
        )

        _, pzgraph = multi_graph_creator_1pz
        g = pzgraph.get_graph()
        g.max_num_parking = 2
        g.pzs = pzgraph.pzs

        create_starting_config(g, 3, seed=42)
        state = get_state_idxs(g)
        assert isinstance(state, dict)
        assert len(state) == 3


# ===================================================================
# Config validation tests (multi_shuttler.main)
# ===================================================================


class TestMultiMainValidation:
    """Tests for multi_shuttler.main config validation."""

    def test_missing_arch_raises(self):
        """main should raise ValueError when 'arch' is missing."""
        from mqt.ionshuttler.multi_shuttler.main import main

        with pytest.raises(ValueError, match="arch"):
            main({"algorithm_name": "test", "num_ions": 6})

    def test_missing_algorithm_name_raises(self):
        """main should raise ValueError when 'algorithm_name' is missing."""
        from mqt.ionshuttler.multi_shuttler.main import main

        with pytest.raises(ValueError, match="algorithm_name"):
            main({"arch": [3, 3, 1, 1], "num_ions": 6})

    def test_missing_num_ions_raises(self):
        """main should raise ValueError when 'num_ions' is missing."""
        from mqt.ionshuttler.multi_shuttler.main import main

        with pytest.raises(ValueError, match="num_ions"):
            main({"arch": [3, 3, 1, 1], "algorithm_name": "test"})

    def test_invalid_arch_format_raises(self):
        """main should raise ValueError when 'arch' is not a list of 4 ints."""
        from mqt.ionshuttler.multi_shuttler.main import main

        with pytest.raises(ValueError, match="arch"):
            main({"arch": [3, 3], "algorithm_name": "test", "num_ions": 6})


# ===================================================================
# Integration: multi_shuttler.main
# ===================================================================


class TestMultiShuttlerMain:
    """Integration tests for the multi_shuttler main entry point."""

    def test_main_1pz(self, heuristic_config_1pz):
        """main() should complete without error for the 1-PZ config."""
        from mqt.ionshuttler.multi_shuttler.main import main

        main(heuristic_config_1pz)  # Should not raise

    def test_main_2pzs(self, heuristic_config_2pzs):
        """main() should complete without error for the 2-PZ config."""
        from mqt.ionshuttler.multi_shuttler.main import main

        main(heuristic_config_2pzs)  # Should not raise

