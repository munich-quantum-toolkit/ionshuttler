"""Tests for single shuttler SAT module."""

from unittest.mock import patch

import networkx as nx
import pytest
from src.mqt.ionshuttler.single_shuttler.SAT import (
    MemorySAT,
    create_graph,
    create_idc_dictionary,
    get_idc_from_idx,
    get_idx_from_idc,
    get_junctions,
    get_path_between_edges,
    get_path_to_node,
    get_possible_moves_over_junction,
    get_possible_moves_through_node,
    get_possible_previous_edges_from_junction_move,
)


class TestCreateGraph:
    """Test the create_graph function."""

    def test_create_graph_basic(self):
        """Test creating a basic graph."""
        m, n = 3, 3
        ion_chain_size_vertical = 2
        ion_chain_size_horizontal = 2

        graph = create_graph(m, n, ion_chain_size_vertical, ion_chain_size_horizontal)

        assert isinstance(graph, nx.Graph)
        assert len(graph.nodes()) > 0
        assert len(graph.edges()) > 0

        # Check that processing zone node exists
        pz_nodes = [node for node in graph.nodes() if graph.nodes[node].get("node_type") == "processing_zone_node"]
        assert len(pz_nodes) == 1

        # Check that junction nodes exist
        junction_nodes = [node for node in graph.nodes() if graph.nodes[node].get("node_type") == "junction_node"]
        assert len(junction_nodes) > 0

    def test_create_graph_edge_types(self):
        """Test that graph has correct edge types."""
        m, n = 2, 2
        ion_chain_size_vertical = 1
        ion_chain_size_horizontal = 1

        graph = create_graph(m, n, ion_chain_size_vertical, ion_chain_size_horizontal)

        edge_types = set()
        for edge in graph.edges():
            edge_type = graph.get_edge_data(edge[0], edge[1]).get("edge_type")
            if edge_type:
                edge_types.add(edge_type)

        expected_types = {"trap", "exit", "entry"}
        assert expected_types.issubset(edge_types)


class TestIdcDictionary:
    """Test IDC dictionary functions."""

    def test_create_idc_dictionary(self):
        """Test creating IDC dictionary."""
        graph = nx.Graph()
        graph.add_edge((0, 0), (0, 1))
        graph.add_edge((0, 1), (1, 1))

        idc_dict = create_idc_dictionary(graph)

        assert isinstance(idc_dict, dict)
        assert len(idc_dict) == 2
        assert all(isinstance(key, int) for key in idc_dict)
        assert all(isinstance(value, tuple) for value in idc_dict.values())

    def test_get_idx_from_idc(self):
        """Test getting index from IDC."""
        edge_dict = {0: ((0, 0), (0, 1)), 1: ((0, 1), (1, 1))}

        idx = get_idx_from_idc(edge_dict, ((0, 0), (0, 1)))
        assert idx == 0

        # Test with reversed order
        idx = get_idx_from_idc(edge_dict, ((0, 1), (0, 0)))
        assert idx == 0

    def test_get_idc_from_idx(self):
        """Test getting IDC from index."""
        edge_dict = {0: ((0, 0), (0, 1)), 1: ((0, 1), (1, 1))}

        idc = get_idc_from_idx(edge_dict, 0)
        assert idc == ((0, 0), (0, 1))


class TestGetPathToNode:
    """Test the get_path_to_node function."""

    def test_get_path_to_node_basic(self):
        """Test getting path between two nodes."""
        graph = nx.Graph()
        graph.add_edge((0, 0), (0, 1), edge_type="trap")
        graph.add_edge((0, 1), (0, 2), edge_type="trap")
        graph.add_edge((0, 2), (0, 3), edge_type="trap")

        path = get_path_to_node(graph, (0, 0), (0, 3))

        expected_path = [((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3))]
        assert path == expected_path

    def test_get_path_to_node_single_edge(self):
        """Test path with single edge."""
        graph = nx.Graph()
        graph.add_edge((0, 0), (0, 1), edge_type="trap")

        path = get_path_to_node(graph, (0, 0), (0, 1))

        expected_path = [((0, 0), (0, 1))]
        assert path == expected_path


class TestGetJunctions:
    """Test the get_junctions function."""

    def test_get_junctions_trap_node(self):
        """Test getting junctions for a trap node."""
        graph = nx.Graph()
        graph.add_node((0, 0), node_type="junction_node")
        graph.add_node((0, 1), node_type="trap_node")
        graph.add_node((0, 2), node_type="junction_node")
        graph.add_edge((0, 0), (0, 1))
        graph.add_edge((0, 1), (0, 2))

        junctions = get_junctions(graph, (0, 1), (0, 0), 1, 1)

        assert (0, 0) in junctions
        assert (0, 2) in junctions

    def test_get_junctions_between_junctions(self):
        """Test getting junctions between two junction nodes."""
        graph = nx.Graph()
        graph.add_node((0, 0), node_type="junction_node")
        graph.add_node((0, 1), node_type="junction_node")

        junctions = get_junctions(graph, (0, 0), (0, 1), 1, 1)

        expected = [(0, 0), (0, 1)]
        assert junctions == expected


class TestGetPossibleMovesOverJunction:
    """Test the get_possible_moves_over_junction function."""

    def test_get_possible_moves_over_junction_basic(self):
        """Test getting possible moves over junction."""
        graph = nx.Graph()
        graph.add_node((0, 0), node_type="junction_node")
        graph.add_node((0, 1), node_type="trap_node")
        graph.add_node((1, 0), node_type="trap_node")
        graph.add_node((0, -1), node_type="trap_node")

        graph.add_edge((0, 0), (0, 1))
        graph.add_edge((0, 0), (1, 0))
        graph.add_edge((0, 0), (0, -1))

        edge = ((0, 1), (0, 0))
        possible_moves = get_possible_moves_over_junction(graph, edge, 1, 1)

        assert isinstance(possible_moves, list)
        # Should include edges connected to the junction except the input edge
        assert len(possible_moves) >= 2


class TestGetPathBetweenEdges:
    """Test the get_path_between_edges function."""

    def test_get_path_between_edges_basic(self):
        """Test getting path between two edges."""
        graph = nx.Graph()
        graph.add_node((0, 0), node_type="junction_node")
        graph.add_node((0, 1), node_type="trap_node")
        graph.add_node((0, 2), node_type="trap_node")
        graph.add_node((0, 3), node_type="junction_node")

        graph.add_edge((0, 0), (0, 1))
        graph.add_edge((0, 1), (0, 2))
        graph.add_edge((0, 2), (0, 3))

        src_edge = ((0, 0), (0, 1))
        tar_edge = ((0, 2), (0, 3))

        path = get_path_between_edges(graph, src_edge, tar_edge)

        # Should return the edge(s) between the source and target edges
        assert isinstance(path, list)


class TestMemorySAT:
    """Test the MemorySAT class."""

    def test_memory_sat_initialization(self):
        """Test MemorySAT initialization."""
        # Create a simple graph
        graph = create_graph(2, 2, 1, 1)
        ions = [0, 1]
        timesteps = 5

        memory_sat = MemorySAT(graph, 1, 1, ions, timesteps)

        assert memory_sat.graph == graph
        assert memory_sat.ion_chain_size_horizontal == 1
        assert memory_sat.ion_chain_size_vertical == 1
        assert memory_sat.ions == ions
        assert memory_sat.timesteps == timesteps
        assert hasattr(memory_sat, "idc_dict")
        assert hasattr(memory_sat, "entry")
        assert hasattr(memory_sat, "exit")
        assert hasattr(memory_sat, "s")  # Z3 solver
        assert hasattr(memory_sat, "states")

    def test_memory_sat_create_constraints(self):
        """Test creating constraints in MemorySAT."""
        graph = create_graph(2, 2, 1, 1)
        ions = [0, 1]
        timesteps = 3

        memory_sat = MemorySAT(graph, 1, 1, ions, timesteps)

        # Find valid starting traps
        starting_traps = []
        for edge in graph.edges():
            if graph.get_edge_data(edge[0], edge[1]).get("edge_type") == "trap":
                starting_traps.append(edge)
                if len(starting_traps) >= len(ions):
                    break

        starting_traps = starting_traps[: len(ions)]

        # Should not raise an exception
        memory_sat.create_constraints(starting_traps)

        assert hasattr(memory_sat, "starting_traps")
        assert memory_sat.starting_traps == starting_traps

    def test_memory_sat_evaluate_basic_sequence(self):
        """Test evaluating a basic sequence."""
        graph = create_graph(2, 2, 1, 1)
        ions = [0, 1]
        timesteps = 10

        memory_sat = MemorySAT(graph, 1, 1, ions, timesteps)

        # Find valid starting traps
        starting_traps = []
        for edge in graph.edges():
            if graph.get_edge_data(edge[0], edge[1]).get("edge_type") == "trap":
                starting_traps.append(edge)
                if len(starting_traps) >= len(ions):
                    break

        starting_traps = starting_traps[: len(ions)]
        memory_sat.create_constraints(starting_traps)

        # Simple sequence
        sequence = [(0,), (1,)]
        num_of_registers = 2

        # Should not raise an exception
        result = memory_sat.evaluate(sequence, num_of_registers)
        assert isinstance(result, bool)

    def test_memory_sat_evaluate_invalid_sequence(self):
        """Test evaluating with invalid sequence parameters."""
        graph = create_graph(2, 2, 1, 1)
        ions = [0, 1]
        timesteps = 5

        memory_sat = MemorySAT(graph, 1, 1, ions, timesteps)

        starting_traps = []
        for edge in graph.edges():
            if graph.get_edge_data(edge[0], edge[1]).get("edge_type") == "trap":
                starting_traps.append(edge)
                if len(starting_traps) >= len(ions):
                    break

        starting_traps = starting_traps[: len(ions)]
        memory_sat.create_constraints(starting_traps)

        # Test with invalid sequence (ion index too high)
        sequence = [(5,)]  # Ion 5 doesn't exist
        num_of_registers = 2

        with pytest.raises(AssertionError):
            memory_sat.evaluate(sequence, num_of_registers)

        # Test with empty sequence
        with pytest.raises(AssertionError):
            memory_sat.evaluate([], num_of_registers)

    @patch("matplotlib.pyplot.show")
    def test_memory_sat_plot(self, mock_show):
        """Test plotting functionality."""
        graph = create_graph(2, 2, 1, 1)
        ions = [0]
        timesteps = 3

        memory_sat = MemorySAT(graph, 1, 1, ions, timesteps)
        memory_sat.check = "unsat"  # Mock unsat result

        # Should not raise an exception
        memory_sat.plot(show_ions=False)
        mock_show.assert_called_once()


class TestGetPossibleMovesThoughNode:
    """Test the get_possible_moves_through_node function."""

    def test_get_possible_moves_through_node_basic(self):
        """Test getting possible moves through a node."""
        graph = nx.Graph()
        graph.add_edge((0, 0), (0, 1))
        graph.add_edge((0, 0), (1, 0))
        graph.add_edge((0, 0), (0, -1))

        idc_dict = create_idc_dictionary(graph)
        node = (0, 0)

        moves = get_possible_moves_through_node(graph, idc_dict, node)

        assert isinstance(moves, list)
        # Should have permutations of connected edges
        assert len(moves) > 0
        # Each move should be a tuple of two edge indices
        for move in moves:
            assert isinstance(move, tuple)
            assert len(move) == 2


class TestGetPossiblePreviousEdgesFromJunctionMove:
    """Test the get_possible_previous_edges_from_junction_move function."""

    def test_get_possible_previous_edges_basic(self):
        """Test getting possible previous edges from junction move."""
        graph = nx.Graph()
        graph.add_node((0, 0), node_type="junction_node")
        graph.add_node((0, 1), node_type="trap_node")
        graph.add_node((1, 0), node_type="trap_node")
        graph.add_node((0, 2), node_type="junction_node")

        graph.add_edge((0, 0), (0, 1))
        graph.add_edge((0, 0), (1, 0))
        graph.add_edge((0, 1), (0, 2))

        edge = ((0, 0), (0, 1))  # Junction edge

        possible_edges = get_possible_previous_edges_from_junction_move(graph, edge, 1, 1)

        assert isinstance(possible_edges, list)

    def test_get_possible_previous_edges_with_entry(self):
        """Test with entry edge type."""
        graph = nx.Graph()
        graph.add_node((0, 0), node_type="junction_node")
        graph.add_node((0, 1), node_type="processing_zone_node")

        graph.add_edge((0, 0), (0, 1), edge_type="entry")

        edge = ((0, 0), (0, 1))

        possible_edges = get_possible_previous_edges_from_junction_move(graph, edge, 1, 1)

        assert isinstance(possible_edges, list)
        # Should include the entry edge itself
        assert edge in possible_edges or tuple(reversed(edge)) in possible_edges
