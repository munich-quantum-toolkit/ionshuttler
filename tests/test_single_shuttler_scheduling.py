"""Tests for single shuttler scheduling module."""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import random
import networkx as nx
from pathlib import Path

from src.mqt.ionshuttler.single_shuttler.scheduling import (
    create_starting_config,
    preprocess,
    create_move_list,
    create_initial_sequence,
    create_circles_for_moves,
    find_movable_circles,
    rotate_free_circles,
    check_duplicates,
)


class TestCreateStartingConfig:
    """Test the create_starting_config function."""
    
    def test_create_starting_config_with_seed(self):
        """Test creating starting configuration with seed."""
        # Create mock graph
        mock_graph = MagicMock()
        mock_graph.edges.return_value = [
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((2, 0), (2, 1)),
        ]
        mock_graph.get_edge_data.return_value = {"edge_type": "trap"}
        
        n_of_chains = 2
        seed = 42
        
        ion_chains, number_of_registers = create_starting_config(n_of_chains, mock_graph, seed)
        
        assert len(ion_chains) == n_of_chains
        assert number_of_registers == n_of_chains
        assert all(isinstance(key, int) for key in ion_chains.keys())
        assert all(isinstance(value, tuple) for value in ion_chains.values())
    
    def test_create_starting_config_without_seed(self):
        """Test creating starting configuration without seed."""
        mock_graph = MagicMock()
        mock_graph.edges.return_value = [
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((2, 0), (2, 1)),
        ]
        mock_graph.get_edge_data.return_value = {"edge_type": "trap"}
        
        n_of_chains = 2
        
        ion_chains, number_of_registers = create_starting_config(n_of_chains, mock_graph)
        
        assert len(ion_chains) == n_of_chains
        assert number_of_registers == n_of_chains
    
    def test_create_starting_config_more_chains_than_traps(self):
        """Test creating starting configuration with more chains than available traps."""
        mock_graph = MagicMock()
        mock_graph.edges.return_value = [
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
        ]
        mock_graph.get_edge_data.return_value = {"edge_type": "trap"}
        
        n_of_chains = 5  # More than available traps
        seed = 42
        
        with pytest.raises(ValueError):
            create_starting_config(n_of_chains, mock_graph, seed)


class TestPreprocess:
    """Test the preprocess function."""
    
    def test_preprocess_basic(self):
        """Test basic preprocessing functionality."""
        # Create mock memorygrid
        mock_memorygrid = MagicMock()
        mock_memorygrid.ion_chains = {0: ((0, 0), (0, 1)), 1: ((1, 0), (1, 1))}
        mock_memorygrid.find_next_edge.return_value = ((0, 1), (0, 2))
        mock_memorygrid.get_state_idxs.return_value = [0, 1]
        mock_memorygrid.have_common_junction_node.return_value = False
        mock_memorygrid.idc_dict = {0: ((0, 0), (0, 1)), 1: ((1, 0), (1, 1))}
        
        sequence = [0, 1]
        
        result = preprocess(mock_memorygrid, sequence)
        
        assert result == mock_memorygrid
        assert mock_memorygrid.find_next_edge.called


class TestCreateMoveList:
    """Test the create_move_list function."""
    
    def test_create_move_list_basic(self):
        """Test creating move list with basic sequence."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.ion_chains = {0: ((0, 0), (0, 1)), 1: ((1, 0), (1, 1))}
        mock_memorygrid.graph = MagicMock()
        mock_memorygrid.graph_creator = MagicMock()
        mock_memorygrid.graph_creator.processing_zone = (2, 2)
        mock_memorygrid.graph_creator.path_to_pz_idxs = []
        mock_memorygrid.graph_creator.path_from_pz_idxs = []
        mock_memorygrid.get_state_idxs.return_value = [0, 1]
        
        # Mock nx.shortest_path
        with patch('networkx.shortest_path', return_value=[(0, 0), (1, 1), (2, 2)]):
            sequence = [0, 1, 0]
            max_length = 10
            
            result = create_move_list(mock_memorygrid, sequence, max_length)
            
            assert isinstance(result, list)
            assert len(result) <= max_length
    
    def test_create_move_list_with_max_length_limit(self):
        """Test creating move list with max length limit."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.ion_chains = {i: ((i, 0), (i, 1)) for i in range(20)}
        mock_memorygrid.graph = MagicMock()
        mock_memorygrid.graph_creator = MagicMock()
        mock_memorygrid.graph_creator.processing_zone = (2, 2)
        mock_memorygrid.graph_creator.path_to_pz_idxs = []
        mock_memorygrid.graph_creator.path_from_pz_idxs = []
        mock_memorygrid.get_state_idxs.return_value = list(range(20))
        
        with patch('networkx.shortest_path', return_value=[(0, 0), (1, 1), (2, 2)]):
            sequence = list(range(20))  # Long sequence
            max_length = 5
            
            result = create_move_list(mock_memorygrid, sequence, max_length)
            
            assert len(result) <= max_length


class TestCreateInitialSequence:
    """Test the create_initial_sequence function."""
    
    def test_create_initial_sequence_without_compilation(self):
        """Test creating initial sequence without compilation."""
        distance_map = {0: 1, 1: 2}
        filename = "test.qasm"
        compilation = False
        
        mock_qasm_content = [(0,), (0, 1), (1,)]
        
        with patch('src.mqt.ionshuttler.single_shuttler.scheduling.is_qasm_file', return_value=True), \
             patch('src.mqt.ionshuttler.single_shuttler.scheduling.parse_qasm', return_value=mock_qasm_content):
            
            seq, flat_seq, dag_dep, next_node = create_initial_sequence(distance_map, filename, compilation)
            
            assert seq == mock_qasm_content
            assert flat_seq == [0, 0, 1, 1]
            assert dag_dep is None
            assert next_node is None
    
    def test_create_initial_sequence_with_compilation(self):
        """Test creating initial sequence with compilation."""
        distance_map = {0: 1, 1: 2}
        filename = "test.qasm"
        compilation = True
        
        mock_qasm_content = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0], q[1];"
        
        with patch('src.mqt.ionshuttler.single_shuttler.scheduling.is_qasm_file', return_value=True), \
             patch('builtins.open', mock_open(read_data=mock_qasm_content)), \
             patch('src.mqt.ionshuttler.single_shuttler.scheduling.update_sequence') as mock_update:
            
            mock_update.return_value = ([[0], [0, 1]], MagicMock())
            
            seq, flat_seq, dag_dep, next_node = create_initial_sequence(distance_map, filename, compilation)
            
            assert isinstance(seq, list)
            assert isinstance(flat_seq, list)
            assert dag_dep is not None
            assert next_node is not None
    
    def test_create_initial_sequence_invalid_qasm(self):
        """Test creating initial sequence with invalid QASM file."""
        distance_map = {0: 1, 1: 2}
        filename = "invalid.qasm"
        compilation = False
        
        with patch('src.mqt.ionshuttler.single_shuttler.scheduling.is_qasm_file', return_value=False):
            with pytest.raises(AssertionError, match="The file is not a valid QASM file"):
                create_initial_sequence(distance_map, filename, compilation)


class TestCreateCirclesForMoves:
    """Test the create_circles_for_moves function."""
    
    def test_create_circles_for_moves_basic(self):
        """Test creating circles for moves with basic setup."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.ion_chains = {0: ((0, 0), (0, 1)), 1: ((1, 0), (1, 1))}
        mock_memorygrid.find_chain_in_edge.return_value = None
        mock_memorygrid.count_chains_in_parking.return_value = 0
        mock_memorygrid.max_num_parking = 2
        mock_memorygrid.find_next_edge.return_value = ((0, 1), (0, 2))
        mock_memorygrid.find_ordered_edges.return_value = (((0, 0), (0, 1)), ((0, 1), (0, 2)))
        mock_memorygrid.check_if_edge_is_filled.return_value = False
        mock_memorygrid.idc_dict = {0: ((0, 0), (0, 1)), 1: ((1, 0), (1, 1))}
        mock_memorygrid.graph_creator = MagicMock()
        mock_memorygrid.graph_creator.path_to_pz_idxs = []
        mock_memorygrid.graph_creator.path_from_pz_idxs = []
        mock_memorygrid.state_idxs = [0, 1]
        
        move_list = [0, 1]
        flat_seq = [0, 1, 0]
        gate_execution_finished = True
        new_gate_starting = False
        
        result = create_circles_for_moves(
            mock_memorygrid, move_list, flat_seq, gate_execution_finished, new_gate_starting
        )
        
        all_circles, rotate_entry, chain_to_move_out_of_pz = result
        
        assert isinstance(all_circles, dict)
        assert isinstance(rotate_entry, bool)


class TestFindMovableCircles:
    """Test the find_movable_circles function."""
    
    def test_find_movable_circles_basic(self):
        """Test finding movable circles."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.find_nonfree_and_free_circle_idxs.return_value = []
        
        all_circles = {0: [((0, 0), (0, 1)), ((0, 1), (0, 2))], 1: [((1, 0), (1, 1)), ((1, 1), (1, 2))]}
        move_list = [0, 1]
        
        result = find_movable_circles(mock_memorygrid, all_circles, move_list)
        
        assert isinstance(result, list)
        assert len(result) <= len(move_list)


class TestRotateFreeCircles:
    """Test the rotate_free_circles function."""
    
    def test_rotate_free_circles_basic(self):
        """Test rotating free circles."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.idc_dict = {0: ((0, 0), (0, 1)), 1: ((0, 1), (0, 2))}
        mock_memorygrid.rotate.return_value = {}
        mock_memorygrid.graph_creator = MagicMock()
        mock_memorygrid.graph_creator.path_from_pz = [((2, 0), (2, 1))]
        
        all_circles = {0: [((0, 0), (0, 1)), ((0, 1), (0, 2))]}
        free_circle_seq_idxs = [0]
        rotate_entry = False
        chain_to_move_out_of_pz = None
        
        # Should not raise any exceptions
        rotate_free_circles(mock_memorygrid, all_circles, free_circle_seq_idxs, rotate_entry, chain_to_move_out_of_pz)
        
        assert mock_memorygrid.rotate.called
    
    def test_rotate_free_circles_with_entry_rotation(self):
        """Test rotating free circles with entry rotation."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.idc_dict = {0: ((0, 0), (0, 1)), 1: ((0, 1), (0, 2))}
        mock_memorygrid.rotate.return_value = {}
        mock_memorygrid.graph_creator = MagicMock()
        mock_memorygrid.graph_creator.path_from_pz = [((2, 0), (2, 1))]
        
        all_circles = {0: [((0, 0), (0, 1)), ((0, 1), (0, 2))]}
        free_circle_seq_idxs = [0]
        rotate_entry = True
        chain_to_move_out_of_pz = 1
        
        rotate_free_circles(mock_memorygrid, all_circles, free_circle_seq_idxs, rotate_entry, chain_to_move_out_of_pz)
        
        assert mock_memorygrid.rotate.called
        # Check that ion chain was updated for entry rotation
        assert mock_memorygrid.ion_chains.__setitem__.called


class TestCheckDuplicates:
    """Test the check_duplicates function."""
    
    def test_check_duplicates_no_duplicates(self):
        """Test check duplicates with no duplicates."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.idc_dict = {0: ((0, 0), (0, 1)), 1: ((1, 0), (1, 1)), 2: ((2, 0), (2, 1))}
        
        lst = [0, 1, 2]  # No duplicates
        parking_idc = ((2, 0), (2, 1))
        max_number_parking = 2
        
        # Should not raise any exceptions
        check_duplicates(lst, mock_memorygrid, parking_idc, max_number_parking)
    
    def test_check_duplicates_with_regular_edge_duplicate(self):
        """Test check duplicates with duplicate in regular edge."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.idc_dict = {0: ((0, 0), (0, 1)), 1: ((1, 0), (1, 1)), 2: ((2, 0), (2, 1))}
        
        lst = [0, 0, 1]  # Duplicate in non-parking edge
        parking_idc = ((2, 0), (2, 1))
        max_number_parking = 2
        
        with pytest.raises(AssertionError, match="More than one chain in edge"):
            check_duplicates(lst, mock_memorygrid, parking_idc, max_number_parking)
    
    def test_check_duplicates_with_parking_edge_overflow(self):
        """Test check duplicates with too many chains in parking edge."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.idc_dict = {0: ((0, 0), (0, 1)), 1: ((1, 0), (1, 1)), 2: ((2, 0), (2, 1))}
        
        lst = [2, 2, 2]  # Three chains in parking edge
        parking_idc = ((2, 0), (2, 1))
        max_number_parking = 2  # Only 2 allowed
        
        with pytest.raises(AssertionError, match="More than 2 chains in parking edge"):
            check_duplicates(lst, mock_memorygrid, parking_idc, max_number_parking)
    
    def test_check_duplicates_with_allowed_parking_duplicates(self):
        """Test check duplicates with allowed duplicates in parking edge."""
        mock_memorygrid = MagicMock()
        mock_memorygrid.idc_dict = {0: ((0, 0), (0, 1)), 1: ((1, 0), (1, 1)), 2: ((2, 0), (2, 1))}
        
        lst = [0, 1, 2, 2]  # Two chains in parking edge
        parking_idc = ((2, 0), (2, 1))
        max_number_parking = 2  # Exactly 2 allowed
        
        # Should not raise any exceptions
        check_duplicates(lst, mock_memorygrid, parking_idc, max_number_parking)