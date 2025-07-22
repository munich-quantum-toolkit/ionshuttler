"""Tests for single shuttler compilation module."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from qiskit.dagcircuit import DAGDependency
from qiskit import QuantumCircuit

from src.mqt.ionshuttler.single_shuttler.compilation import (
    is_qasm_file,
    extract_qubits_from_gate,
    parse_qasm,
    get_front_layer,
    remove_node,
    find_best_gate,
    manual_copy_dag,
    update_sequence,
)


class TestIsQasmFile:
    """Test the is_qasm_file function."""
    
    def test_is_qasm_file_with_valid_extension_and_content(self):
        """Test with valid .qasm file containing OPENQASM."""
        mock_content = "\n" * 6 + "OPENQASM 2.0;\n"
        with patch("pathlib.Path.open", mock_open(read_data=mock_content)):
            assert is_qasm_file("test.qasm") is True
    
    def test_is_qasm_file_with_invalid_extension(self):
        """Test with invalid file extension."""
        assert is_qasm_file("test.txt") is False
    
    def test_is_qasm_file_without_openqasm(self):
        """Test with .qasm extension but no OPENQASM content."""
        mock_content = "\n" * 6 + "some other content\n"
        with patch("pathlib.Path.open", mock_open(read_data=mock_content)):
            assert is_qasm_file("test.qasm") is False
    
    def test_is_qasm_file_with_file_error(self):
        """Test with file that cannot be opened."""
        with patch("pathlib.Path.open", side_effect=OSError("File not found")):
            assert is_qasm_file("nonexistent.qasm") is False


class TestExtractQubitsFromGate:
    """Test the extract_qubits_from_gate function."""
    
    def test_extract_single_qubit(self):
        """Test extracting single qubit from gate line."""
        gate_line = "h q[0];"
        result = extract_qubits_from_gate(gate_line)
        assert result == [0]
    
    def test_extract_multiple_qubits(self):
        """Test extracting multiple qubits from gate line."""
        gate_line = "cx q[1], q[3];"
        result = extract_qubits_from_gate(gate_line)
        assert result == [1, 3]
    
    def test_extract_no_qubits(self):
        """Test with line containing no qubits."""
        gate_line = "barrier;"
        result = extract_qubits_from_gate(gate_line)
        assert result == []
    
    def test_extract_qubits_with_higher_indices(self):
        """Test extracting qubits with higher indices."""
        gate_line = "cx q[10], q[25];"
        result = extract_qubits_from_gate(gate_line)
        assert result == [10, 25]


class TestParseQasm:
    """Test the parse_qasm function."""
    
    def test_parse_qasm_with_gates(self):
        """Test parsing QASM file with gate operations."""
        mock_content = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];"""
        
        with patch("pathlib.Path.open", mock_open(read_data=mock_content)):
            result = parse_qasm("test.qasm")
            expected = [(0,), (0, 1)]
            assert result == expected
    
    def test_parse_qasm_with_path_object(self):
        """Test parsing with Path object input."""
        mock_content = """OPENQASM 2.0;
h q[0];"""
        
        with patch("pathlib.Path.open", mock_open(read_data=mock_content)):
            result = parse_qasm(Path("test.qasm"))
            expected = [(0,)]
            assert result == expected
    
    def test_parse_qasm_empty_gates(self):
        """Test parsing QASM file with no gate operations."""
        mock_content = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];"""
        
        with patch("pathlib.Path.open", mock_open(read_data=mock_content)):
            result = parse_qasm("test.qasm")
            assert result == []


class TestGetFrontLayer:
    """Test the get_front_layer function."""
    
    def test_get_front_layer_with_mock_dag(self):
        """Test getting front layer from DAG."""
        # Create mock DAG and nodes
        mock_dag = MagicMock()
        mock_node1 = MagicMock()
        mock_node1.node_id = 1
        mock_node2 = MagicMock()
        mock_node2.node_id = 2
        mock_node3 = MagicMock()
        mock_node3.node_id = 3
        
        mock_dag.get_nodes.return_value = [mock_node1, mock_node2, mock_node3]
        mock_dag.direct_predecessors.side_effect = lambda x: [] if x in [1, 2] else [1]
        
        result = get_front_layer(mock_dag)
        assert len(result) == 2
        assert mock_node1 in result
        assert mock_node2 in result


class TestRemoveNode:
    """Test the remove_node function."""
    
    def test_remove_node_from_dag(self):
        """Test removing node from DAG."""
        mock_dag = MagicMock()
        mock_node = MagicMock()
        mock_node.node_id = 1
        
        remove_node(mock_dag, mock_node)
        mock_dag._multi_graph.remove_node.assert_called_once_with(1)


class TestFindBestGate:
    """Test the find_best_gate function."""
    
    def test_find_best_gate_two_qubit_in_pz(self):
        """Test finding best gate when two-qubit gate has both qubits in PZ."""
        mock_node1 = MagicMock()
        mock_node1.qindices = [0, 1]
        mock_node2 = MagicMock()
        mock_node2.qindices = [2]
        
        front_layer = [mock_node1, mock_node2]
        dist_map = {0: 0, 1: 0, 2: 5}  # Both qubits of first gate in PZ (distance 0)
        
        result = find_best_gate(front_layer, dist_map)
        assert result == mock_node1
    
    def test_find_best_gate_minimum_cost(self):
        """Test finding gate with minimum cost."""
        mock_node1 = MagicMock()
        mock_node1.qindices = [0]
        mock_node2 = MagicMock()
        mock_node2.qindices = [1]
        
        front_layer = [mock_node1, mock_node2]
        dist_map = {0: 3, 1: 1}  # Second gate has lower cost
        
        result = find_best_gate(front_layer, dist_map)
        assert result == mock_node2


class TestManualCopyDag:
    """Test the manual_copy_dag function."""
    
    def test_manual_copy_dag(self):
        """Test manually copying a DAG."""
        # Create a simple quantum circuit and convert to DAG
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        from qiskit.converters import circuit_to_dagdependency
        original_dag = circuit_to_dagdependency(qc)
        
        copied_dag = manual_copy_dag(original_dag)
        
        # Check that it's a different object but has same structure
        assert copied_dag is not original_dag
        assert isinstance(copied_dag, DAGDependency)
        assert len(list(copied_dag.get_nodes())) == len(list(original_dag.get_nodes()))


class TestUpdateSequence:
    """Test the update_sequence function."""
    
    def test_update_sequence_basic(self):
        """Test updating sequence from DAG."""
        # Create a simple quantum circuit and convert to DAG
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        from qiskit.converters import circuit_to_dagdependency
        dag = circuit_to_dagdependency(qc)
        
        dist_map = {0: 1, 1: 2}
        
        sequence, first_node = update_sequence(dag, dist_map)
        
        assert isinstance(sequence, list)
        assert len(sequence) > 0
        assert first_node is not None
        # Check that sequence contains tuples of qubit indices
        for gate in sequence:
            assert isinstance(gate, tuple)
            assert all(isinstance(qubit, int) for qubit in gate)