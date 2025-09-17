from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dagdependency
from qiskit.dagcircuit import DAGDependency
from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements

from .cycles import get_state_idxs
from .graph_utils import create_dist_dict, update_distance_map
from .scheduling import pick_pz_for_2_q_gate

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit.dagcircuit import DAGDepNode

    from .graph import Graph


def is_qasm_file(file_path: Path) -> bool:
    with file_path.open(encoding="utf-8") as file:
        # Read the first line of the file (7th line, specific to MQT Bench)
        for _ in range(7):
            first_line = file.readline()
        # Check if the first line contains the OPENQASM identifier
        return "OPENQASM" in first_line


def extract_qubits_from_gate(gate_line: str) -> list[int]:
    """Extract qubit numbers from a gate operation line."""
    # Regular expression to match qubits (assuming they are in the format q[<number>])
    pattern = re.compile(r"q\[(\d+)\]")
    matches = pattern.findall(gate_line)

    # Convert matched qubit numbers to integers
    return [int(match) for match in matches]


def parse_qasm(filename: Path) -> list[tuple[int, ...]]:
    """Parse a QASM file and return qubits used for each gate
    preserving their order."""
    gates_and_qubits = []
    with filename.open(encoding="utf-8") as file:
        for _line in file:
            line = _line.strip()

            # Check if line represents a gate operation
            if not line.startswith(("OPENQASM", "include", "qreg", "creg", "gate", "barrier", "measure")):
                qubits = extract_qubits_from_gate(line)
                if qubits:
                    gates_and_qubits.append(tuple(qubits))
    return gates_and_qubits


def compile_qasm_file(filename: Path) -> list[tuple[int, ...]]:
    """Compile a QASM file and return the compiled sequence of qubits."""
    # Check if the file is a valid QASM file
    if not is_qasm_file(filename):
        msg = "Invalid QASM file format"
        raise ValueError(msg)
    # Parse the QASM file to extract the qubits used for each gate
    return parse_qasm(filename)


def get_front_layer(dag: DAGDependency) -> list[DAGDepNode]:
    """Get the front layer of the DAG."""
    front_layer = []
    for node in dag.get_nodes():
        # If a node has no predecessors, it's in the front layer
        if not dag.direct_predecessors(node.node_id):
            front_layer.append(node)
    return front_layer


def remove_node(dag: DAGDependency, node: DAGDepNode) -> None:
    """Execute a node and update the DAG (remove the node and its edges)."""
    # if dag.direct_successors(node.node_id):
    #    for successor in dag.direct_successors(node.node_id):
    #        dag._multi_graph.remove_edge(node.node_id, successor)
    dag._multi_graph.remove_node(node.node_id)


def find_best_gate(
    graph: Graph,
    front_layer: list[DAGDepNode],
    dist_map: dict[int, dict[str, int]],
    gate_info_map: dict[DAGDepNode, str],
) -> DAGDepNode:
    """Find the best gate to execute based on distance."""
    min_gate_cost = math.inf
    for _, gate_node in enumerate(front_layer):
        qubit_indices = gate_node.qindices
        pz_of_node = gate_info_map[gate_node]
        pz = graph.pzs_name_map[pz_of_node]
        if gate_node in pz.getting_processed:
            return gate_node
        gate_cost = max(dist_map[qs][pz_of_node] for qs in qubit_indices)
        # if both ions of 2-qubit gate are in pz execute 2-qubit gate
        if len(qubit_indices) == 2 and gate_cost == 0:
            return gate_node
        if gate_cost < min_gate_cost:
            min_gate_cost = gate_cost
            best_gate = gate_node
    return best_gate


def manual_copy_dag(dag: DAGDependency) -> DAGDependency:
    new_dag = DAGDependency()
    # Recreate quantum registers in the new DAG
    for qreg in dag.qregs.values():
        new_dag.add_qreg(qreg)
    # Iterate over all operation nodes in the original DAG and copy them
    for node in dag.get_nodes():
        new_dag.add_op_node(node.op, node.qargs, node.cargs)
    return new_dag


def create_dag(filename: Path) -> DAGDependency:
    qc = QuantumCircuit.from_qasm_file(str(filename))
    # Remove barriers
    qc = RemoveBarriers()(qc)
    # Remove measurement operations
    qc = RemoveFinalMeasurements()(qc)
    return circuit_to_dagdependency(qc)


def create_initial_sequence(filename: Path) -> list[tuple[int, ...]]:
    # assert file is a qasm file
    assert is_qasm_file(filename), "The file is not a valid QASM file."
    return parse_qasm(filename)


def create_updated_sequence_destructive(
    graph: Graph,
    filename: Path,
    dag_dep: DAGDependency,
    use_dag: bool,
) -> tuple[list[tuple[int, ...]], list[int], DAGDependency]:
    # assert file is a qasm file
    assert is_qasm_file(filename), "The file is not a valid QASM file."

    # generate sequence
    if use_dag is False:
        seq = parse_qasm(filename)
        dag_dep = None
    else:
        working_dag = manual_copy_dag(dag_dep)
        seq = []

        graph.dist_dict = create_dist_dict(graph)
        state = get_state_idxs(graph)
        dist_map = update_distance_map(graph, state)

        # first_flag = True
        while True:
            first_gates = get_front_layer(working_dag)
            if not first_gates:
                break
            # get all pzs of this front layer gates (in this logic have to reverse the dict)
            pz_info_map = map_front_gates_to_pzs(graph, front_layer_nodes=first_gates)
            # reverse it
            gate_info_map = {value: key for key, values in pz_info_map.items() for value in values}

            for pz_name in pz_info_map:
                # only include pzs that can process a gate of front layer
                if pz_info_map[pz_name]:
                    first_gate_to_execute = find_best_gate(graph, pz_info_map[pz_name], dist_map, gate_info_map)
                    # if first_flag == True:
                    #     next_node = first_gate_to_execute
                    # first_flag = False
                    remove_node(working_dag, first_gate_to_execute)
                    seq.append(tuple(first_gate_to_execute.qindices))

    flat_seq = [item for sublist in seq for item in sublist]

    return seq, flat_seq, dag_dep  # , next_node


def get_front_layer_non_destructive(dag: DAGDependency, virtually_processed_nodes: set[int]) -> list[DAGDepNode]:
    """Get the front layer of the DAG without modifying it."""
    front_layer = []

    for node in dag.get_nodes():
        # Skip nodes we've already processed
        if node.node_id in virtually_processed_nodes:
            continue

        # Check if all predecessors have been processed
        predecessors = dag.direct_predecessors(node.node_id)
        if not predecessors or all(pred in virtually_processed_nodes for pred in predecessors):
            front_layer.append(node)

    return front_layer


def map_front_gates_to_pzs(graph: Graph, front_layer_nodes: list[DAGDepNode]) -> dict[str, list[DAGDepNode]]:
    """Create list of all front layer gates at each processing zone."""
    gates_of_pz_info: dict[str, list[DAGDepNode]] = {pz.name: [] for pz in graph.pzs}
    for seq_node in front_layer_nodes:
        seq_elem = tuple(seq_node.qindices)
        if len(seq_elem) == 1:
            elem = seq_elem[0]
            pz = graph.map_to_pz[elem]
            # if gates_of_pz_info[pz] == []:

        elif len(seq_elem) == 2:
            # only pick processing zone for 2-qubit gate if not already locked -> then lock it
            # TODO 26.03. (2): seq_elem: [11, 10], in locked_gates: (11, 10) -> checked diese if clause nicht
            if seq_elem not in graph.locked_gates:
                pz = pick_pz_for_2_q_gate(graph, seq_elem[0], seq_elem[1])
                graph.locked_gates[seq_elem] = pz
            else:
                pz = graph.locked_gates[seq_elem]
            # if gates_of_pz_info[pz] == []:
            # gates_of_pz_info[pz].append(seq_elem[0])
            # gates_of_pz_info[pz].append(seq_elem[1])
        else:
            msg = "wrong gate type"
            raise ValueError(msg)

        gates_of_pz_info[pz].append(seq_node)
    # print('\ngates of pz info: ', {pz: [node.qindices for node in nodes] for pz, nodes in gates_of_pz_info.items()}, '\n')
    return gates_of_pz_info


def remove_processed_gates(graph: Graph, dag: DAGDependency, removed_nodes: dict[str, DAGDepNode]) -> None:
    """
    Remove the processed gates of each processing zone from both the DAG and sequence.

    Args:
        graph: Graph object containing the gate sequence
        dag: DAG representing dependencies between gates
        first_gates_by_pz: Dictionary mapping processing zones to their first gates
    """
    # Track which gates are removed
    removed_gates = []

    # Process each processing zone's first gate
    for _pz_name, first_gate in removed_nodes.items():
        # Remove the gate from the sequence
        gate_indices = tuple(first_gate.qindices)
        if gate_indices in graph.sequence:
            graph.sequence.remove(gate_indices)
            removed_gates.append(first_gate)
            # print(f"Removed gate {gate_indices} from sequence for PZ {pz_name}")

        # Remove the gate from the DAG
        node_id = first_gate.node_id
        if dag.get_node(node_id):
            dag._multi_graph.remove_node(node_id)
            # print(f"Removed node {node_id} from DAG for PZ {pz_name}")


def get_all_first_gates_and_update_sequence_non_destructive(
    graph: Graph,
    dag: DAGDependency,
    max_rounds: int = 5,
) -> dict[str, DAGDepNode]:
    """Get the first gates from the DAG for each processing zone (only first round, so they are simultaneously processable).
    Continue finding the subsequent "first gates" and update the sequence accordingly.
    Creates a compiled list of gates (ordered) for each pz from the DAG Dependency."""

    ordered_sequence = []
    processed_nodes: set[DAGDepNode] = set()  # Track nodes we've "virtually removed"
    # Dictionary to store the first gate for each processing zone
    first_nodes_by_pz = {}

    # update dist map
    state = get_state_idxs(graph)
    dist_map = update_distance_map(graph, state)
    for round_recalc_fl in range(max_rounds):
        # Get front layer excluding already processed nodes
        front_layer_nodes = get_front_layer_non_destructive(dag, processed_nodes)

        # If front layer is empty, done
        if not front_layer_nodes:
            break

        pz_info_map = map_front_gates_to_pzs(graph, front_layer_nodes)
        gate_info_map = {value: key for key, values in pz_info_map.items() for value in values}

        # Track gates processed in this round to ensure maximum parallelism
        round_processed_gates = []

        # Process one gate for each processing zone that has available gates
        for pz_name in pz_info_map:
            if pz_info_map[pz_name]:
                # Find the best gate for this processing zone
                best_gate = find_best_gate(graph, pz_info_map[pz_name], dist_map, gate_info_map)

                # Save the first gate that can be processed for each pz (only of first round, since otherwise can not be simultaneously processed)
                if round_recalc_fl == 0 and pz_name not in first_nodes_by_pz:
                    first_nodes_by_pz[pz_name] = best_gate

                # Add to the processed list for this round
                round_processed_gates.append(best_gate)

                # Update the ordered sequence
                ordered_sequence.append(tuple(best_gate.qindices))

                # Mark as processed
                processed_nodes.add(best_gate.node_id)

        # Remove all processed gates from the original sequence
        for gate in round_processed_gates:
            if tuple(gate.qindices) in graph.sequence:
                graph.sequence.remove(tuple(gate.qindices))

    # Update the final sequence
    graph.sequence = ordered_sequence + graph.sequence

    return first_nodes_by_pz
