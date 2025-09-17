import re
from pathlib import Path


def is_qasm_file(filename: Path) -> bool:
    # Check if the file has a .qasm extension
    if filename.suffix != ".qasm":
        return False

    try:
        with filename.open(encoding="utf-8") as file:
            # Read the first line of the file (7th line, specific to MQT Bench)
            for _f in range(7):
                first_line = file.readline()
            # Check if the first line contains the OPENQASM identifier
            return "OPENQASM" in first_line
    except OSError:
        # If the file cannot be opened, return False
        return False


def extract_qubits_from_gate(gate_line: str) -> list[int]:
    """Extract qubit numbers from a gate operation line."""
    # Regular expression to match qubits (assuming they are in the format q[<number>])
    pattern = re.compile(r"q\[(\d+)\]")
    matches = pattern.findall(gate_line)

    # Convert matched qubit numbers to integers
    return [int(match) for match in matches]


def parse_qasm(filename: Path) -> list[tuple[int, ...]]:
    """Parse a QASM file and return qubits used for each gate preserving their order."""
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


def compile(filename: Path | str) -> list[tuple[int, ...]]:  # noqa: A001
    """Compile a QASM file and return the compiled sequence of qubits."""
    filename = Path(filename)

    # Check if the file is a valid QASM file
    if not is_qasm_file(filename):
        msg = "Invalid QASM file format"
        raise ValueError(msg)

    # Parse the QASM file to extract the qubits used for each gate
    return parse_qasm(filename)

    # print(gates_and_qubits)
    # # Compile the sequence of qubits
    # sequence = []
    # for qubits in gates_and_qubits:
    #     for qubit in qubits:
    #         sequence.append(qubit)


# def get_front_layer(dag):
#     """Get the front layer of the DAG."""
#     front_layer = []
#     for node in dag.get_nodes():
#         # If a node has no predecessors, it's in the front layer
#         if not dag.direct_predecessors(node.node_id):
#             front_layer.append(node)
#     return front_layer


# def remove_node(dag, node):
#     """Execute a node and update the DAG (remove the node and its edges)."""
#     # if dag.direct_successors(node.node_id):
#     #    for successor in dag.direct_successors(node.node_id):
#     #        dag._multi_graph.remove_edge(node.node_id, successor)
#     dag._multi_graph.remove_node(node.node_id)


# def find_best_gate(front_layer, dist_map):
#     """Find the best gate to execute based on distance."""
#     min_gate_cost = math.inf
#     for _i, gate_node in enumerate(front_layer):
#         qubit_indices = gate_node.qindices
#         gate_cost = max([dist_map[qs] for qs in qubit_indices])
#         # if both ions of 2-qubit gate are in pz execute 2-qubit gate
#         if len(qubit_indices) == 2 and gate_cost == 0:
#             return gate_node
#         if gate_cost < min_gate_cost:
#             min_gate_cost = gate_cost
#             best_gate = gate_node
#     return best_gate


# def manual_copy_dag(dag):
#     new_dag = DAGDependency()

#     # Recreate quantum registers in the new DAG
#     for qreg in dag.qregs.values():
#         new_dag.add_qreg(qreg)

#     # Iterate over all operation nodes in the original DAG and copy them
#     for node in dag.get_nodes():
#         new_dag.add_op_node(node.op, node.qargs, node.cargs)

#     return new_dag


# def update_sequence(dag, dist_map):
#     """Get the sequence of gates from the DAG.
#     Creates a new DAG and removes all gates from it while creating the sequence."""
#     working_dag = manual_copy_dag(dag)
#     sequence = []
#     i = 0
#     while True:
#         first_gates = get_front_layer(working_dag)
#         if not first_gates:
#             break
#         first_gate_to_execute = find_best_gate(first_gates, dist_map)
#         if i == 0:
#             first_node = first_gate_to_execute
#         i = 1
#         remove_node(working_dag, first_gate_to_execute)
#         sequence.append(first_gate_to_execute.qindices)
#     return sequence, first_node
