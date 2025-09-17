from __future__ import annotations

from pathlib import Path

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements


def read_qasm_file(file_path: Path) -> QuantumCircuit:
    circuit = QuantumCircuit.from_qasm_file(file_path)
    # Remove barriers
    circuit = RemoveBarriers()(circuit)
    # Remove measurement operations
    return RemoveFinalMeasurements()(circuit)


def construct_interaction_graph(circuit: QuantumCircuit) -> nx.Graph[int]:
    graph: nx.Graph[int] = nx.Graph()
    qubits = circuit.qubits
    for qubit in qubits:
        graph.add_node(qubit._index)

    for gate in circuit.data:
        if len(gate.qubits) == 2:  # Check if it is a 2-qubit gate
            q0 = gate.qubits[0]._index
            q1 = gate.qubits[1]._index
            if graph.has_edge(q0, q1):
                graph[q0][q1]["weight"] += 1
            else:
                graph.add_edge(q0, q1, weight=1)
        elif len(gate.qubits) > 2:
            msg = "Circuit contains gates with more than 2 qubits"
            raise ValueError(msg)

    # # plot graph
    # nx.draw(graph, with_labels=True)
    # nx.draw_networkx_edge_labels(
    #     graph, pos=nx.circular_layout(graph),
    #     edge_labels={(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
    #     )
    # plt.show()

    return graph


# def partition_graph(graph, n):
#     #partitions = []
#     subgraphs = [graph]
#     assert n > 1, "Number of partitions must be greater than 1"
#     assert n <= len(graph.nodes), "Number of partitions
# must be less than or equal to the number of nodes in the graph"
#     #assert n != len(graph.nodes), "Number of partitions must
# be less than the number of nodes in the graph TODO: handle this case?"
#     while len(subgraphs) < n:
#         print('new iteration')
#         new_subgraphs = []
#         for subgraph in subgraphs:
#             #print(f"Start subgraph nodes: {list(subgraph.nodes)}",
# len(subgraph)+len(new_subgraphs),
# [list(new_subg.nodes) for new_subg in new_subgraphs])
#             print('len(subgraph):', len(subgraph))
#             print('len(new_subgraphs):', len(new_subgraphs),'\n')
#             if len(subgraphs) + len(new_subgraphs) - 1 < n:
#                 # handle cases where the subgraph has less
# than 2 nodes (happens if n = # of nodes)
#                 if len(subgraph.nodes()) < 2:
#                 #     #print(f"Subgraph nodes: {list(subgraph.nodes)}")
#                 #     #print(f"Subgraph edges: {list(subgraph.edges)}")
#                     new_subgraphs.append(subgraph)
#                     continue
#                 part1, part2 = kernighan_lin_bisection(subgraph)
#                 #print(f"Part1: {part1}, Part2: {part2}")
#                 #print(f"Subgraph nodes: {list(subgraph.nodes)}")
#                 #print(f"Subgraph edges: {list(subgraph.edges)}")
#                 new_subgraphs.append(graph.subgraph(part1).copy())
#                 new_subgraphs.append(graph.subgraph(part2).copy())
#             else:
#                 new_subgraphs.append(subgraph)
#         subgraphs = new_subgraphs
#         #partitions = subgraphs

#     return subgraphs#partitions


# def partition_graph_new(graph, n):
#     if n == 1:
#         return [graph]

#     assert n <= len(
#         graph.nodes
#     ), f"Number of partitions must be less or equal to the number of nodes {len(graph.nodes), graph.nodes, n}"
#     partitions = [graph.copy()]
#     new_partitions = []
#     while len(partitions) < n:
#         len_partitions = len(partitions)
#         for i, partition in enumerate(partitions):
#             if len_partitions + len(new_partitions) < n:
#                 if len(partition) < 2:
#                     new_partitions.append(partition)
#                     len_partitions -= 1
#                     continue

#                 part1, part2 = kernighan_lin_bisection(partition)
#                 new_partitions.append(graph.subgraph(part1).copy())
#                 new_partitions.append(graph.subgraph(part2).copy())
#             else:
#                 new_partitions.append(partition)
#         partitions = new_partitions
#         new_partitions = []
#     return partitions


# def get_partition(qasm_file_path, n):
#     circuit = read_qasm_file(qasm_file_path)
#     interaction_graph = construct_interaction_graph(circuit)
#     partition_graphs = partition_graph_new(interaction_graph, n)

#     partition = []
#     for graph in partition_graphs:
#         partition.append(list(graph.nodes))

#     return partition


def partition_graph_balanced(graph: nx.Graph[int], n: int) -> list[nx.Graph[int]]:
    """
    Partitions 'graph' into n subgraphs, attempting to keep them balanced
    in size, while using Kernighan-Lin bisection at every step.
    """
    if n == 1:
        return [graph]

    partitions = [graph.copy()]  # start with the entire graph in one partition

    while len(partitions) < n:
        # Pick the index of the largest partition.
        # We'll try to split that partition further.
        largest_idx = max(range(len(partitions)), key=lambda i: len(partitions[i]))
        largest_partition = partitions.pop(largest_idx)

        # If the partition is too small to split, put it back and stop.
        if len(largest_partition) < 2:
            partitions.append(largest_partition)
            break

        # Biset using Kernighan-Lin. This tries to produce two partitions
        # of roughly equal size that minimize inter-partition edge cuts.
        part1, part2 = kernighan_lin_bisection(largest_partition)
        sub1 = largest_partition.subgraph(part1).copy()
        sub2 = largest_partition.subgraph(part2).copy()

        # Add these two new partitions back
        partitions.append(sub1)
        partitions.append(sub2)

    # If we cannot reach exactly n partitions because subgraphs are all of size <2,
    # then 'partitions' just won't grow further. The code ensures we never overshoot n.
    return partitions


def get_partition(qasm_file_path: Path, n: int) -> list[list[int]]:
    circuit = read_qasm_file(qasm_file_path)
    interaction_graph = construct_interaction_graph(circuit)

    partition_graphs = partition_graph_balanced(interaction_graph, n)

    partition = []
    for pg in partition_graphs:
        partition.append(list(pg.nodes))

    return partition


if __name__ == "__main__":
    # Example usage

    qasm_file_path = Path(__file__).absolute().parent.parent.parent.parent.parent.parent / (
        "inputs/qasm_files/qft_no_swaps_nativegates_quantinuum_tket/qft_no_swaps_nativegates_quantinuum_tket_6.qasm"
    )
    n = 4
    print(get_partition(qasm_file_path, n))
