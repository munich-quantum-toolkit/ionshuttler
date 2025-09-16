from __future__ import annotations

import contextlib
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dagdependency
from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements

from .compilation import is_qasm_file, manual_copy_dag, parse_qasm, remove_node, update_sequence
from .graph_utils import get_idc_from_idx, get_idx_from_idc
from .plotting import plot_state

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGDependency, DAGDepNode

    from .cycles import MemoryZone
    from .types import Edge, Graph


show_plot = False
save_plot = False
if save_plot:
    # Create a folder for each run with a timestamp (plot widget)
    run_folder = Path(f"plots/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_folder.mkdir(parents=True, exist_ok=True)
else:
    run_folder = Path()


def create_starting_config(n_of_chains: int, graph: Graph, seed: int | None = None) -> tuple[dict[int, Edge], int]:
    if seed is not None:
        random.seed(seed)
        starting_traps = []
        traps = [edges for edges in graph.edges() if graph.get_edge_data(edges[0], edges[1])["edge_type"] == "trap"]
        n_of_traps = len(traps)
        random_starting_traps = random.sample(range(n_of_traps), (n_of_chains))
        for trap in random_starting_traps:
            starting_traps.append(traps[trap])
    else:
        starting_traps = [
            edges for edges in graph.edges() if graph.get_edge_data(edges[0], edges[1])["edge_type"] == "trap"
        ][:n_of_chains]
    number_of_registers = len(starting_traps)

    # place ions onto traps (ion0 on starting_trap0)
    ion_chains = {}
    for ion, idc in enumerate(starting_traps):
        ion_chains[ion] = idc

    return ion_chains, number_of_registers


def preprocess(memorygrid: MemoryZone, sequence: list[int]) -> MemoryZone:
    # TODO check if this loop is needed (use unique_sequence instead of sequence now)
    # TODO combine with create_move_list? But max_length is different
    # unique sequence is sequence without repeating elements (for move_list and 2-qubit gates)
    unique_sequence = []
    for seq_elem in sequence:
        if seq_elem not in unique_sequence:
            unique_sequence.append(seq_elem)
    sequence = unique_sequence

    need_rotate = [False] * len(sequence)
    while sum(need_rotate) < len(sequence):
        for i, rotate_chain in enumerate(sequence):
            edge_idc = memorygrid.ion_chains[rotate_chain]
            next_edge = memorygrid.find_next_edge(edge_idc)
            state_edges_idx = memorygrid.get_state_idxs()

            if (
                memorygrid.have_common_junction_node(edge_idc, next_edge) is False
                and get_idx_from_idc(memorygrid.idc_dict, next_edge) not in state_edges_idx
            ):
                memorygrid.ion_chains[rotate_chain] = next_edge
            else:
                need_rotate[i] = True

    return memorygrid


def create_move_list(memorygrid: MemoryZone, sequence: list[int], max_length: int = 10) -> list[int]:
    """
    max_length: max length of move_list (if sequence is longer than max_length, only first max_length elements are considered)
    """
    # unique sequence is sequence without repeating elements (for move_list and 2-qubit gates)
    unique_sequence = []
    for seq_elem in sequence:
        if seq_elem not in unique_sequence:
            unique_sequence.append(seq_elem)
            if len(unique_sequence) == max_length:
                break

    path_length_sequence = {}
    move_list: list[int] = []
    for i, rotate_chain in enumerate(unique_sequence):
        edge_idc = memorygrid.ion_chains[rotate_chain]
        # TODO shortest path here maybe not optimal?
        path_to_go = nx.shortest_path(
            memorygrid.graph,
            edge_idc[0],
            memorygrid.graph_creator.processing_zone,
            lambda _, __, edge_attr_dict: (edge_attr_dict["edge_type"] == "first_entry_connection") * 1e8 + 1,
        )
        path_length_sequence[rotate_chain] = len(path_to_go)

        if i == 0 or sum(
            np.array([path_length_sequence[rotate_chain]] * len(move_list))
            > np.array([path_length_sequence[chain] for chain in move_list])
        ) == len(move_list):
            move_list.append(rotate_chain)

    # # add exit edges (needed in rare cases, when chain was moved into exit but dag dependency changed right after that -> chain is in exit but not in move sequence)
    # for exit_connection_idc in memorygrid.graph_creator.path_to_pz:
    #     ion = memorygrid.find_chain_in_edge(exit_connection_idc)
    #     if ion is not None and ion not in move_list:
    #         move_list.insert(0, ion)
    # NEW: add chains in exit connections to move_list as below for entry connections
    # -> for rare case that dag changed -> chain in exit connection was placed in front
    # -> overwrote other chain in exit connection which was still in move list but now later than the one that was inserted in front
    chains_in_exit_connections: list[int] = []
    for ion, chain_edge_idx in enumerate(memorygrid.get_state_idxs()):
        if chain_edge_idx in memorygrid.graph_creator.path_to_pz_idxs:
            chains_in_exit_connections.insert(0, ion)

    if len(chains_in_exit_connections) > 0:
        for ion in chains_in_exit_connections:
            with contextlib.suppress(Exception):
                move_list.remove(ion)
            move_list = [ion, *move_list]

    # get chains in all entry edges and place in front
    # chain in entry must move out
    chains_in_entry_connections = []
    for ion, chain_edge_idx in enumerate(memorygrid.get_state_idxs()):
        if chain_edge_idx in memorygrid.graph_creator.path_from_pz_idxs:
            if chain_edge_idx == get_idx_from_idc(memorygrid.idc_dict, memorygrid.graph_creator.entry_edge):
                # place chain in entry at the end of move_list -> so later looping over list leads to chain in entry being first
                chains_in_entry_connections.append(ion)
            else:
                chains_in_entry_connections.insert(0, ion)

    if len(chains_in_entry_connections) > 0:
        for ion in chains_in_entry_connections:
            with contextlib.suppress(Exception):
                move_list.remove(ion)
            move_list = [ion, *move_list]

    return move_list


def create_initial_sequence(
    distance_map: dict[int, int], filename: Path, compilation: bool
) -> tuple[list[tuple[int, ...]], list[int], DAGDependency | None, DAGDepNode | None]:
    # assert file is a qasm file
    assert is_qasm_file(filename), "The file is not a valid QASM file."

    # generate sequence
    if compilation is False:
        seq = parse_qasm(filename)
        next_node = None
        dag_dep = None
    else:
        qc = QuantumCircuit.from_qasm_file(filename)
        # Remove barriers
        qc = RemoveBarriers()(qc)
        # Remove measurement operations
        qc = RemoveFinalMeasurements()(qc)

        dag_dep = circuit_to_dagdependency(qc)
        gate_ids, next_node = update_sequence(dag_dep, distance_map)
        seq = [tuple(gate) for gate in gate_ids]

    flat_seq = [item for sublist in seq for item in sublist]

    return seq, flat_seq, dag_dep, next_node


def create_circles_for_moves(
    memorygrid: MemoryZone,
    move_list: list[int],
    flat_seq: list[int],
    gate_execution_finished: bool,
    new_gate_starting: bool,
) -> tuple[dict[int, list[Edge]], bool, int | None]:
    # CREATE CIRCLES #
    # create circles for all chains in move_list (dictionary with chain as key and circle_idcs as value)
    rotate_entry = False
    chain_to_park = memorygrid.find_chain_in_edge(memorygrid.graph_creator.path_to_pz[-1])
    chain_to_move_out_of_pz = None
    if memorygrid.count_chains_in_parking() < memorygrid.max_num_parking or gate_execution_finished:
        parking_open = True
    else:
        parking_open = False

    all_circles: dict[int, list[Edge]] = {}
    # need to find next_edges before for bfs search of "out of entry move"
    next_edges: dict[int, Edge] = {}
    for rotate_chain in move_list:
        edge_idc = memorygrid.ion_chains[rotate_chain]
        # if chain is needed again (is present in rest of sequence) -> move (only out of entry) towards exit instead of top left
        towards = "exit" if rotate_chain in flat_seq[1:] else (0, 0)
        next_edges[rotate_chain] = memorygrid.find_next_edge(edge_idc, towards=towards)

    in_and_into_exit_moves = {}
    for rotate_chain in move_list:
        edge_idc = memorygrid.ion_chains[rotate_chain]
        next_edge = next_edges[rotate_chain]

        # make edge_idc and next_edge consistent
        edge_idc, next_edge = memorygrid.find_ordered_edges(edge_idc, next_edge)

        # moves in pz
        if get_idx_from_idc(memorygrid.idc_dict, next_edge) in {
            *memorygrid.graph_creator.path_to_pz_idxs,
            get_idx_from_idc(memorygrid.idc_dict, memorygrid.graph_creator.parking_edge),
        }:
            in_and_into_exit_moves[rotate_chain] = edge_idc
            all_circles[rotate_chain] = [edge_idc, next_edge]
            # block moves to pz if parking is full (now blocks if parking not open and chain moving in exit and its next edge is in state_idxs)
            if (
                #     get_idx_from_idc(memorygrid.idc_dict, next_edge)
                #     in [
                #         *memorygrid.graph_creator.path_to_pz_idxs,
                #         get_idx_from_idc(memorygrid.idc_dict, memorygrid.graph_creator.parking_edge),
                #     ]
                #     and
                parking_open is False
            ) and (get_idx_from_idc(memorygrid.idc_dict, next_edge) in memorygrid.state_idxs):
                all_circles[rotate_chain] = [edge_idc, edge_idc]

            # #) and (get_idx_from_idc(memorygrid.idc_dict, next_edge) in stop_exit_edges or stop_exit_edges == []):

            #     # old
            #     # all_circles[rotate_chain] = [edge_idc, edge_idc]
            #     # # needed later for blocking moves to parking
            #     # stop_exit_edges.append(get_idx_from_idc(memorygrid.idc_dict, edge_idc))

            #     # new
            #     # needed later for blocking moves to parking
            #     stop_exit_edges.append(get_idx_from_idc(memorygrid.idc_dict, edge_idc))
            #     print('stop_exit_edges: ', stop_exit_edges, 'rotate_chain: ', rotate_chain)
            #     #((20, 20.0), (21, 21.0))
            #     #((21, 21.0), (22, 22.0))
            #     # should only stop if next edge is in stop_exit_edges -> if [] then no stop
            #     if len(stop_exit_edges) > 1:
            #         all_circles[rotate_chain] = [edge_idc, edge_idc]
            #         print('rotate_chain1: ', rotate_chain, stop_exit_edges)

        # moves without circle
        # also if chain is moving out of entry connections (entry is handled in create_outer_circle)
        elif (
            not memorygrid.check_if_edge_is_filled(next_edge)
            or get_idx_from_idc(memorygrid.idc_dict, edge_idc) in memorygrid.graph_creator.path_from_pz_idxs[:-1]
        ):
            all_circles[rotate_chain] = [edge_idc, next_edge]

        # moves with circle
        else:
            # create circle (deleted in create_outer_circle: in parking circle is a "stop move")
            all_circles[rotate_chain] = memorygrid.create_outer_circle(edge_idc, next_edge, list(next_edges.values()))

    # move chain out of parking edge if needed
    chains_in_parking = memorygrid.find_chains_in_parking()

    # if pz full and no chain is moving out (not in state_idxs entry edge) but chain is moving in
    if memorygrid.count_chains_in_parking() >= memorygrid.max_num_parking and chain_to_park is not None:
        # if gate finished -> new space could be in parking
        if gate_execution_finished:
            # find least important chain in parking edge
            chain_to_move_out_of_pz = memorygrid.find_least_import_chain_in_parking(
                flat_seq, [*chains_in_parking, chain_to_park]
            )
            if chain_to_move_out_of_pz != chain_to_park:
                # move it to entry
                rotate_entry = True
                # change its path/circle to a stop move -> will be later placed into entry
                all_circles[chain_to_move_out_of_pz] = [
                    memorygrid.graph_creator.path_from_pz[0],
                    memorygrid.graph_creator.path_from_pz[0],
                ]

            # else -> chain_to_park not needed right now
            # if new gate can be executed -> bring everything to a stop in enxit
            elif new_gate_starting:
                # if chain to park should move to entry -> all chains with next edge in exit -> stop move
                for chain, edge_idc in in_and_into_exit_moves.items():
                    all_circles[chain] = [edge_idc, edge_idc]
                # maybe already covered above
                all_circles[chain_to_move_out_of_pz] = [
                    memorygrid.graph_creator.path_to_pz[-1],
                    memorygrid.graph_creator.path_to_pz[-1],
                ]
            # else -> no new gate possible with only parking chains
            # -> but also chain can't move to parking since it is least important
            # -> should be rare edge case -> chain moves from exit to entry
            else:
                rotate_entry = True
                # change its path/circle to a stop move -> will be later placed into entry
                all_circles[chain_to_move_out_of_pz] = [
                    memorygrid.graph_creator.path_from_pz[0],
                    memorygrid.graph_creator.path_from_pz[0],
                ]
        else:
            # else bring everything to a stop in exit
            # same as above
            for chain, edge_idc in in_and_into_exit_moves.items():
                all_circles[chain] = [edge_idc, edge_idc]

            # maybe already covered above
            # all_circles[chain_to_move_out_of_pz] = (
            #     memorygrid.graph_creator.path_to_pz[-1],
            #     memorygrid.graph_creator.path_to_pz[-1],
            # )

    return all_circles, rotate_entry, chain_to_move_out_of_pz


def find_movable_circles(memorygrid: MemoryZone, all_circles: dict[int, list[Edge]], move_list: list[int]) -> list[int]:
    # FIND CIRCLES THAT CAN MOVE #
    # find circles that can move while first seq ion is moving
    nonfree_circles = memorygrid.find_nonfree_and_free_circle_idxs(all_circles)
    free_circle_seq_idxs = [move_list[0]]
    for seq_circ in move_list[1:]:
        nonfree = False
        for mov_circ in free_circle_seq_idxs:
            if (seq_circ, mov_circ) in nonfree_circles or (mov_circ, seq_circ) in nonfree_circles:
                nonfree = True
                break
        if nonfree is False:
            free_circle_seq_idxs.append(seq_circ)
    return free_circle_seq_idxs


def rotate_free_circles(
    memorygrid: MemoryZone,
    all_circles: dict[int, list[Edge]],
    free_circle_seq_idxs: list[int],
    rotate_entry: bool,
    chain_to_move_out_of_pz: int | None,
) -> None:
    # ROTATE CIRCLES #
    # need circles given in idxs for rotate function
    free_circle_idxs = {}
    for seq_idx in free_circle_seq_idxs:
        free_circle_idxs[seq_idx] = [
            get_idx_from_idc(memorygrid.idc_dict, edge_idc) for edge_idc in all_circles[seq_idx]
        ]
        # rotate chains
        memorygrid.rotate(free_circle_idxs[seq_idx])
    if rotate_entry:
        assert chain_to_move_out_of_pz is not None
        memorygrid.ion_chains[chain_to_move_out_of_pz] = memorygrid.graph_creator.path_from_pz[0]


def update_sequence_and_process_gate(
    memorygrid: MemoryZone,
    seq_length: int,
    gate_execution_finished: bool,
    new_gate_starting: bool,
    dag_dep: DAGDependency | None,
    next_node: DAGDepNode | None,
    timestep: int,
    seq: list[tuple[int, ...]],
    flat_seq: list[int],
    time_in_pz_counter: int,
    next_gate_is_two_qubit_gate: bool,
    show_plot: bool,
) -> tuple[bool, bool, bool, list[tuple[int, ...]], list[int], int, DAGDependency | None, DAGDepNode | None, bool]:
    gate = seq[0]
    chains_in_parking = memorygrid.find_chains_in_parking()
    time_gate = memorygrid.time_2qubit_gate if next_gate_is_two_qubit_gate else memorygrid.time_1qubit_gate

    plot_filename = Path(run_folder) / f"plot_{timestep:03d}.pdf"

    # UPDATE SEQUENCE / PROCESS GATE #
    if sum((gate_element in chains_in_parking) for gate_element in gate) == len(gate):
        gate_execution_finished = False

        time_in_pz_counter += 1
        plot_state(
            memorygrid.graph,
            [get_idx_from_idc(memorygrid.idc_dict, edge_idc) for edge_idc in memorygrid.ion_chains.values()],
            labels=[
                f"time step {timestep}",
                f"seq elem {seq[0]} execution",
            ],
            show_plot=show_plot,
            save_plot=save_plot,
            filename=[plot_filename if save_plot else None][0],
        )

        # print time step and gate (gate x out of y)
        print(
            f"time step: {timestep}, execution of gate ({seq_length - len(seq) + 1} / {seq_length}) on qubit(s) {seq[0]}"
        )
        time_gate = memorygrid.time_2qubit_gate if next_gate_is_two_qubit_gate else memorygrid.time_1qubit_gate

        if time_in_pz_counter == time_gate:
            # END IF SEQUENCE IS FINISHED #
            if len(seq) == 1:
                print(f"\nCircuit successfully executed in {timestep} time steps.")
                return (
                    True,
                    gate_execution_finished,
                    new_gate_starting,
                    seq,
                    flat_seq,
                    time_in_pz_counter,
                    dag_dep,
                    next_node,
                    next_gate_is_two_qubit_gate,
                )

            # for _ in gate:
            #     flat_seq.pop(0)
            time_in_pz_counter = 0
            gate_execution_finished = True

            if dag_dep is None:
                assert next_node is None
                seq.pop(0)
            else:
                # update dag
                remove_node(dag_dep, next_node)
                dag_dep = manual_copy_dag(dag_dep)
                new_dist_map = memorygrid.update_distance_map()
                gate_ids, next_node = update_sequence(dag_dep, new_dist_map)
                seq = [tuple(gate) for gate in gate_ids]

            flat_seq = [item for sublist in seq for item in sublist]
            next_gate_is_two_qubit_gate = len(seq[0]) == 2

            # need counter that tracks if new gate can start with chains from parking
            # -> if last entry connection is blocked with chain that is less important than chains in parking
            # -> move chain from exit to entry without parking
            chains_in_parking = memorygrid.find_chains_in_parking()
            for gate_element in seq[0]:
                new_gate_starting = gate_element in chains_in_parking

    else:
        plot_state(
            memorygrid.graph,
            [get_idx_from_idc(memorygrid.idc_dict, edge_idc) for edge_idc in memorygrid.ion_chains.values()],
            labels=[f"time step {timestep}", f"next seq elem: {seq[0]}"],
            show_plot=show_plot,
            save_plot=save_plot,
            filename=[plot_filename if save_plot else None][0],
        )

    return (
        False,
        gate_execution_finished,
        new_gate_starting,
        seq,
        flat_seq,
        time_in_pz_counter,
        dag_dep,
        next_node,
        next_gate_is_two_qubit_gate,
    )


def check_duplicates(lst: list[int], memorygrid: MemoryZone, parking_idc: Edge, max_number_parking: int) -> None:
    parking_idx = get_idx_from_idc(memorygrid.idc_dict, parking_idc)

    # Count occurrences of each integer
    counts = Counter(lst)

    for num, count in counts.items():
        if num != parking_idx and count > 1:
            message = f"More than one chain in edge {get_idc_from_idx(memorygrid.idc_dict, num)}!"
            raise AssertionError(message)
        if num == parking_idx and count > max_number_parking:
            message = (
                f"More than {max_number_parking} chains in parking edge {get_idc_from_idx(memorygrid.idc_dict, num)}!"
            )
            raise AssertionError(message)


def run_simulation(
    memorygrid: MemoryZone,
    max_timesteps: int,
    seq: list[tuple[int, ...]],
    flat_seq: list[int],
    dag_dep: DAGDependency,
    next_node_initial: DAGDepNode,
    max_length: int,
) -> int:
    time_in_pz_counter = 0
    next_gate_is_two_qubit_gate = len(seq[0]) == 2
    gate_execution_finished = True
    new_gate_starting = False
    timestep = 0
    next_node = next_node_initial

    seq_length = len(seq)

    while timestep < max_timesteps:
        rotate_entry = False

        # update state idxs
        state_idxs = memorygrid.get_state_idxs()
        # assert check that each edge has only one chain (parking edge at most max parking)
        check_duplicates(state_idxs, memorygrid, memorygrid.graph_creator.parking_edge, memorygrid.max_num_parking)
        # preprocess (move chains within junctions)
        memorygrid = preprocess(memorygrid, flat_seq)
        # move list
        move_list = create_move_list(memorygrid, flat_seq, max_length)
        # memorygrid.state_idxs are updated in create_move_list
        # create circles for moves
        all_circles, rotate_entry, chain_to_move_out_of_pz = create_circles_for_moves(
            memorygrid, move_list, flat_seq, gate_execution_finished, new_gate_starting
        )
        new_gate_starting = False
        # find movable circles
        free_circle_seq_idxs = find_movable_circles(memorygrid, all_circles, move_list)
        # rotate free circles
        rotate_free_circles(memorygrid, all_circles, free_circle_seq_idxs, rotate_entry, chain_to_move_out_of_pz)
        # update sequence and process gate
        (
            finished,
            gate_execution_finished,
            new_gate_starting,
            seq,
            flat_seq,
            time_in_pz_counter,
            dag_dep,
            next_node,
            next_gate_is_two_qubit_gate,
        ) = update_sequence_and_process_gate(
            memorygrid,
            seq_length,
            gate_execution_finished,
            new_gate_starting,
            dag_dep,
            next_node,
            timestep,
            seq,
            flat_seq,
            time_in_pz_counter,
            next_gate_is_two_qubit_gate,
            show_plot,
        )
        if finished:
            return timestep
        timestep += 1

        state_idxs = memorygrid.get_state_idxs()

    msg = f"Maximum number of time steps {max_timesteps} reached"
    raise RuntimeError(msg)
