from __future__ import annotations

import pathlib
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

from .compilation import get_all_first_gates_and_update_sequence_non_destructive, remove_processed_gates
from .cycles import get_ions
from .graph_utils import get_idc_from_idx, get_idx_from_idc
from .plotting import plot_state
from .scheduling import (
    create_cycles_for_moves,
    create_gate_info_list,
    create_move_list,
    create_priority_queue,
    find_movable_cycles,
    find_out_of_entry_moves,
    get_partitioned_priority_queues,
    preprocess,
    rotate_free_cycles,
    update_entry_and_exit_cycles,
)

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGDependency

    from .graph import Graph
    from .types import Edge


def check_duplicates(graph: Graph) -> None:
    edge_idxs_occupied = []
    for edge_idc in graph.state.values():
        edge_idxs_occupied.append(get_idx_from_idc(graph.idc_dict, edge_idc))
    # Count occurrences of each integer
    counts = Counter(edge_idxs_occupied)

    for idx, count in counts.items():
        edge_idc = get_idc_from_idx(graph.idc_dict, idx)
        if graph.get_edge_data(edge_idc[0], edge_idc[1])["edge_type"] != "parking_edge" and count > 1:
            message = f"More than one ion in edge {edge_idc}, arch: {graph.arch}, circuit depth: {len(graph.sequence)}, seed: {graph.seed}!"
            raise AssertionError(message)

        if (
            graph.get_edge_data(edge_idc[0], edge_idc[1])["edge_type"] == "parking_edge"
            and count > graph.max_num_parking
        ):
            message = f"More than {graph.max_num_parking} chains in parking edge {edge_idc}!"
            raise AssertionError(message)


def find_pz_order(graph: Graph, gate_info_list: dict[str, list[int]]) -> list[str]:
    # find next processing zone that will execute a gate
    pz_order = []
    for gate in graph.sequence:
        if len(gate) == 1:
            ion = gate[0]
            for pz in graph.pzs:
                if ion in gate_info_list[pz.name]:
                    pz_order.append(pz.name)
                    break
        elif len(gate) == 2:
            ion1, ion2 = gate
            for pz in graph.pzs:
                if ion1 in gate_info_list[pz.name] and ion2 in gate_info_list[pz.name]:
                    pz_order.append(pz.name)
                    break
    return pz_order


def shuttle(
    graph: Graph,
    priority_queue: dict[int, str],
    timestep: int,
    cycle_or_paths: str,
    unique_folder: pathlib.Path,
) -> None:
    preprocess(graph, priority_queue)

    # Update ion chains after preprocess
    graph.state = get_ions(graph)

    check_duplicates(graph)
    part_prio_queues = get_partitioned_priority_queues(priority_queue)

    all_cycles: dict[int, list[Edge]] = {}
    # Iterate over all processing zones
    # create move list for each pz -> needed to get all cycles
    # priority queue later picks the cycles to rotate
    in_and_into_exit_moves_of_pz = {}
    for pz in graph.pzs:
        prio_queue = part_prio_queues[pz.name]
        move_list = create_move_list(graph, prio_queue, pz)
        cycles, in_and_into_exit_moves = create_cycles_for_moves(graph, move_list, cycle_or_paths, pz)
        # add cycles to all_cycles
        all_cycles = {**all_cycles, **cycles}

    out_of_entry_moves = find_out_of_entry_moves(graph, other_next_edges=[])

    for pz in graph.pzs:
        prio_queue = part_prio_queues[pz.name]
        out_of_entry_moves_of_pz = out_of_entry_moves.get(pz, None)
        if pz.name in in_and_into_exit_moves:
            in_and_into_exit_moves_of_pz = in_and_into_exit_moves[pz.name]
        update_entry_and_exit_cycles(
            graph, pz, all_cycles, in_and_into_exit_moves_of_pz, out_of_entry_moves_of_pz, prio_queue
        )

    # now general priority queue picks cycles to rotate
    chains_to_rotate = find_movable_cycles(graph, all_cycles, priority_queue, cycle_or_paths)
    rotate_free_cycles(graph, all_cycles, chains_to_rotate)

    # Update ions after rotate
    graph.state = get_ions(graph)

    labels = (
        f"timestep and seq length {timestep} {len(graph.sequence)}",
        "Sequence: %s" % [graph.sequence if len(graph.sequence) < 8 else graph.sequence[:8]],
    )

    if graph.plot is True or graph.save is True:
        plot_state(
            graph,
            labels,
            plot_ions=True,
            show_plot=graph.plot,
            save_plot=graph.save,
            plot_cycle=False,
            plot_pzs=False,
            filename=unique_folder / f"{graph.arch}_timestep_{timestep}.png",
        )


def main(graph: Graph, dag: DAGDependency, cycle_or_paths: str, use_dag: bool) -> int:
    timestep = 0
    max_timesteps = 1e6
    graph.state = get_ions(graph)

    unique_folder = pathlib.Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    if graph.save is True:
        unique_folder.mkdir(exist_ok=True, parents=True)

    if any([graph.plot, graph.save]):
        plot_state(
            graph,
            labels=("Initial state", None),
            plot_ions=True,
            show_plot=graph.plot,
            save_plot=graph.save,
            plot_cycle=False,
            plot_pzs=True,
            filename=unique_folder / f"{graph.arch}_timestep_{timestep}.pdf",
        )

    for pz in graph.pzs:
        pz.time_in_pz_counter = 0
        pz.gate_execution_finished = True

    graph.in_process = []

    if use_dag:
        next_processable_gate_nodes = get_all_first_gates_and_update_sequence_non_destructive(graph, dag)

    locked_gates: dict[tuple[int, ...], str] = {}
    while timestep < max_timesteps:
        for pz in graph.pzs:
            pz.rotate_entry = False
            pz.out_of_parking_cycle = None
            pz.out_of_parking_move = None

        if use_dag:
            gate_info_list: dict[str, list[int]] = {pz.name: [] for pz in graph.pzs}
            for pz_name, node in next_processable_gate_nodes.items():
                for ion in node.qindices:
                    gate_info_list[pz_name].append(ion)
        else:
            # update gate_info_list (list of next gate at pz)
            gate_info_list = create_gate_info_list(graph)

        # pz order - find next pz that processes a gate (only used to add ions in exit to front of prio queue)
        pz_executing_gate_order = find_pz_order(graph, gate_info_list)

        # # reset locked gates, prio q recalcs them (2-qubit gates get locked after each execution)
        graph.locked_gates = locked_gates
        # priority queue is dict with ions as keys and pz as values
        # (for 2-qubit gates pz may not match the pz of the individual ion)
        priority_queue, next_gate_at_pz_dict = create_priority_queue(graph, pz_executing_gate_order)

        # check if ions are already in processing zone
        # -> important for 2-qubit gates
        # -> leave ion in processing zone if needed in a 2-qubit gate
        for i in range(min(len(graph.pzs), len(graph.sequence))):
            # only continue if previous ion was processed
            gate = graph.sequence[i]

            if len(gate) == 2:
                ion1, ion2 = gate
                for pz in graph.pzs:
                    state1 = graph.state[ion1]
                    state2 = graph.state[ion2]
                    # append ion to in_process if it is in the correct processing zone
                    if (
                        state1 == pz.parking_edge
                        and ion1 in next_gate_at_pz_dict[pz.name]
                        and ion2 in next_gate_at_pz_dict[pz.name]
                    ):
                        graph.in_process.append(ion1)
                    if (
                        state2 == pz.parking_edge
                        and ion1 in next_gate_at_pz_dict[pz.name]
                        and ion2 in next_gate_at_pz_dict[pz.name]
                    ):
                        graph.in_process.append(ion2)

        # shuttle one timestep
        shuttle(graph, priority_queue, timestep, cycle_or_paths, unique_folder)

        # reset ions in process
        graph.in_process = []

        # Check the state of each ion in the sequence
        graph.state = get_ions(graph)

        if use_dag:
            processed_nodes = {}
            for pz_name, gate_node in next_processable_gate_nodes.items():
                pz = graph.pzs_name_map[pz_name]
                gate = tuple(ion for ion in gate_node.qindices)
                if len(gate) == 1:
                    ion = gate[0]
                    if get_idx_from_idc(graph.idc_dict, graph.state[ion]) == get_idx_from_idc(
                        graph.idc_dict, pz.parking_edge
                    ):
                        pz.gate_execution_finished = (
                            False  # set False, then check below if gate time is finished -> then True
                        )
                        pz.getting_processed.append(gate_node)
                        pz.time_in_pz_counter += 1
                        gate_time = 1

                        if pz.time_in_pz_counter == gate_time:
                            processed_nodes[pz_name] = gate_node
                            pz.getting_processed.remove(gate_node)
                            # remove the processing zone from the list
                            # (it can only process one ion)
                            # pzs.remove(pz)
                            # graph.in_process.append(ion)

                            pz.time_in_pz_counter = 0
                            pz.gate_execution_finished = True
                            # break
                elif len(gate) == 2:
                    ion1, ion2 = gate
                    state1 = graph.state[ion1]
                    state2 = graph.state[ion2]

                    # if both ions are in the processing zone, process the gate
                    if get_idx_from_idc(graph.idc_dict, state1) == get_idx_from_idc(
                        graph.idc_dict, pz.parking_edge
                    ) and get_idx_from_idc(graph.idc_dict, state2) == get_idx_from_idc(graph.idc_dict, pz.parking_edge):
                        pz.gate_execution_finished = (
                            False  # set False, then check below if gate time is finished -> then True
                        )
                        pz.getting_processed.append(gate_node)
                        pz.time_in_pz_counter += 1

                        gate_time = 3
                        if pz.time_in_pz_counter == gate_time:
                            processed_nodes[pz_name] = gate_node
                            # remove the processing zone from the list
                            # (it can only process one gate)
                            # pzs.remove(pz)

                            # remove the locked pz of the processed two-qubit gate
                            if gate in graph.locked_gates and graph.locked_gates[gate] == pz.name:
                                graph.locked_gates.pop(gate)
                            pz.time_in_pz_counter = 0
                            pz.gate_execution_finished = True
                            pz.getting_processed.remove(gate_node)
                            # break
                else:
                    msg = "Invalid gate format"
                    raise ValueError(msg)

        else:
            processed_ions: list[tuple[int, ...]] = []
            previous_ion_processed = True
            pzs = graph.pzs.copy()
            next_gates = graph.sequence[: min(len(graph.pzs), len(graph.sequence))]
            # go through the first gates in the sequence (as many as pzs or sequence length)
            # for now, gates are processed in order
            # (can only be processed in parallel if previous gates are processed)
            for i in range(min(len(graph.pzs), len(graph.sequence))):
                # only continue if previous ion was processed
                if not previous_ion_processed:
                    break
                gate = next_gates[i]
                ion_processed = False
                # wenn auf weg zu pz in anderer pz -> wird processed?
                # Problem nur für 2-qubit gate?
                for pz in pzs:
                    if len(gate) == 1:
                        ion = gate[0]
                        if get_idx_from_idc(graph.idc_dict, graph.state[ion]) == get_idx_from_idc(
                            graph.idc_dict, pz.parking_edge
                        ):
                            pz.gate_execution_finished = (
                                False  # set False, then check below if gate time is finished -> then True
                            )
                            pz.time_in_pz_counter += 1
                            gate_time = 1
                            if pz.time_in_pz_counter == gate_time:
                                processed_ions.insert(0, (ion,))
                                ion_processed = True
                                # remove the processing zone from the list
                                # (it can only process one ion)
                                pzs.remove(pz)
                                # graph.in_process.append(ion)

                                pz.time_in_pz_counter = 0
                                pz.gate_execution_finished = True
                                break
                    elif len(gate) == 2:
                        ion1, ion2 = gate
                        state1 = graph.state[ion1]
                        state2 = graph.state[ion2]

                        # if both ions are in the processing zone, process the gate
                        if get_idx_from_idc(graph.idc_dict, state1) == get_idx_from_idc(
                            graph.idc_dict, pz.parking_edge
                        ) and get_idx_from_idc(graph.idc_dict, state2) == get_idx_from_idc(
                            graph.idc_dict, pz.parking_edge
                        ):
                            pz.gate_execution_finished = (
                                False  # set False, then check below if gate time is finished -> then True
                            )
                            pz.time_in_pz_counter += 1
                            gate_time = 3
                            if pz.time_in_pz_counter == gate_time:
                                processed_ions.insert(0, (ion1, ion2))
                                ion_processed = True
                                # remove the processing zone from the list
                                # (it can only process one gate)
                                pzs.remove(pz)  # noqa: B909

                                # remove the locked pz of the processed two-qubit gate
                                if gate in graph.locked_gates and graph.locked_gates[gate] == pz.name:
                                    graph.locked_gates.pop(gate)
                                pz.time_in_pz_counter = 0
                                pz.gate_execution_finished = True
                                break
                    else:
                        msg = "Invalid gate format"
                        raise ValueError(msg)
                previous_ion_processed = ion_processed

        # Remove processed ions from the sequence (and dag if use_dag)
        if use_dag:
            if processed_nodes:
                remove_processed_gates(graph, dag, processed_nodes)
                next_processable_gate_nodes = get_all_first_gates_and_update_sequence_non_destructive(graph, dag)
                for pz_name, node in next_processable_gate_nodes.items():
                    locked_gates[tuple(node.qindices)] = pz_name
        else:
            for gate in processed_ions:
                graph.sequence.remove(gate)

        if len(graph.sequence) == 0:
            break

        timestep += 1

    return timestep
