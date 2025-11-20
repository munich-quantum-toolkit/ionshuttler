from __future__ import annotations

import pathlib
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

# Ensure a headless backend for saving figures
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # for saving mpl figures
except Exception:
    plt = None

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


# Gate time configuration (overridable from run scripts)
GATE_TIME_1Q = 1
GATE_TIME_2Q = 3


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
    all_in_and_into_exit_moves = {}
    for pz in graph.pzs:
        prio_queue = part_prio_queues[pz.name]
        move_list = create_move_list(graph, prio_queue, pz)
        cycles, in_and_into_exit_moves = create_cycles_for_moves(graph, move_list, cycle_or_paths, pz)
        all_in_and_into_exit_moves[pz.name] = in_and_into_exit_moves
        # add cycles to all_cycles
        all_cycles.update(cycles)

    out_of_entry_moves = find_out_of_entry_moves(graph, other_next_edges=[])

    for pz in graph.pzs:
        prio_queue = part_prio_queues[pz.name]
        out_of_entry_moves_of_pz = out_of_entry_moves.get(pz, None)
        if pz.name in all_in_and_into_exit_moves:
            in_and_into_exit_moves_of_pz = all_in_and_into_exit_moves[pz.name]
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


def _save_dag_snapshot(dag: 'DAGDependency', out_path: pathlib.Path) -> None:
    """Best-effort save of a DAG snapshot with detailed debug prints.
    Tries, in order:
      1) dag.draw(filename=...) if supported by Qiskit version
      2) If draw() returns a Matplotlib figure, savefig
      3) If draw() returns a Graphviz object, try render/write/pipe
      4) Fallback to DOT/TEXT via attributes or str(dag)
    """
    print(f"[DAG SAVE] Attempting to save snapshot to {out_path}")
    # 1) Try direct write via filename param (no unsupported 'output' kw)
    try:
        dag.draw(filename=str(out_path))
        print(f"[DAG SAVE] Saved DAG snapshot via dag.draw(filename=...) to {out_path}")
        return
    except TypeError as e:
        print(f"[DAG SAVE] dag.draw(filename=...) TypeError: {e}")
    except Exception as e:
        print(f"[DAG SAVE] dag.draw(filename=...) failed: {e}")

    # 2) Try draw() and inspect the return
    try:
        obj = dag.draw()
        print(f"[DAG SAVE] dag.draw() returned: {type(obj)}")
        # Matplotlib figure
        if hasattr(obj, 'savefig'):
            try:
                obj.savefig(str(out_path))
                if plt is not None:
                    try:
                        plt.close(obj)
                    except Exception as ce:
                        print(f"[DAG SAVE] Warning closing figure: {ce}")
                print(f"[DAG SAVE] Saved DAG snapshot via fig.savefig to {out_path}")
                return
            except Exception as e:
                print(f"[DAG SAVE] fig.savefig failed: {e}")
        # Graphviz API (graphviz.Digraph or pydot/Source-like)
        # Try render
        if hasattr(obj, 'render'):
            try:
                obj.render(filename=str(out_path), cleanup=True, format=out_path.suffix.lstrip('.'))
                print(f"[DAG SAVE] Saved DAG snapshot via graphviz.render to {out_path}")
                return
            except Exception as e:
                print(f"[DAG SAVE] graphviz.render failed: {e}")
        # Try write
        if hasattr(obj, 'write'):
            try:
                obj.write(str(out_path))
                print(f"[DAG SAVE] Saved DAG snapshot via graphviz.write to {out_path}")
                return
            except Exception as e:
                print(f"[DAG SAVE] graphviz.write failed: {e}")
        # Try pipe to PNG bytes
        if hasattr(obj, 'pipe'):
            try:
                data = obj.pipe(format='png')
                out_path.write_bytes(data)
                print(f"[DAG SAVE] Saved DAG snapshot via graphviz.pipe to {out_path}")
                return
            except Exception as e:
                print(f"[DAG SAVE] graphviz.pipe failed: {e}")
        # Try source -> DOT file
        src = getattr(obj, 'source', None)
        if isinstance(src, str):
            try:
                out_dot = out_path.with_suffix('.dot')
                out_dot.write_text(src)
                print(f"[DAG SAVE] Saved DAG DOT via .source to {out_dot}")
                return
            except Exception as e:
                print(f"[DAG SAVE] writing DOT via .source failed: {e}")
        # If it's a string and looks like DOT, save it
        if isinstance(obj, str) and 'digraph' in obj:
            try:
                out_dot = out_path.with_suffix('.dot')
                out_dot.write_text(obj)
                print(f"[DAG SAVE] Saved DAG DOT (string) to {out_dot}")
                return
            except Exception as e:
                print(f"[DAG SAVE] writing DOT string failed: {e}")
    except Exception as e:
        print(f"[DAG SAVE] dag.draw() call failed: {e}")

    # 4) Last resort: save str(dag) to TXT
    try:
        out_txt = out_path.with_suffix('.txt')
        out_txt.write_text(str(dag))
        print(f"[DAG SAVE] Saved DAG TEXT (str(dag)) to {out_txt}")
        return
    except Exception as e:
        print(f"[DAG SAVE] writing str(dag) failed: {e}")

    print("[DAG SAVE] All snapshot save attempts failed.")


def _rehome_after_2q(graph: 'Graph', ion_a: int, ion_b: int, pz_name: str) -> None:
    """Set the 'home' PZ of the ion that moved to the other ion's home PZ."""
    home_a = graph.map_to_pz.get(ion_a)
    home_b = graph.map_to_pz.get(ion_b)
    # Only rehome if the executing PZ was one of the two homes
    if pz_name not in (home_a, home_b):
        return
    # Rehome the one whose home != pz_name
    if home_a == pz_name and home_b != pz_name:
        graph.map_to_pz[ion_b] = pz_name
    elif home_b == pz_name and home_a != pz_name:
        graph.map_to_pz[ion_a] = pz_name
    # else both already have same home -> nothing to do


def main(graph: Graph, dag: DAGDependency, cycle_or_paths: str, use_dag: bool, save_dag: bool = False) -> int:
    timestep = 0
    max_timesteps = 1e6
    graph.state = get_ions(graph)

    unique_folder = pathlib.Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure folders exist even if graph.save is False (for plot_state outputs)
    unique_folder.mkdir(exist_ok=True, parents=True)
    dag_folder = unique_folder / "dags"
    if save_dag:
        dag_folder.mkdir(exist_ok=True, parents=True)

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

    # Initial snapshot
    if save_dag and use_dag and dag is not None:
        try:
            _save_dag_snapshot(dag, dag_folder / f"dag_timestep_{timestep}.png")
        except Exception:
            pass

    for pz in graph.pzs:
        pz.time_in_pz_counter = 0
        pz.gate_execution_finished = True
        # NEW: track gate start time (t0) per PZ
        if not hasattr(pz, 'active_start_t'):
            pz.active_start_t = None

    graph.in_process = []

    if use_dag:
        next_processable_gate_nodes = get_all_first_gates_and_update_sequence_non_destructive(graph, dag)

    locked_gates: dict[tuple[int, ...], str] = {}
    graph.locked_gates = locked_gates
    while timestep < max_timesteps:
        print(f"Timestep {timestep}")
        for pz in graph.pzs:
            pz.rotate_entry = False
            pz.out_of_parking_cycle = None
            pz.out_of_parking_move = None

        if use_dag:
            gate_info_list: dict[str, list[int]] = {pz.name: [] for pz in graph.pzs}
            for pz_name, node in next_processable_gate_nodes.items():
                for ion in [q._index for q in node.qargs]:
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

        # # bug? this should not be necessary nor logical: Remove executing ions from priority queue so they are not moved this step
        # if getattr(graph, 'in_process', None):
        #     priority_queue = {ion: pz for ion, pz in priority_queue.items() if ion not in graph.in_process}

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
                gate = tuple(ion for ion in [q._index for q in gate_node.qargs])
                if len(gate) == 1:
                    ion = gate[0]
                    if get_idx_from_idc(graph.idc_dict, graph.state[ion]) == get_idx_from_idc(
                        graph.idc_dict, pz.parking_edge
                    ):
                        pz.gate_execution_finished = (
                            False  # set False, then check below if gate time is finished -> then True
                        )
                        # NEW: set t0 on first tick and pin ion
                        if pz.active_start_t is None:
                            pz.active_start_t = timestep
                        if ion not in graph.in_process:
                            graph.in_process.append(ion)
                        pz.getting_processed.append(gate_node)
                        pz.time_in_pz_counter += 1
                        gate_time_1q = GATE_TIME_1Q

                        if pz.time_in_pz_counter == gate_time_1q:
                            processed_nodes[pz_name] = gate_node
                            if gate_node in pz.getting_processed:
                                pz.getting_processed.remove(gate_node)
                            pz.time_in_pz_counter = 0
                            pz.gate_execution_finished = True
                elif len(gate) == 2:
                    ion1, ion2 = gate
                    state1 = graph.state[ion1]
                    state2 = graph.state[ion2]

                    # if both ions are in the processing zone, process the gate
                    if get_idx_from_idc(graph.idc_dict, state1) == get_idx_from_idc(
                        graph.idc_dict, pz.parking_edge
                    ) and get_idx_from_idc(graph.idc_dict, state2) == get_idx_from_idc(graph.idc_dict, pz.parking_edge):
                        pz.gate_execution_finished = False
                        # NEW: set t0 on first tick and pin ions
                        if pz.active_start_t is None:
                            pz.active_start_t = timestep
                        for ion in (ion1, ion2):
                            if ion not in graph.in_process:
                                graph.in_process.append(ion)
                        pz.getting_processed.append(gate_node)
                        pz.time_in_pz_counter += 1

                        gate_time_2q = GATE_TIME_2Q
                        if pz.time_in_pz_counter == gate_time_2q:
                            processed_nodes[pz_name] = gate_node
                            # # rehome the moved ion to the executing PZ - now done when selecting pz for 2-qubit gate in scheduling
                            _rehome_after_2q(graph, ion1, ion2, pz.name)
                            # remove the locked pz of the processed two-qubit gate
                            if gate in graph.locked_gates and graph.locked_gates[gate] == pz.name:
                                graph.locked_gates.pop(gate)
                            pz.time_in_pz_counter = 0
                            pz.gate_execution_finished = True
                            if gate_node in pz.getting_processed:
                                pz.getting_processed.remove(gate_node)
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

                            # NEW: set t0 on first tick and pin ion
                            if pz.active_start_t is None:
                                pz.active_start_t = timestep
                            if ion not in graph.in_process:
                                graph.in_process.append(ion)
                            pz.time_in_pz_counter += 1
                            gate_time_1q = GATE_TIME_1Q
                            if pz.time_in_pz_counter == gate_time_1q:
                                processed_ions.insert(0, (ion,))
                                ion_processed = True
                                # remove the processing zone from the list
                                # (it can only process one ion)
                                pzs.remove(pz)
                                # graph.in_process.append(ion)

                                pz.time_in_pz_counter = 0
                                pz.gate_execution_finished = True
                                # NEW: clear start time
                                pz.active_start_t = None
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

                            # NEW: set t0 first tick and pin ions
                            if pz.active_start_t is None:
                                pz.active_start_t = timestep
                            # for ion in (ion1, ion2):
                            #     if ion not in graph.in_process:
                            #         graph.in_process.append(ion)
                            pz.time_in_pz_counter += 1
                            gate_time_2q = GATE_TIME_2Q
                            if pz.time_in_pz_counter == gate_time_2q:
                                processed_ions.insert(0, (ion1, ion2))
                                ion_processed = True
                                # # rehome the moved ion to the executing PZ - now done when selecting pz for 2-qubit gate in scheduling
                                _rehome_after_2q(graph, ion1, ion2, pz.name)
                                # remove the processing zone from the list
                                # (it can only process one gate)
                                pzs.remove(pz)  # noqa: B909

                                # remove the locked pz of the processed two-qubit gate
                                if gate in graph.locked_gates and graph.locked_gates[gate] == pz.name:
                                    graph.locked_gates.pop(gate)
                                pz.time_in_pz_counter = 0
                                pz.gate_execution_finished = True
                                # NEW: clear start time
                                pz.active_start_t = None
                                break
                    else:
                        msg = "Invalid gate format"
                        raise ValueError(msg)
                previous_ion_processed = ion_processed

        # Remove processed ions from the sequence (and dag if use_dag)
        if use_dag:
            # expose processed_nodes for timeline collection
            if processed_nodes:
                execs = []
                for pz_name, gate_node in processed_nodes.items():
                    try:
                        pz = graph.pzs_name_map[pz_name]
                        gtype = (
                            getattr(getattr(gate_node, "op", None), "name", None)
                            or getattr(gate_node, "name", None)
                            or "OP"
                        )
                        qubits = list(getattr(gate_node, "qindices", [q._index for q in getattr(gate_node, "qargs", [])]))
                        duration = GATE_TIME_2Q if len(qubits) >= 2 else GATE_TIME_1Q
                        execs.append({
                            "id": f"t{timestep}_{pz_name}",
                            "type": gtype,
                            "qubits": qubits,
                            "pz": pz_name,
                            "edge_idc": pz.parking_edge,  # for placement
                            "duration": duration,
                            # NEW: include t0 from PZ state
                            "t0": int(pz.active_start_t) if isinstance(pz.active_start_t, int) else timestep - duration + 1,
                        })
                    except Exception:
                        continue
                    finally:
                        # NEW: reset start time after emitting
                        pz.active_start_t = None
                graph.executed_gates_next = execs
            else:
                graph.executed_gates_next = []
            if processed_nodes:
                remove_processed_gates(graph, dag, processed_nodes)
                next_processable_gate_nodes = get_all_first_gates_and_update_sequence_non_destructive(graph, dag)
                for pz_name, node in next_processable_gate_nodes.items():
                    locked_gates[tuple(q._index for q in node.qargs)] = pz_name
        else:
            # publish processed_ions with duration and t0 to the timeline collector
            if processed_ions:
                execs = []
                for i, g in enumerate(processed_ions):
                    duration = GATE_TIME_2Q if len(g) >= 2 else GATE_TIME_1Q
                    # Find pz for this gate's ions (last used above)
                    # We attach the first matching PZ where both ions (or single) are in parking
                    pz_used = None
                    for pz in graph.pzs:
                        try:
                            if len(g) == 1 and get_idx_from_idc(graph.idc_dict, graph.state[g[0]]) == get_idx_from_idc(graph.idc_dict, pz.parking_edge):
                                pz_used = pz
                                break
                            if len(g) == 2:
                                if all(get_idx_from_idc(graph.idc_dict, graph.state[q]) == get_idx_from_idc(graph.idc_dict, pz.parking_edge) for q in g):
                                    pz_used = pz
                                    break
                        except Exception:
                            continue
                    execs.append({
                        "id": f"t{timestep}_{i}",
                        "type": "OP",
                        "qubits": list(g),
                        "duration": duration,
                        **({"pz": getattr(pz_used, 'name', None)} if pz_used else {}),
                        **({"edge_idc": getattr(pz_used, 'parking_edge', None)} if pz_used else {}),
                        # Use pz.active_start_t if available
                        **({"t0": int(getattr(pz_used, 'active_start_t'))} if getattr(pz_used, 'active_start_t', None) is not None else {"t0": timestep - duration + 1}),
                    })
                    # reset start time on the used PZ
                    if pz_used is not None:
                        pz_used.active_start_t = None
                graph.executed_gates_next = execs
            else:
                graph.executed_gates_next = []
            for gate in processed_ions:
                graph.sequence.remove(gate)

        if len(graph.sequence) == 0:
            # Save DAG snapshot for final state as well
            try:
                if save_dag and use_dag and dag is not None:
                    _save_dag_snapshot(dag, dag_folder / f"dag_timestep_{timestep}.png")
            except Exception:
                pass
            break

        # Save DAG snapshot at the end of this timestep
        try:
            if save_dag and use_dag and dag is not None:
                _save_dag_snapshot(dag, dag_folder / f"dag_timestep_{timestep}.png")
                print("Saved DAG snapshot.")
        except Exception:
            pass

        timestep += 1

    return timestep
