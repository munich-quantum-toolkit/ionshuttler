from __future__ import annotations

import contextlib
import pathlib
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

# Keep all imports at module top; optional deps are handled via try/except imports.
try:
    import matplotlib as mpl  # ICN001
    import matplotlib.pyplot as plt  # ICN001

    mpl.use("Agg")
except Exception:  # pragma: no cover
    mpl = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

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
    from .processing_zone import ProcessingZone
    from .types import Edge

# Gate time configuration (overridable from run scripts)
GATE_TIME_1Q = 1
GATE_TIME_2Q = 3
REHOME = True


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


def _save_dag_snapshot(dag: DAGDependency, out_path: pathlib.Path) -> None:
    """Best-effort save of a DAG snapshot with detailed debug prints."""
    print(f"[DAG SAVE] Attempting to save snapshot to {out_path}")

    saved = False

    # Prefer filename API if it exists
    try:
        dag.draw(filename=str(out_path))
    except TypeError as e:
        print(f"[DAG SAVE] dag.draw(filename=...) TypeError: {e}")
    except Exception as e:
        print(f"[DAG SAVE] dag.draw(filename=...) failed: {e}")
    else:
        print(f"[DAG SAVE] Saved DAG snapshot via dag.draw(filename=...) to {out_path}")
        saved = True

    if saved:
        return

    # Fallback: draw() without filename
    try:
        obj = dag.draw()
        print(f"[DAG SAVE] dag.draw() returned: {type(obj)}")

        if hasattr(obj, "savefig"):
            try:
                obj.savefig(str(out_path))
                if plt is not None:
                    with contextlib.suppress(Exception):
                        plt.close(obj)
                print(f"[DAG SAVE] Saved DAG snapshot via fig.savefig to {out_path}")
                saved = True
            except Exception as e:
                print(f"[DAG SAVE] fig.savefig failed: {e}")

        if not saved and hasattr(obj, "render"):
            try:
                obj.render(filename=str(out_path), cleanup=True, format=out_path.suffix.lstrip("."))
                print(f"[DAG SAVE] Saved DAG snapshot via graphviz.render to {out_path}")
                saved = True
            except Exception as e:
                print(f"[DAG SAVE] graphviz.render failed: {e}")

        if not saved and hasattr(obj, "write"):
            try:
                obj.write(str(out_path))
                print(f"[DAG SAVE] Saved DAG snapshot via graphviz.write to {out_path}")
                saved = True
            except Exception as e:
                print(f"[DAG SAVE] graphviz.write failed: {e}")

        if not saved and hasattr(obj, "pipe"):
            try:
                data = obj.pipe(format="png")
                out_path.write_bytes(data)
                print(f"[DAG SAVE] Saved DAG snapshot via graphviz.pipe to {out_path}")
                saved = True
            except Exception as e:
                print(f"[DAG SAVE] graphviz.pipe failed: {e}")

        if not saved:
            src = getattr(obj, "source", None)
            if isinstance(src, str):
                try:
                    out_dot = out_path.with_suffix(".dot")
                    out_dot.write_text(src)
                    print(f"[DAG SAVE] Saved DAG DOT via .source to {out_dot}")
                    saved = True
                except Exception as e:
                    print(f"[DAG SAVE] writing DOT via .source failed: {e}")

        if not saved and isinstance(obj, str) and "digraph" in obj:
            try:
                out_dot = out_path.with_suffix(".dot")
                out_dot.write_text(obj)
                print(f"[DAG SAVE] Saved DAG DOT (string) to {out_dot}")
                saved = True
            except Exception as e:
                print(f"[DAG SAVE] writing DOT string failed: {e}")

    except Exception as e:
        print(f"[DAG SAVE] dag.draw() call failed: {e}")

    if saved:
        return

    try:
        out_txt = out_path.with_suffix(".txt")
        out_txt.write_text(str(dag))
        print(f"[DAG SAVE] Saved DAG TEXT (str(dag)) to {out_txt}")
        saved = True
    except Exception as e:
        print(f"[DAG SAVE] writing str(dag) failed: {e}")

    if not saved:
        print("[DAG SAVE] All snapshot save attempts failed.")


def _rehome_after_2q(graph: Graph, ion_a: int, ion_b: int, pz_name: str) -> None:
    """Set the 'home' PZ of the ion that moved to the other ion's home PZ."""
    home_a = graph.map_to_pz.get(ion_a)
    home_b = graph.map_to_pz.get(ion_b)
    if pz_name not in {home_a, home_b}:
        return
    if home_a == pz_name and home_b != pz_name:
        graph.map_to_pz[ion_b] = pz_name
    elif home_b == pz_name and home_a != pz_name:
        graph.map_to_pz[ion_a] = pz_name


def main(graph: Graph, dag: DAGDependency, cycle_or_paths: str, use_dag: bool, save_dag: bool = False) -> int:
    timestep = 0
    max_timesteps = 1_000_000
    graph.state = get_ions(graph)

    unique_folder = pathlib.Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_folder.mkdir(exist_ok=True, parents=True)
    dag_folder = unique_folder / "dags"
    if save_dag:
        dag_folder.mkdir(exist_ok=True, parents=True)

    for pz in graph.pzs:
        pz.time_in_pz_counter = 0
        pz.gate_execution_finished = True
        # NEW: track gate start time (t0) per PZ
        pz.active_start_t = None

    graph.in_process = []

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

    if save_dag and use_dag and dag is not None:
        with contextlib.suppress(Exception):
            _save_dag_snapshot(dag, dag_folder / f"dag_timestep_{timestep}.png")

    for pz in graph.pzs:
        pz.time_in_pz_counter = 0
        pz.gate_execution_finished = True
        # NEW: track gate start time (t0) per PZ
        # Removed redundant hasattr check since class definition now ensures it
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
            gate_info_list = create_gate_info_list(graph)

        pz_executing_gate_order = find_pz_order(graph, gate_info_list)
        graph.locked_gates = locked_gates
        priority_queue, next_gate_at_pz_dict = create_priority_queue(graph, pz_executing_gate_order)

        for i in range(min(len(graph.pzs), len(graph.sequence))):
            gate = graph.sequence[i]

            if len(gate) == 2:
                ion1, ion2 = gate
                for pz in graph.pzs:
                    state1 = graph.state[ion1]
                    state2 = graph.state[ion2]
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

        shuttle(graph, priority_queue, timestep, cycle_or_paths, unique_folder)

        graph.in_process = []
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
                        pz.gate_execution_finished = False
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

                    if get_idx_from_idc(graph.idc_dict, state1) == get_idx_from_idc(
                        graph.idc_dict, pz.parking_edge
                    ) and get_idx_from_idc(graph.idc_dict, state2) == get_idx_from_idc(graph.idc_dict, pz.parking_edge):
                        pz.gate_execution_finished = False
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
                            if REHOME:
                                _rehome_after_2q(graph, ion1, ion2, pz.name)
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
            for i in range(min(len(graph.pzs), len(graph.sequence))):
                if not previous_ion_processed:
                    break
                gate = next_gates[i]
                ion_processed = False
                pz_to_remove = None
                for pz in list(pzs):
                    if len(gate) == 1:
                        ion = gate[0]
                        if get_idx_from_idc(graph.idc_dict, graph.state[ion]) == get_idx_from_idc(
                            graph.idc_dict, pz.parking_edge
                        ):
                            pz.gate_execution_finished = False

                            if pz.active_start_t is None:
                                pz.active_start_t = timestep
                            if ion not in graph.in_process:
                                graph.in_process.append(ion)
                            pz.time_in_pz_counter += 1
                            gate_time_1q = GATE_TIME_1Q
                            if pz.time_in_pz_counter == gate_time_1q:
                                processed_ions.insert(0, (ion,))
                                ion_processed = True
                                pz_to_remove = pz
                                pz.time_in_pz_counter = 0
                                pz.gate_execution_finished = True
                                pz.active_start_t = None
                                break
                    elif len(gate) == 2:
                        ion1, ion2 = gate
                        state1 = graph.state[ion1]
                        state2 = graph.state[ion2]

                        if get_idx_from_idc(graph.idc_dict, state1) == get_idx_from_idc(
                            graph.idc_dict, pz.parking_edge
                        ) and get_idx_from_idc(graph.idc_dict, state2) == get_idx_from_idc(
                            graph.idc_dict, pz.parking_edge
                        ):
                            pz.gate_execution_finished = False

                            if pz.active_start_t is None:
                                pz.active_start_t = timestep
                            pz.time_in_pz_counter += 1
                            gate_time_2q = GATE_TIME_2Q
                            if pz.time_in_pz_counter == gate_time_2q:
                                processed_ions.insert(0, (ion1, ion2))
                                ion_processed = True
                                if REHOME:
                                    _rehome_after_2q(graph, ion1, ion2, pz.name)
                                pz_to_remove = pz
                                if gate in graph.locked_gates and graph.locked_gates[gate] == pz.name:
                                    graph.locked_gates.pop(gate)
                                pz.time_in_pz_counter = 0
                                pz.gate_execution_finished = True
                                pz.active_start_t = None
                                break
                    else:
                        msg = "Invalid gate format"
                        raise ValueError(msg)
                if pz_to_remove is not None:
                    with contextlib.suppress(ValueError):
                        pzs.remove(pz_to_remove)
                previous_ion_processed = ion_processed

        if use_dag:
            if processed_nodes:
                execs = []
                for pz_name, gate_node in processed_nodes.items():
                    try:
                        pz_obj = graph.pzs_name_map.get(pz_name)
                        start_t: int | None = None
                        if pz_obj is not None:
                            start_t = pz_obj.active_start_t
                            pz_obj.active_start_t = None
                        gtype = (
                            getattr(getattr(gate_node, "op", None), "name", None)
                            or getattr(gate_node, "name", None)
                            or "OP"
                        )
                        qubits = list(
                            getattr(gate_node, "qindices", [q._index for q in getattr(gate_node, "qargs", [])])
                        )
                        duration = GATE_TIME_2Q if len(qubits) >= 2 else GATE_TIME_1Q
                        t0_val = start_t if isinstance(start_t, int) else max(0, timestep - (duration - 1))
                        execs.append({
                            "id": f"t{timestep}_{pz_name}",
                            "type": gtype,
                            "qubits": qubits,
                            "pz": pz_name,
                            "edge_idc": getattr(pz_obj, "parking_edge", None) if pz_obj else None,
                            "duration": duration,
                            "t0": t0_val,
                        })
                    except Exception:
                        continue
                graph.executed_gates_next = execs
            else:
                graph.executed_gates_next = []
            if processed_nodes:
                remove_processed_gates(graph, dag, processed_nodes)
                next_processable_gate_nodes = get_all_first_gates_and_update_sequence_non_destructive(graph, dag)
                for pz_name, node in next_processable_gate_nodes.items():
                    locked_gates[tuple(q._index for q in node.qargs)] = pz_name
        else:
            if processed_ions:
                execs = []
                for i, g in enumerate(processed_ions):
                    duration = GATE_TIME_2Q if len(g) >= 2 else GATE_TIME_1Q
                    # Explicitly type pz_used as ProcessingZone or None
                    pz_used: ProcessingZone | None = None
                    for pz in graph.pzs:
                        try:
                            if len(g) == 1 and get_idx_from_idc(graph.idc_dict, graph.state[g[0]]) == get_idx_from_idc(
                                graph.idc_dict, pz.parking_edge
                            ):
                                pz_used = pz
                                break
                            if len(g) == 2 and all(
                                get_idx_from_idc(graph.idc_dict, graph.state[q])
                                == get_idx_from_idc(graph.idc_dict, pz.parking_edge)
                                for q in g
                            ):
                                pz_used = pz
                                break
                        except Exception:
                            continue

                    # Now we can access active_start_t directly because pz_used is typed
                    start_t2: int | None = pz_used.active_start_t if pz_used is not None else None
                    t0_val2 = start_t2 if isinstance(start_t2, int) else timestep - duration + 1

                    execs.append({
                        "id": f"t{timestep}_{i}",
                        "type": "OP",
                        "qubits": list(g),
                        "duration": duration,
                        **({"pz": getattr(pz_used, "name", None)} if pz_used else {}),
                        **({"edge_idc": getattr(pz_used, "parking_edge", None)} if pz_used else {}),
                        "t0": t0_val2,
                    })
                    if pz_used is not None:
                        pz_used.active_start_t = None
                graph.executed_gates_next = execs
            else:
                graph.executed_gates_next = []
            for gate in processed_ions:
                graph.sequence.remove(gate)

        if len(graph.sequence) == 0:
            try:
                if save_dag and use_dag and dag is not None:
                    _save_dag_snapshot(dag, dag_folder / f"dag_timestep_{timestep}.png")
            except Exception:
                pass
            break

        try:
            if save_dag and use_dag and dag is not None:
                _save_dag_snapshot(dag, dag_folder / f"dag_timestep_{timestep}.png")
                print("Saved DAG snapshot.")
        except Exception:
            pass

        timestep += 1

    return timestep
