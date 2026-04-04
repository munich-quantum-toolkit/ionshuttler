from __future__ import annotations

import pathlib
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

import networkx as nx

from .outside.compilation import create_dag, create_initial_sequence, create_updated_sequence_destructive
from .outside.cycles import create_starting_config, get_ions
from .outside.graph_creator import GraphCreator, PZCreator
from .outside.partition import get_partition
from .outside.processing_zone import ProcessingZone
from .outside.shuttle import main as run_shuttle_main

if TYPE_CHECKING:
    from collections.abc import Mapping


def validate_conflict_resolution_mode(config: Mapping[str, Any]) -> str:
    """Validate and normalize the conflict-resolution mode from config.

    Args:
        config: Parsed user configuration.

    Returns:
        Normalized mode string (`"cycles"`, `"paths"`, or `"hybrid"`).

    Raises:
        TypeError: If the value is present but not a string.
        ValueError: If the value is a string but unsupported.
    """
    allowed_modes = {"cycles", "paths", "hybrid"}

    if "use_cycle_or_paths" not in config:
        return "cycles"

    raw_mode = config["use_cycle_or_paths"]
    if not isinstance(raw_mode, str):
        msg = "Config parameter 'use_cycle_or_paths' must be a string and one of: 'cycles', 'paths', or 'hybrid'."
        raise TypeError(msg)

    mode = raw_mode.strip().lower()
    if mode not in allowed_modes:
        msg = f"Invalid value for 'use_cycle_or_paths': {raw_mode!r}. Allowed values are: 'cycles', 'paths', 'hybrid'."
        raise ValueError(msg)

    return mode


def main(config: dict[str, Any]) -> int:
    # --- Extract Parameters from Config ---
    arch = config.get("arch")
    num_pzs_config = config.get("num_pzs", 1)
    seed = config.get("seed", 0)
    algorithm_name = config.get("algorithm_name")

    # Extract Ion Count Parameters
    perc_num_ions_raw = config.get("perc_num_ions")
    abs_num_ions_raw = config.get("abs_num_ions")

    perc_num_ions: float | None
    abs_num_ions: int | None

    if perc_num_ions_raw is None:
        perc_num_ions = None
    elif isinstance(perc_num_ions_raw, (int, float)):
        perc_num_ions = float(perc_num_ions_raw)
    else:
        msg = "Config parameter 'perc_num_ions' must be a number."
        raise ValueError(msg)

    if abs_num_ions_raw is None:
        abs_num_ions = None
    elif isinstance(abs_num_ions_raw, int):
        abs_num_ions = abs_num_ions_raw
    else:
        msg = "Config parameter 'abs_num_ions' must be an integer."
        raise ValueError(msg)

    use_dag = config.get("use_dag", True)
    use_cycle_or_paths = validate_conflict_resolution_mode(config)
    pz_assignment_policy = config.get("pz_assignment_policy", "legacy")
    max_timesteps = config.get("max_timesteps", 1_000_000)
    plot_flag = config.get("plot", False)
    save_flag = config.get("save", False)
    failing_junctions = config.get("failing_junctions", 0)
    parameter = config.get("parameter", 1)  # For hybrid cost function, if used

    # Define base path for QASM files if needed
    qasm_base_dir_string = config.get("qasm_base_dir")
    if qasm_base_dir_string is None:
        qasm_base_dir = pathlib.Path(__file__).absolute().parent.parent.parent.parent / "inputs" / "qasm_files"
    else:
        qasm_base_dir = pathlib.Path(qasm_base_dir_string)

    # --- Validate Config ---
    if arch is None:
        msg = "Config parameter 'arch' is required but not set"
        raise ValueError(msg)

    if algorithm_name is None:
        msg = "Config parameter 'algorithm_name' is required but not set"
        raise ValueError(msg)

    # Validate Mutual Exclusivity of Ion Count Parameters
    if perc_num_ions is None and abs_num_ions is None:
        perc_num_ions = 0.5
    elif perc_num_ions is not None and abs_num_ions is not None:
        msg = "Config must specify exactly one of: 'perc_num_ions' or 'abs_num_ions'."
        raise ValueError(msg)

    if not isinstance(arch, list) or len(arch) != 4:
        msg = "Config parameter 'arch' must be a list of 4 integers [m, n, v, h]"
        raise ValueError(msg)

    # --- Setup ---
    start_time = datetime.now()

    m, n, v, h = arch

    # --- PZ Definitions ---
    height = -4.5
    pz_definitions = {
        "pz1": ProcessingZone(
            "pz1",
            [
                (float((m - 1) * v), float((n - 1) * h)),
                (float((m - 1) * v), float(0)),
                (float((m - 1) * v - height), float((n - 1) * h / 2)),
            ],
        ),
        "pz2": ProcessingZone("pz2", [(0.0, 0.0), (0.0, float((n - 1) * h)), (float(height), float((n - 1) * h / 2))]),
        "pz3": ProcessingZone(
            "pz3", [(float((m - 1) * v), float(0)), (float(0), float(0)), (float((m - 1) * v / 2), float(height))]
        ),
        "pz4": ProcessingZone(
            "pz4",
            [
                (float(0), float((n - 1) * h)),
                (float((m - 1) * v), float((n - 1) * h)),
                (float((m - 1) * v / 2), float((n - 1) * h - height)),
            ],
        ),
    }
    available_pz_names = list(pz_definitions.keys())
    pzs_to_use = [pz_definitions[name] for name in available_pz_names[:num_pzs_config]]

    if not pzs_to_use:
        print(f"Error: num_pzs ({num_pzs_config}) is invalid or results in no PZs selected.")
        sys.exit(1)

    print(f"Using {len(pzs_to_use)} PZs: {[pz.name for pz in pzs_to_use]}")
    print(f"Architecture: {arch}, Seed: {seed}")
    print(f"Algorithm: {algorithm_name}")
    print(f"DAG-Compilation: {use_dag}, Conflict Resolution: {use_cycle_or_paths}")

    # --- Graph Creation ---
    basegraph_creator = GraphCreator(m, n, v, h, failing_junctions, pzs_to_use, seed)
    mz_graph = basegraph_creator.get_graph()
    pzgraph_creator = PZCreator(m, n, v, h, failing_junctions, pzs_to_use, seed)
    graph = pzgraph_creator.get_graph()
    graph.mz_graph = mz_graph  # Attach MZ graph for BFS lookups if needed by Cycles/Paths

    graph.seed = seed
    graph.max_num_parking = 2
    graph.pzs = pzs_to_use  # List of ProcessingZone objects
    graph.max_timesteps = max_timesteps

    graph.plot = plot_flag
    graph.save = save_flag
    graph.arch = str(arch)  # For plotting/logging
    graph.pz_assignment_policy = pz_assignment_policy
    print(f"PZ assignment policy: {graph.pz_assignment_policy}")

    graph.parameter = parameter  # For hybrid cost function, if used

    # --- Calculate Absolute Number of Ions ---
    if abs_num_ions is not None:
        num_ions = abs_num_ions
        print(f"Targeting absolute number of ions: {num_ions}")
    else:
        # here perc_num_ions is guaranteed not None by earlier validation logic
        assert perc_num_ions is not None
        trap_edges = [edges for edges in graph.edges() if nx.get_edge_attributes(graph, "edge_type")[edges] == "trap"]
        num_ions = round(perc_num_ions * len(trap_edges))
        print(f"Targeting percentage: {perc_num_ions * 100}% -> {num_ions} ions")

    print(f"Number of edges in current graph: {len(graph.edges())}")

    qasm_file_path = qasm_base_dir / algorithm_name / f"{algorithm_name}_{num_ions}.qasm"

    if not qasm_file_path.is_file():
        print(f"Error: QASM file not found at {qasm_file_path}")
        sys.exit(1)

    # --- Initial State & Sequence ---
    create_starting_config(graph, num_ions, seed=seed)
    graph.state = get_ions(graph)  # Get initial state {ion: edge_idc}

    graph.sequence = create_initial_sequence(qasm_file_path)
    seq_length = len(graph.sequence)
    print(f"Number of Gates: {seq_length}")

    # --- Partitioning ---
    partitioning = True  # Make configurable
    partitions: dict[str, list[int]] = {}
    if partitioning:
        part = get_partition(qasm_file_path, len(graph.pzs))
        # Ensure partition list length matches num_pzs
        if len(part) != len(graph.pzs):
            print(f"Warning: Partitioning returned {len(part)} parts, but expected {len(graph.pzs)}. Adjusting...")
            if len(part) < len(graph.pzs):
                print("Error: Partitioning failed to produce enough parts.")
                sys.exit(1)
            else:  # More parts than PZs, merge extra parts into the last ones
                merged = [qubit for sublist in part[len(graph.pzs) - 1 :] for qubit in sublist]
                part = [*part[: len(graph.pzs) - 1], merged]

        partitions = {pz.name: part[i] for i, pz in enumerate(graph.pzs)}
        print(f"Partitions: {partitions}")
    else:
        msg = "Disabling Partitioning has to be implemented. For now, only example for random_connecting 22 ions and 2 PZs."
        raise NotImplementedError(msg)

    # Create reverse map and validate partition
    map_to_pz: dict[int, str] = {}
    all_partition_elements = []
    for pz_name, elements in partitions.items():
        all_partition_elements.extend(elements)
        for element in elements:
            if element in map_to_pz:
                print(
                    f"Warning: Qubit {element} assigned to multiple partitions ({map_to_pz[element]}, {pz_name}). Check partitioning logic."
                )
            map_to_pz[element] = pz_name
    graph.map_to_pz = map_to_pz

    # Validation
    unique_sequence_qubits = {item for sublist in graph.sequence for item in sublist}
    missing_qubits = unique_sequence_qubits - set(all_partition_elements)
    if missing_qubits:
        print(f"Error: Qubits {missing_qubits} from sequence are not in any partition.")
        sys.exit(1)

    # --- DAG-Compilation Setup (if enabled) ---
    dag = None
    if use_dag:
        try:
            for pz in graph.pzs:
                pz.getting_processed = []
            dag = create_dag(qasm_file_path)
            graph.locked_gates = {}
            dag.copy()  # Keep a copy of the original DAG if needed later
            # Initial DAG-based sequence update
            sequence, _, dag = create_updated_sequence_destructive(graph, qasm_file_path, dag, use_dag=True)
            graph.sequence = sequence

        except Exception as e:
            print(f"Error during DAG creation or initial sequence update: {e}")
            print("Falling back to non-compiled sequence.")
            use_dag = False  # Disable use_dag if setup fails
            dag = None
            graph.sequence = create_initial_sequence(qasm_file_path)  # Revert to basic sequence
    else:
        print("DAG disabled, using static QASM sequence.")

    # --- Run Simulation ---

    # Initialize PZ states
    for pz in graph.pzs:
        pz.getting_processed = []  # Track nodes being processed by this PZ

    print("\nStarted shuttling simulation...")

    def _assert_idc_consistency(graph: Any) -> None:
        missing: list[tuple[Any, Any]] = []
        for u, v in graph.edges():
            if (u, v) not in graph.idc_dict and (v, u) not in graph.idc_dict:
                missing.append((u, v))
                if len(missing) >= 5:
                    break

        if missing:
            print("IDC MISSING EDGES (showing up to 5):", missing)
            msg = "idc_dict does not cover current graph edges."
            raise RuntimeError(msg)

    _assert_idc_consistency(graph)

    # Run the main shuttling logic
    final_timesteps = run_shuttle_main(graph, dag, use_cycle_or_paths, use_dag=use_dag)

    # --- Results ---
    end_time = datetime.now()
    cpu_time = end_time - start_time

    print(f"\nSimulation finished in {final_timesteps} timesteps.")
    print(f"Total CPU time: {cpu_time}")

    return final_timesteps
