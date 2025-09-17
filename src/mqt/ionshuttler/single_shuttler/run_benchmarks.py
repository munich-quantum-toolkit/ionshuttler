import math
import time
from pathlib import Path

import numpy as np

from .cycles import MemoryZone
from .graph_utils import GraphCreator
from .scheduling import create_initial_sequence, create_starting_config, run_simulation


def run_simulation_for_architecture(
    arch: list[int], seeds: list[int], pz: str, max_timesteps: int, compilation: bool = True
) -> tuple[list[int], list[float], int, int, int]:
    """
    Runs simulations for the given architecture and seeds, logs the results.

    Args:
        arch: Architecture parameters.
        seeds: List of seed values.
        pz: Position of Processing zone.
        max_timesteps: Maximum timesteps.
        compilation: Compilation flag (Gate Selection Step).

    Returns:
        (timestep_arr, cpu_time_arr, number_of_registers, n_of_traps, seq_length)
    """
    timestep_arr: list[int] = []
    cpu_time_arr: list[float] = []
    start_time = time.time()
    number_of_registers = 0
    n_of_traps = 0
    seq_length = 0
    for seed in seeds:
        m, n, v, h = arch
        graph = GraphCreator(m, n, v, h, pz).get_graph()
        n_of_traps = len([
            trap for trap in graph.edges() if graph.get_edge_data(trap[0], trap[1])["edge_type"] == "trap"
        ])
        num_ion_chains = math.ceil(n_of_traps / 2)
        try:
            ion_chains, number_of_registers = create_starting_config(num_ion_chains, graph, seed=seed)
        except Exception:
            continue
        print(f"ion chains: {ion_chains}, number of registers: {number_of_registers}")
        filename = (
            Path(__file__).absolute().parent.parent.parent.parent.parent
            / f"inputs/qasm_files/qft_no_swaps_nativegates_quantinuum_tket/qft_no_swaps_nativegates_quantinuum_tket_{num_ion_chains}.qasm"
        )
        print(f"arch: {arch}, seed: {seed}, registers: {number_of_registers}\n")

        time_2qubit_gate = 3
        time_1qubit_gate = 1
        max_chains_in_parking = 3

        memorygrid = MemoryZone(
            m,
            n,
            v,
            h,
            ion_chains,
            max_timesteps,
            max_chains_in_parking,
            pz,
            time_2qubit_gate=time_2qubit_gate,
            time_1qubit_gate=time_1qubit_gate,
        )

        memorygrid.update_distance_map()
        seq, flat_seq, dag_dep, next_node_initial = create_initial_sequence(
            memorygrid.distance_map, filename, compilation=compilation
        )
        seq_length = len(seq)
        print(f"seq: {seq}")
        timestep = run_simulation(memorygrid, max_timesteps, seq, flat_seq, dag_dep, next_node_initial, max_length=10)
        timestep_arr.append(timestep)
        cpu_time = time.time() - start_time
        cpu_time_arr.append(cpu_time)

    return timestep_arr, cpu_time_arr, number_of_registers, n_of_traps, seq_length


def log_results(
    arch: list[int],
    timestep_arr: list[int],
    cpu_time_arr: list[float],
    number_of_registers: int,
    n_of_traps: int,
    seq_length: int,
    compilation: bool = True,
) -> None:
    """
    Logs the results of the simulation to a file.

    Args:
        arch: Architecture parameters.
        timestep_arr: List of timesteps.
        cpu_time_arr: List of CPU times.
        number_of_registers: Number of registers.
        n_of_traps: Number of traps.
        seq_length: Sequence length.
        compilation: Compilation flag (Gate Selection Step).
    """
    timestep_mean = np.mean(timestep_arr)
    timestep_var = np.var(timestep_arr)
    cpu_time_mean = np.mean(cpu_time_arr)
    print(cpu_time_mean)
    print(f"timestep mean: {timestep_mean}, timestep var: {timestep_var}, cpu time mean: {cpu_time_mean}")

    # file_path = Path("results.txt")
    # try:
    #     with file_path.open("a") as file:
    #         line = (
    #             f"& {arch[0]} {arch[1]} {arch[2]} {arch[3]} & {number_of_registers}/{n_of_traps} & {seq_length} "
    #             f"& {timestep_mean} & {cpu_time_mean} s & Gate Selection={compilation} \\\\"
    #         )
    #         file.write(f"array ts: {timestep_arr}\n" + line + "\n\n")
    # except:
    #     pass


def main() -> None:
    archs: list[list[int]] = [[3, 3, 1, 1]]
    seeds: list[int] = [0]
    pz: str = "outer"
    max_timesteps: int = 10000000
    compilation: bool = True
    for arch in archs:
        timestep_arr, cpu_time_arr, number_of_registers, n_of_traps, seq_length = run_simulation_for_architecture(
            arch, seeds, pz, max_timesteps, compilation=compilation
        )
        log_results(
            arch, timestep_arr, cpu_time_arr, number_of_registers, n_of_traps, seq_length, compilation=compilation
        )


if __name__ == "__main__":
    main()
