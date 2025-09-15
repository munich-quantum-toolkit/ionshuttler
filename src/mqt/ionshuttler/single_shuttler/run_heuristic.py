import argparse
import json
import pathlib
import time

import numpy as np
from cycles import GraphCreator, MemoryZone
from scheduling import create_initial_sequence, create_starting_config, run_simulation


def run_simulation_for_architecture(
    arch, seeds, pz, max_timesteps, time_1qubit_gate=1, time_2qubit_gate=3, max_chains_in_parking=3, compilation=True
):
    """
    Runs simulations for the given architecture and seeds, logs the results.

    Args:
        arch (list): Architecture parameters.
        seeds (list): List of seed values.
        pz (str): Position of Processing zone.
        max_timesteps (int): Maximum timesteps.
        compilation (bool): Compilation flag (Gate Selection Step).

    Returns:
        tuple: (timestep_arr, cpu_time_arr, number_of_registers, n_of_traps, seq_length)
    """
    timestep_arr = []
    cpu_time_arr = []
    start_time = time.time()

    for seed in seeds:
        m, n, v, h = arch
        graph = GraphCreator(m, n, v, h, pz).get_graph()
        try:
            ion_chains, number_of_registers = create_starting_config(num_ion_chains, graph, seed=seed)
        except:  # noqa: E722
            continue
        print(f"ion chains: {ion_chains}, number of registers: {number_of_registers}")
        print(f"arch: {arch}, seed: {seed}, registers: {number_of_registers}\n")

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
        timestep = run_simulation(memorygrid, max_timesteps, seq, flat_seq, dag_dep, next_node_initial, max_length=10)
        timestep_arr.append(timestep)
        cpu_time = time.time() - start_time
        cpu_time_arr.append(cpu_time)

    return timestep_arr, cpu_time_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to json config file")
    # parser.add_argument("--plot", action="store_true", help="plot grid")
    args = parser.parse_args()

    with pathlib.Path(args.config_file).open("r", encoding="utf-8") as f:
        config = json.load(f)
    arch = config["arch"]
    max_timesteps = config["max_timesteps"]
    num_ion_chains = config["num_ion_chains"]
    filename = config["qu_alg"]

    seeds = [0]
    pz = "outer"

    timestep_arr, cpu_time_arr = run_simulation_for_architecture(arch, seeds, pz, max_timesteps)
    print(f"CPU time: {np.mean(cpu_time_arr)} s")
