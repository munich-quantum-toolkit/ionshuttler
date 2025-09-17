import random
import sys
import time
from typing import Any

from .memory_sat import MemorySAT, create_graph


def main(config: dict[str, Any], plot: bool = False) -> None:
    arch = config["arch"]
    max_timesteps = config["max_timesteps"]
    num_ion_chains = config["num_ion_chains"]
    qu_alg = [(q[0] if len(q) == 1 else tuple(q)) for q in config["qu_alg"]]

    # create graph
    m, n = arch[0], arch[1]
    ion_chain_size_vertical = arch[2]
    ion_chain_size_horizontal = arch[3]
    graph = create_graph(m, n, ion_chain_size_vertical, ion_chain_size_horizontal)
    n_of_traps = len([trap for trap in graph.edges() if graph.get_edge_data(trap[0], trap[1])["edge_type"] == "trap"])

    # starting edges / ions
    rand = False
    if rand is True:
        random.seed(0)
        random_starting_traps = random.sample(range(n_of_traps), (num_ion_chains))
        starting_traps = []
        for trap in random_starting_traps:
            starting_traps.append(
                [edges for edges in graph.edges() if graph.get_edge_data(edges[0], edges[1])["edge_type"] == "trap"][
                    trap
                ]
            )
    else:
        starting_traps = [
            edges for edges in graph.edges() if graph.get_edge_data(edges[0], edges[1])["edge_type"] == "trap"
        ][:num_ion_chains]
    number_of_registers = len(starting_traps)

    # place ions onto traps (ion0 on starting_trap0)
    ions = []
    for ion, idx in enumerate(starting_traps):
        graph[idx[0]][idx[1]]["ion_chain"] = ion
        ions.append(ion)
    start = time.time()
    for timesteps in range(2, max_timesteps + 1):
        print(f"{time.time() - start:.1f}s Checking for {timesteps} timesteps... ", end="")
        sat = MemorySAT(graph, ion_chain_size_horizontal, ion_chain_size_vertical, ions, timesteps)
        sat.create_constraints(starting_traps)
        is_satisfied = sat.evaluate(qu_alg, number_of_registers)

        if is_satisfied:
            print(f"{time.time() - start:.1f}s Found satisfying solution with {timesteps} time steps.")
            if plot:
                sat.plot(show_ions=True)
            break
    else:
        print(f"{time.time() - start:.1f}s Reached {max_timesteps} time steps without a satisfying solution.")
        sys.exit(1)
