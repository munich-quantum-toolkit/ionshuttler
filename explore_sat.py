import time

from src.mqt.ionshuttler.single_shuttler.SAT import MemorySAT, create_graph


def simulate_single_shuttler(plot: bool, arch: list[int], max_timesteps: int, num_ion_chains: int, qu_alg: list) -> int:
    m, n, v, h = arch
    graph = create_graph(m, n, v, h)

    # Select trap edges
    trap_edges = [e for e in graph.edges() if graph.get_edge_data(*e)["edge_type"] == "trap"]
    starting_traps = trap_edges[:num_ion_chains]  # deterministic version

    ions = list(range(num_ion_chains))
    for ion, edge in zip(ions, starting_traps, strict=False):
        graph[edge[0]][edge[1]]["ion_chain"] = ion

    start = time.time()
    for timesteps in range(2, max_timesteps + 1):
        print(f"{time.time() - start:.1f}s Checking {timesteps} timesteps...", end="")
        sat = MemorySAT(graph, h, v, ions, timesteps)
        MemorySAT.create_constraints(sat, starting_traps)
        if MemorySAT.evaluate(sat, qu_alg, len(starting_traps)):
            print(f" Found solution at {timesteps} steps.")
            if plot:
                MemorySAT.plot(sat, show_ions=True)
            return timesteps
        print(" no solution.")

    print(f"{time.time() - start:.1f}s No solution up to {max_timesteps} steps.")
    return -1


if __name__ == "__main__":
    arch = [3, 3, 2, 2]  # example: 3x3 grid, 2 vertical/horizontal trap capacity
    max_ts = 20
    num_ions = 4
    qu_alg = [(0, 2), 1]  # simple example algorithm
    ts = simulate_single_shuttler(plot=True, arch=arch, max_timesteps=max_ts, num_ion_chains=num_ions, qu_alg=qu_alg)
    print(f"Finished in {ts} time steps.")
