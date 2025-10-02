import math

from src.mqt.ionshuttler.multi_shuttler.Outside.compilation import (
    create_dag,
    create_dist_dict,
    create_initial_sequence,
    create_updated_sequence_destructive,
    get_front_layer_non_destructive,
    map_front_gates_to_pzs,
    update_distance_map,
)
from src.mqt.ionshuttler.multi_shuttler.Outside.cycles import create_starting_config, get_state_idxs
from src.mqt.ionshuttler.multi_shuttler.Outside.graph_utils import (
    GraphCreator,
    ProcessingZone,
    PZCreator,
    create_idc_dictionary,
    get_idx_from_idc,
)
from src.mqt.ionshuttler.multi_shuttler.Outside.partition import get_partition
from src.mqt.ionshuttler.multi_shuttler.Outside.plotting import plot_state
from src.mqt.ionshuttler.multi_shuttler.Outside.scheduling import get_ions
from src.mqt.ionshuttler.multi_shuttler.Outside.shuttle import main as run_shuttle_main


def simulate_shuttling(plot: bool, arch: list[int], number_of_pzs: int) -> int:
    m, n, v, h = arch
    seed = 0
    failing_junctions = 0
    cycle_or_paths = "Cycles"

    # Define processing zones
    height = -4.5
    pzs = [
        ProcessingZone(
            "pz1",
            [
                (float((m - 1) * v), float((n - 1) * h)),
                (float((m - 1) * v), float(0)),
                (float((m - 1) * v - height), float((n - 1) * h / 2)),
            ],
        ),
        ProcessingZone("pz2", [(0.0, 0.0), (0.0, float((n - 1) * h)), (float(height), float((n - 1) * h / 2))]),
        ProcessingZone("pz3", [(float((m - 1) * v), float(0)), (0.0, 0.0), (float((m - 1) * v / 2), float(height))]),
        ProcessingZone(
            "pz4",
            [
                (0.0, float((n - 1) * h)),
                (float((m - 1) * v), float((n - 1) * h)),
                (float((m - 1) * v / 2), float((n - 1) * h - height)),
            ],
        ),
    ][:number_of_pzs]

    basegraph_creator = GraphCreator(m, n, v, h, failing_junctions, pzs)
    MZ_graph = basegraph_creator.get_graph()
    pzgraph_creator = PZCreator(m, n, v, h, failing_junctions, pzs)
    G = pzgraph_creator.get_graph()
    G.mz_graph = MZ_graph
    G.idc_dict = create_idc_dictionary(G)
    G.pzs = pzs
    G.pzs_name_map = {pz.name: pz for pz in pzs}

    G.parking_edges_idxs = [get_idx_from_idc(G.idc_dict, pz.parking_edge) for pz in pzs]
    G.edge_to_pz_map = {
        idx: pz
        for pz in pzs
        for idx in [*pz.path_to_pz_idxs, get_idx_from_idc(G.idc_dict, pz.parking_edge), *pz.path_from_pz_idxs]
    }
    G.max_num_parking = 3
    for pz in pzs:
        pz.max_num_parking = G.max_num_parking
        pz.getting_processed = []

    G.plot = plot
    G.save = False
    G.arch = str(arch)
    G.locked_gates = {}  # <-- added to prevent AttributeError

    number_of_chains = math.ceil(1.0 * len(MZ_graph.edges()))
    algorithm = "qft_no_swaps_nativegates_quantinuum_tket"
    qasm_path = f"QASM_files/development/{algorithm}/{algorithm}_{number_of_chains}.qasm"

    create_starting_config(G, number_of_chains, seed)
    G.state = get_ions(G)
    G.sequence = create_initial_sequence(qasm_path)
    partition = {pz.name: ions for pz, ions in zip(pzs, get_partition(qasm_path, len(pzs)), strict=False)}
    G.map_to_pz = {ion: pz for pz, ions in partition.items() for ion in ions}

    dag = create_dag(qasm_path)
    front_layer_nodes = get_front_layer_non_destructive(dag, [])
    map_front_gates_to_pzs(G, front_layer_nodes)
    G.dist_dict = create_dist_dict(G)
    G.dist_map = update_distance_map(G, get_state_idxs(G))

    sequence, _, dag = create_updated_sequence_destructive(G, qasm_path, dag, use_dag=True)
    G.sequence = sequence

    timesteps = run_shuttle_main(G, partition, dag, cycle_or_paths, use_dag=True)

    if plot:
        plot_state(G, (None, None), plot_ions=False, show_plot=True, save_plot=False)

    return timesteps


if __name__ == "__main__":
    ts = simulate_shuttling(plot=True, arch=[3, 3, 2, 2], number_of_pzs=1)
    print(f"Total time steps: {ts}")
