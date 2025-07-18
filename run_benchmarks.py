from src.mqt.ionshuttler.multi_shuttler.Outside.graph_utils import GraphCreator, PZCreator, ProcessingZone, create_idc_dictionary
from src.mqt.ionshuttler.multi_shuttler.Outside.cycles import create_starting_config, find_path_edge_to_edge, get_state_idxs
from src.mqt.ionshuttler.multi_shuttler.Outside.scheduling import get_ions
from src.mqt.ionshuttler.multi_shuttler.Outside.shuttle import main as run_shuttle_main
import math
import networkx as nx
import numpy as np
from datetime import datetime
from src.mqt.ionshuttler.multi_shuttler.Outside.plotting import plot_state
from src.mqt.ionshuttler.multi_shuttler.Outside.graph_utils import get_idx_from_idc
from src.mqt.ionshuttler.multi_shuttler.Outside.compilation import create_initial_sequence, create_dag, create_updated_sequence_destructive, get_front_layer_non_destructive, get_all_first_gates_and_update_sequence_non_destructive, map_front_gates_to_pzs, create_dist_dict, update_distance_map
from src.mqt.ionshuttler.multi_shuttler.Outside.partition import get_partition

plot = False
save = False

paths = True   #3 3 5 5 failing junctions 0 seed 0 #pzs 4
cycle_or_paths = "Paths" if paths else "Cycles"

failing_junctions = 0

# 3333 seed0 pzs2 failing junctions1 paths -> can't push through to pz because of a blockage
archs = [
    # [5, 5, 5, 5],
    #[3, 3, 1, 1],
     [4, 4, 1, 1],
    #[3, 3, 3, 3],   # TODO hier langsamer als ohne compilation - nutzt pz4 erst zum Schluss - partitioning praktisch max schlecht? - eval für mehr seeds und vergleiche - gate selection anpassen, dass es so kommutiert, dass alle pzs beladen? - sollte das nicht eig. schon so sein?
    #[4, 4, 1, 1],
    # [4, 4, 2, 2],
    #[3, 3, 5, 5],
    #[5, 5, 1, 1],
    #[4, 4, 3, 3],
    #[5, 5, 3, 3],
    # [7, 7, 1, 1],
    # [8, 8, 1, 1]
]
# run all seeds
seeds = [1]#, 2, 3, 4, 5]#, 6, 7, 8, 9, 10]
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
number_of_pzs = [1]#, 2, 3, 4]

for m, n, v, h in archs:
    timesteps_average = {}
    cpu_time_average = {}
    
    
    for number_of_pz in number_of_pzs:
        timesteps_array = []
        cpu_time_array = []
        for seed in seeds:
            start_time = datetime.now()

            height = -4.5 # height > -1 (position of pz outside of mz) -> all edges can be ordered by sum -> if height -1 -> edge may be smaller in edge_idc (important for get_idx_from_idc())

            exit1 = (float((m-1)*v), float((n-1)*h))
            entry1 = (float((m-1)*v), float(0))
            processing_zone1 = (float((m-1)*v-height), float((n-1)*h/2))

            exit2 = (0.0, 0.0)
            entry2 = (0.0, float((n-1)*h))
            processing_zone2 = (float(height), float((n-1)*h/2))

            exit3 = (float((m-1)*v), float(0))
            entry3 = (float(0), float(0))
            processing_zone3 = (float((m-1)*v/2), float(height))

            exit4 = (float(0), float((n-1)*h))
            entry4 = (float((m-1)*v), float((n-1)*h))
            processing_zone4 = (float((m-1)*v/2), float((n-1)*h-height))

            pz1 = ProcessingZone("pz1", [exit1, entry1, processing_zone1])
            pz2 = ProcessingZone("pz2", [exit2, entry2, processing_zone2])
            pz3 = ProcessingZone("pz3", [exit3, entry3, processing_zone3])
            pz4 = ProcessingZone("pz4", [exit4, entry4, processing_zone4])
            pzs = [pz1, pz2, pz3, pz4][0:number_of_pz]

            basegraph_creator = GraphCreator(m, n, v, h, failing_junctions, pzs)
            MZ_graph = basegraph_creator.get_graph()        
            pzgraph_creator = PZCreator(m, n, v, h, failing_junctions, pzs)
            G = pzgraph_creator.get_graph()
            G.mz_graph = MZ_graph
            
            G.seed = seed

            G.idc_dict = create_idc_dictionary(G)
            G.pzs = pzs
            G.parking_edges_idxs = []
            G.pzs_name_map = {}
            G.edge_to_pz_map = {}
            for pz in G.pzs:
                G.parking_edges_idxs.append(get_idx_from_idc(G.idc_dict, pz.parking_edge))
                G.pzs_name_map[pz.name] = pz
                for edge_idx in pz.path_to_pz_idxs:
                    G.edge_to_pz_map[edge_idx] = pz
                G.edge_to_pz_map[get_idx_from_idc(G.idc_dict, pz.parking_edge)] = pz
                for edge_idx in pz.path_from_pz_idxs:
                    G.edge_to_pz_map[edge_idx] = pz
            print(f"parking_edges_idxs: {G.parking_edges_idxs}")
            
            G.max_num_parking = 3
            for pz in G.pzs:
                pz.max_num_parking = G.max_num_parking # if changed here (meaning pzs can hold different amounts of ions), also change in shuttle.py (check_duplicates) and check for further updates to max_num_parking

            G.plot = plot
            G.save = save
            G.arch = str([m, n, v, h])

            number_of_mz_edges = len(MZ_graph.edges())
            number_of_chains = math.ceil(.5*len(MZ_graph.edges()))
            

            # # plot for paper
            # plot_state(
            #     G, (None, None), plot_ions=False, show_plot=plot, save_plot=save
            # )

            print(f"Number of chains: {number_of_chains}")
            
            #algorithm = "qv_test_transpiled"
            #algorithm = "random_500_gates_no_swaps_nativegates_quantinuum_tket"
            algorithm = "qft_no_swaps_nativegates_quantinuum_tket"
            #algorithm = "full_register_access"
            qasm_file_path = (
                #f"../../../QASM_files/{algorithm}/{algorithm}_{number_of_chains}.qasm"
                f"QASM_files/development/{algorithm}/{algorithm}_{number_of_chains}.qasm"
            )

            edges = list(G.edges())

            create_starting_config(G, number_of_chains, seed=seed)

            G.state = get_ions(G)


            ### initial sequence (naive) ###
            G.sequence = create_initial_sequence(qasm_file_path)
            print('seq_length: ', len(G.sequence))
            seq_length = len(G.sequence)

            ### partitioning ###
            partitioning = True
            # if there is a real tuple in sequence (2-qbuit gate) use partitioning
            if partitioning: #any(len(i) > 1 for i in G.sequence):
                part = get_partition(qasm_file_path, len(G.pzs))
                partition = {pz.name: part[i] for i, pz in enumerate(G.pzs)}
                num_pzs = len(G.pzs)
            else:
                # else place them in the closest processing zone (equally distributed)
                # TODO double check
                partition = {pz.name: [] for pz in G.pzs}
                # Assign each ion to the closest processing zone
                for ion, position in G.state.items():
                    closest_pz = None
                    min_distance = float("inf")
                    for pz in G.pzs:
                        distance = len(find_path_edge_to_edge(G, position, pz.parking_edge))
                        if distance < min_distance:
                            min_distance = distance
                            closest_pz = pz
                    partition[closest_pz.name].append(ion)
                # Balance the ions among the processing zones
                all_ions = [ion for ions in partition.values() for ion in ions]
                all_ions.sort(key=lambda ion: G.state[ion])
                num_pzs = len(G.pzs)
                ions_per_pz = len(all_ions) // num_pzs
                for i, pz in enumerate(G.pzs):
                    start_index = i * ions_per_pz
                    end_index = start_index + ions_per_pz
                    partition[pz.name] = all_ions[start_index:end_index]
                # Distribute any remaining ions
                remaining_ions = all_ions[num_pzs * ions_per_pz :]
                for i, ion in enumerate(remaining_ions):
                    partition[G.pzs[i % num_pzs].name].append(ion)

            print('partition: ', partition)

            # Create a reverse mapping from element to partition name
            map_to_pz = {
                element: pz
                for pz, elements in partition.items()
                for element in elements
            }
            G.map_to_pz = map_to_pz

            # Ensure all elements are in one of the partitions
            all_partition_elements = []
            for elements in partition.values():
                all_partition_elements.extend(elements)
            unique_sequence = []
            for seq_elem in G.sequence:
                for elem in seq_elem:
                    if elem not in unique_sequence:
                        unique_sequence.append(elem)
            assert all(element in all_partition_elements for element in unique_sequence)

            if len(G.pzs) > 1:
                # and no element is in both partitions
                pz_sets = {pz: set(elements) for pz, elements in partition.items()}
                common_elements = set.intersection(*pz_sets.values())
                assert (
                    not common_elements
                ), f"{common_elements} are overlapping in partitions"






            use_dag = True

            if use_dag:
                for pz in G.pzs:
                    pz.getting_processed = []
                else:
                    dag = create_dag(qasm_file_path)
                G.locked_gates = {}
                front_layer_nodes = get_front_layer_non_destructive(dag, virtually_processed_nodes=[])
                pz_info_map = map_front_gates_to_pzs(G, front_layer_nodes=front_layer_nodes)
                gate_info_map = {value: key for key, values in pz_info_map.items() for value in values}

                G.dist_dict = create_dist_dict(G)
                state = get_state_idxs(G)
                G.dist_map = update_distance_map(G, state)

                sequence, flat_sequence, dag = create_updated_sequence_destructive(G, qasm_file_path, dag, use_dag=use_dag)
                G.sequence = sequence
            else:
                dag = None







            timesteps = run_shuttle_main(G, partition, dag, cycle_or_paths, use_dag=use_dag)
            end_time = datetime.now()
            cpu_time = end_time - start_time

            timesteps_array.append(timesteps)
            cpu_time_array.append(cpu_time)

            del dag

            # # save timesteps in a file
            # with open(f"benchmarks/{time}{algorithm}.txt", "a") as f:
            #     f.write(
            #         f"{m, n, v, h}, #pzs: {num_pzs}, ts: {timesteps}, seed: {seed}\n"
            #     )

        # calculate averages
        timesteps_array = np.array(timesteps_array)
        cpu_time_array = np.array(cpu_time_array)
        timesteps_average[number_of_pz] = np.mean(timesteps_array)
        cpu_time_average[number_of_pz] = np.mean(cpu_time_array)

        print('\nseed: ', seed)

        # save averages
        with open(f"benchmarks/{time}{algorithm}.txt", "a") as f:
            f.write(
            f"{m, n, v, h}, ions{number_of_chains}/pos{number_of_mz_edges}: {number_of_chains/number_of_mz_edges}, #pzs: {num_pzs}, avg_ts: {timesteps_average[num_pzs]}, avg_cpu_time: {cpu_time_average[num_pzs]}, gates: {seq_length}, baseline: {None}, use_dag: {use_dag}, paths: {paths}\n"
        )

for num_pzs in number_of_pzs:
    print(f"{m, n, v, h}, ions{number_of_chains}/pos{number_of_mz_edges}: {number_of_chains/number_of_mz_edges}, #pzs: {num_pzs}, average_ts: {timesteps_average[num_pzs]}, average_cpu_time: {cpu_time_average[num_pzs]}, use_dag: {use_dag}, paths: {paths}")

# TODOs:
# - TODO 1: gate_execution_finished -> implement different gate execution times
# - TODO 2: other_next_edges -> needed for path out of pz if cycles is True - would now have to be calculated for each pz before
# - TODO 3: new_gate_starting -> would need to know next gate at pz - checks in scheduling.py if new gate can start with ions in parking - would stop all ions in exit

# TODO maybe check logic of moving into exit (check if it is really important enough to move into exit -> cannot really check anymore if is important enough at pz, since new logic trys to just move everything through -> so maybe need to implement bouncer at exit?)

# TODO check at 3333 seed 0 use_dag why slower with use_dag -> check 4 moving to pz4 but getting pushed back by others? Priority queue correct?
# for qce benchmarks: maybe architectural design explore? -> constant number of ions and see impact of pzs -> can better evaluate if architecture is influenced (otherwise could be influence of longer circuits)
# for later papers: failing junctions + more design exploration like qce24? -> ions can vary, but circuit size same? -> see impact of unused ions etc.