import networkx as nx
from more_itertools import pairwise

# global delete_node
# delete_node = (3, 4)


# create dictionary to swap from idx to idc and vice versa
def create_idc_dictionary(nx_g):
    edge_dict = {}
    for edge_idx, edge_idc in enumerate(nx_g.edges()):
        edge_dict[edge_idx] = tuple(sorted(edge_idc, key=sum))
    return edge_dict


def get_idx_from_idc(edge_dictionary, idc):
    idc = tuple(sorted(idc, key=sum))
    return list(edge_dictionary.values()).index(idc)


def get_idc_from_idx(edge_dictionary, idx):
    return edge_dictionary[idx]


def order_edges(edge1, edge2):
    # Find the common node shared between the two edges
    common_node = set(edge1).intersection(set(edge2))

    if len(common_node) != 1 and edge1 != edge2:
        msg = f"The input edges are not connected. Edges: {edge1}, {edge2}"
        raise ValueError(msg)

    common_node = common_node.pop()
    if edge1[0] == common_node:
        edge1_in_order = (edge1[1], common_node)
        edge2_in_order = (common_node, edge2[1]) if edge2[0] == common_node else (common_node, edge2[0])
    else:
        edge1_in_order = (edge1[0], common_node)
        edge2_in_order = (common_node, edge2[1]) if edge2[0] == common_node else (common_node, edge2[0])

    return edge1_in_order, edge2_in_order


def get_path_to_node(nx_g, src, tar, exclude_exit=False, exclude_first_entry_connection=True):
    edge_path = []
    if exclude_first_entry_connection is True:
        # lambda function to give path over processing zone huge weight -> doesn't take that path if not necessary - now only encludes entry edge -> can use exit (in MemGrid was != trap before and then to exit node -> not PZ node)
        node_path = nx.shortest_path(
            nx_g,
            src,
            tar,
            lambda _, __, edge_attr_dict: (edge_attr_dict["edge_type"] == "first_entry_connection") * 1e8 + 1,
        )
        # also exclude exit edge if necessary
        if exclude_exit is True:
            node_path = nx.shortest_path(
                nx_g,
                src,
                tar,
                lambda _, __, edge_attr_dict: (edge_attr_dict["edge_type"] in {"first_entry_connection", "exit"}) * 1e8
                + 1,
            )

    # only exclude exit edge
    elif exclude_exit is True:
        node_path = nx.shortest_path(
            nx_g, src, tar, lambda _, __, edge_attr_dict: (edge_attr_dict["edge_type"] == "exit") * 1e8 + 1
        )

    else:
        node_path = nx.shortest_path(nx_g, src, tar)
    # shortest path should always be the correct path in a grid -> care for changes

    for edge in pairwise(node_path):
        edge_path.append(edge)

    return edge_path


def calc_dist_to_pz(nx_g_creator, edge_idx):
    edge_idc = get_idc_from_idx(nx_g_creator.idc_dict, edge_idx)
    node1, node2 = edge_idc[0], edge_idc[1]

    path1 = get_path_to_node(
        nx_g_creator.networkx_graph, node1, nx_g_creator.processing_zone, exclude_first_entry_connection=True
    )
    path2 = get_path_to_node(
        nx_g_creator.networkx_graph, node2, nx_g_creator.processing_zone, exclude_first_entry_connection=True
    )
    if edge_idx == get_idx_from_idc(nx_g_creator.idc_dict, nx_g_creator.parking_edge):
        return 0
    if edge_idx == get_idx_from_idc(nx_g_creator.idc_dict, nx_g_creator.first_entry_connection_from_pz):
        return max(len(path1), len(path2)) + 1
    return min(len(path1), len(path2)) + 1


class MZGraphCreator:
    def __init__(self, m, n, ion_chain_size_vertical, ion_chain_size_horizontal, pz):
        self.m = m
        self.n = n
        self.ion_chain_size_vertical = ion_chain_size_vertical
        self.ion_chain_size_horizontal = ion_chain_size_horizontal

        self.pz = pz
        self.networkx_graph = self.create_graph()
        self.idc_dict = create_idc_dictionary(self.networkx_graph)

    def create_graph(self):
        self.m_extended = self.m + (self.ion_chain_size_vertical - 1) * (self.m - 1)
        self.n_extended = self.n + (self.ion_chain_size_horizontal - 1) * (self.n - 1)

        networkx_graph = nx.grid_2d_graph(self.m_extended, self.n_extended)
        self._set_trap_nodes(networkx_graph)
        self._remove_horizontal_edges(networkx_graph)
        self._remove_vertical_edges(networkx_graph)
        self._remove_horizontal_nodes(networkx_graph)
        self._set_junction_nodes(networkx_graph)
        if self.pz == "mid":
            self._remove_mid_part(networkx_graph)
        nx.set_edge_attributes(networkx_graph, "trap", "edge_type")
        # self._delete_junction(networkx_graph, delete_node)

        return networkx_graph

    def _set_trap_nodes(self, networkx_graph):
        for node in networkx_graph.nodes():
            networkx_graph.add_node(node, node_type="trap_node", color="b")

    def _remove_horizontal_edges(self, networkx_graph):
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(self.n_extended - 1):
                    networkx_graph.remove_edge((i + k, j), (i + k, j + 1))

    def _remove_vertical_edges(self, networkx_graph):
        for i in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
            for k in range(1, self.ion_chain_size_horizontal):
                for j in range(self.m_extended - 1):
                    networkx_graph.remove_edge((j, i + k), (j + 1, i + k))

    def _remove_horizontal_nodes(self, networkx_graph):
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
                    for s in range(1, self.ion_chain_size_horizontal):
                        networkx_graph.remove_node((i + k, j + s))

    def _remove_mid_part(self, networkx_graph):
        for i in range(self.ion_chain_size_vertical):
            networkx_graph.remove_node((self.m_extended // 2, self.n_extended // 2 + i))
        for i in range(1, self.ion_chain_size_vertical):
            networkx_graph.remove_node((self.m_extended // 2, self.n_extended // 2 - i))
        for i in range(1, self.ion_chain_size_horizontal):
            networkx_graph.remove_node((self.m_extended // 2 + i, self.n_extended // 2))
        for i in range(1, self.ion_chain_size_horizontal):
            networkx_graph.remove_node((self.m_extended // 2 - i, self.n_extended // 2))

    def _set_junction_nodes(self, networkx_graph):
        for i in range(0, self.m_extended, self.ion_chain_size_vertical):
            for j in range(0, self.n_extended, self.ion_chain_size_horizontal):
                networkx_graph.add_node((i, j), node_type="junction_node", color="g")

    def _delete_junction(self, networkx_graph, junction_node):
        # Remove the junction node
        networkx_graph.remove_node(junction_node)

    def get_graph(self):
        return self.networkx_graph


class GraphCreator:
    def __init__(self, m, n, ion_chain_size_vertical, ion_chain_size_horizontal, pz):
        self.m = m
        self.n = n
        self.ion_chain_size_vertical = ion_chain_size_vertical
        self.ion_chain_size_horizontal = ion_chain_size_horizontal

        self.pz = pz
        self.networkx_graph = self.create_graph()

        self.idc_dict = create_idc_dictionary(self.networkx_graph)
        self.path_to_pz_idxs = [get_idx_from_idc(self.idc_dict, edge) for edge in self.path_to_pz]
        self.path_from_pz_idxs = [get_idx_from_idc(self.idc_dict, edge) for edge in self.path_from_pz]

        # create lookup dictionaries for rest of path to and from processing zone
        self.rest_of_path_to_pz = {edge: self.path_to_pz[i + 1 :] for i, edge in enumerate(self.path_to_pz)}
        self.rest_of_path_from_pz = {edge: self.path_from_pz[i + 1 :] for i, edge in enumerate(self.path_from_pz)}

        self.pz_edges_idx = [
            get_idx_from_idc(self.idc_dict, edge)
            for edge in self.networkx_graph.edges()
            if nx.get_edge_attributes(self.networkx_graph, "edge_type")[edge] != "trap"
        ]

    def create_graph(self):
        self.m_extended = self.m + (self.ion_chain_size_vertical - 1) * (self.m - 1)
        self.n_extended = self.n + (self.ion_chain_size_horizontal - 1) * (self.n - 1)
        self.num_edges = self.n // 2

        networkx_graph = nx.grid_2d_graph(self.m_extended, self.n_extended)
        self._set_trap_nodes(networkx_graph)
        self._remove_horizontal_edges(networkx_graph)
        self._remove_vertical_edges(networkx_graph)
        self._remove_horizontal_nodes(networkx_graph)
        self._set_junction_nodes(networkx_graph)
        if self.pz == "mid":
            self._remove_mid_part(networkx_graph)
        nx.set_edge_attributes(networkx_graph, "trap", "edge_type")
        self._set_processing_zone(networkx_graph)
        # self._delete_junction(networkx_graph, delete_node)

        return networkx_graph

    def _set_trap_nodes(self, networkx_graph):
        for node in networkx_graph.nodes():
            networkx_graph.add_node(node, node_type="trap_node", color="b")

    def _remove_horizontal_edges(self, networkx_graph):
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(self.n_extended - 1):
                    networkx_graph.remove_edge((i + k, j), (i + k, j + 1))

    def _remove_vertical_edges(self, networkx_graph):
        for i in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
            for k in range(1, self.ion_chain_size_horizontal):
                for j in range(self.m_extended - 1):
                    networkx_graph.remove_edge((j, i + k), (j + 1, i + k))

    def _remove_horizontal_nodes(self, networkx_graph):
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
                    for s in range(1, self.ion_chain_size_horizontal):
                        networkx_graph.remove_node((i + k, j + s))

    def _remove_mid_part(self, networkx_graph):
        for i in range(self.ion_chain_size_vertical):
            networkx_graph.remove_node((self.m_extended // 2, self.n_extended // 2 + i))
        for i in range(1, self.ion_chain_size_vertical):
            networkx_graph.remove_node((self.m_extended // 2, self.n_extended // 2 - i))
        for i in range(1, self.ion_chain_size_horizontal):
            networkx_graph.remove_node((self.m_extended // 2 + i, self.n_extended // 2))
        for i in range(1, self.ion_chain_size_horizontal):
            networkx_graph.remove_node((self.m_extended // 2 - i, self.n_extended // 2))

        assert self.m % 2 == 1

        assert self.ion_chain_size_vertical >= 2
        assert self.ion_chain_size_horizontal >= 2

    def _set_junction_nodes(self, networkx_graph):
        for i in range(0, self.m_extended, self.ion_chain_size_vertical):
            for j in range(0, self.n_extended, self.ion_chain_size_horizontal):
                networkx_graph.add_node((i, j), node_type="junction_node", color="g")

    def _set_processing_zone(self, networkx_graph):
        if self.pz == "mid":
            self.exit = (self.m_extended // 2, self.n_extended // 2 + self.ion_chain_size_vertical)
            self.entry = (self.m_extended // 2, self.n_extended // 2 - self.ion_chain_size_vertical)
            self.processing_zone = (self.m_extended // 2, self.n_extended // 2)
            self.parking_node = (self.processing_zone[0] + 1, self.processing_zone[1])
            self.parking_edge = (self.processing_zone, self.parking_node)

            # Add the processing zone node
            networkx_graph.add_node(self.processing_zone, node_type="processing_zone_node", color="r")

            # new parking edge
            networkx_graph.add_node(self.parking_node, node_type="parking_node", color="r")
            networkx_graph.add_edge(self.parking_edge[0], self.parking_edge[1], edge_type="parking_edge", color="g")

            self.path_to_pz = []
            self.path_from_pz = []

            for i in range(1, self.ion_chain_size_horizontal + 1):
                if i == self.ion_chain_size_horizontal:
                    node_type = "exit_node"
                    self.exit_edge = (
                        (self.m_extended // 2, self.n_extended // 2 + i),
                        (self.m_extended // 2, self.n_extended // 2 + i - 1),
                    )
                else:
                    node_type = "exit_connection_node"
                networkx_graph.add_node(
                    (self.m_extended // 2, self.n_extended // 2 + i), node_type=node_type, color="y"
                )
                networkx_graph.add_edge(
                    (self.m_extended // 2, self.n_extended // 2 + i),
                    (self.m_extended // 2, self.n_extended // 2 + i - 1),
                    edge_type="exit",
                    color="k",
                )

                self.path_to_pz.insert(
                    0,
                    (
                        (self.m_extended // 2, self.n_extended // 2 + i),
                        (self.m_extended // 2, self.n_extended // 2 + i - 1),
                    ),
                )

            for i in range(1, self.ion_chain_size_horizontal + 1):
                if i == 1:
                    node_type = "entry_connection_node"
                    edge_type = "first_entry_connection"
                    self.first_entry_connection_from_pz = (
                        (self.m_extended // 2, self.n_extended // 2 - i + 1),
                        (self.m_extended // 2, self.n_extended // 2 - i),
                    )
                    if self.ion_chain_size_horizontal == 1:
                        self.entry_edge = (
                            (self.m_extended // 2, self.n_extended // 2 - i + 1),
                            (self.m_extended // 2, self.n_extended // 2 - i),
                        )
                elif i == self.ion_chain_size_horizontal:
                    node_type = "entry_node"
                    edge_type = "entry"
                    self.entry_edge = (
                        (self.m_extended // 2, self.n_extended // 2 - i + 1),
                        (self.m_extended // 2, self.n_extended // 2 - i),
                    )
                else:
                    node_type = "entry_connection_node"
                    edge_type = "entry"
                networkx_graph.add_node(
                    (self.m_extended // 2, self.n_extended // 2 - i), node_type=node_type, color="orange"
                )
                networkx_graph.add_edge(
                    (self.m_extended // 2, self.n_extended // 2 - i + 1),
                    (self.m_extended // 2, self.n_extended // 2 - i),
                    edge_type=edge_type,
                    color="k",
                )

                self.path_from_pz.append((
                    (self.m_extended // 2, self.n_extended // 2 - i + 1),
                    (self.m_extended // 2, self.n_extended // 2 - i),
                ))

        elif self.pz == "outer":
            # Define the key nodes
            self.exit = (self.m_extended - 1, self.n_extended - 1)
            self.processing_zone = (self.m_extended + self.num_edges - 1, self.n_extended + self.num_edges - 1)
            self.entry = (self.m_extended - 1, 0)
            self.parking_node = (self.processing_zone[0] + 1, self.processing_zone[1])
            self.parking_edge = (self.processing_zone, self.parking_node)

            # differences
            dy_exit = self.exit[1] - self.processing_zone[1]
            dy_entry = self.processing_zone[1] - self.entry[1]

            self.path_to_pz = []
            self.path_from_pz = []

            # Add exit edges
            for i in range(self.num_edges):
                exit_node = (self.exit[0] + (i + 1), self.exit[1] - (i + 1) * dy_exit / self.num_edges)

                if i == 0:
                    networkx_graph.add_node(exit_node, node_type="exit_node", color="y")
                    previous_exit_node = self.exit
                    self.exit_edge = (previous_exit_node, exit_node)

                networkx_graph.add_node(exit_node, node_type="exit_connection_node", color="y")
                networkx_graph.add_edge(previous_exit_node, exit_node, edge_type="exit", color="k")
                self.path_to_pz.append((previous_exit_node, exit_node))
                previous_exit_node = exit_node

            # Add entry edges
            for i in range(self.num_edges):
                entry_node = (self.entry[0] + (i + 1), self.entry[1] + (i + 1) * dy_entry / self.num_edges)

                if i == 0:
                    networkx_graph.add_node(entry_node, node_type="entry_node", color="orange")
                    previous_entry_node = self.entry
                    self.entry_edge = (previous_entry_node, entry_node)

                networkx_graph.add_node(entry_node, node_type="entry_connection_node", color="orange")
                # first entry connection is first edge after pz
                # entry is edge connected to memory grid, so last entry connection
                # if entry is one edge only -> first entry connection is the same as entry edge
                if entry_node == self.processing_zone:
                    self.first_entry_connection_from_pz = (entry_node, previous_entry_node)
                    networkx_graph.add_edge(
                        previous_entry_node, entry_node, edge_type="first_entry_connection", color="k"
                    )
                else:
                    networkx_graph.add_edge(previous_entry_node, entry_node, edge_type="entry", color="k")
                self.path_from_pz.insert(0, (entry_node, previous_entry_node))

                previous_entry_node = entry_node

            assert exit_node == entry_node, "Exit and entry do not end in same node"
            assert exit_node == self.processing_zone, "Exit and entry do not end in processing zone"

            # Add the processing zone node
            networkx_graph.add_node(self.processing_zone, node_type="processing_zone_node", color="r")

            # new parking edge
            networkx_graph.add_node(self.parking_node, node_type="parking_node", color="r")
            networkx_graph.add_edge(self.parking_edge[0], self.parking_edge[1], edge_type="parking_edge", color="g")

        else:
            msg = "pz must be 'mid' or 'outer'"
            raise ValueError(msg)

    def _delete_junction(self, networkx_graph, junction_node):
        # Remove the junction node
        networkx_graph.remove_node(junction_node)

    def get_graph(self):
        return self.networkx_graph

    def find_connected_edges(self):
        connected_edge_pairs = set()
        for edge in self.networkx_graph.edges():
            node1, node2 = edge
            # Find edges connected to node1
            for neighbor in self.networkx_graph.neighbors(node1):
                if neighbor != node2:  # avoid the original edge
                    edge_pair = tuple(sorted([edge, (node1, neighbor)]))
                    connected_edge_pairs.add(edge_pair)
            # Find edges connected to node2
            for neighbor in self.networkx_graph.neighbors(node2):
                if neighbor != node1:  # avoid the original edge
                    edge_pair = tuple(sorted([edge, (node2, neighbor)]))
                    connected_edge_pairs.add(edge_pair)
        # order edges (also include reverse order -> opposite direction moves are now needed if a junction fails)
        connected_edge_pairs = [order_edges(edge_pair[0], edge_pair[1]) for edge_pair in connected_edge_pairs] + [
            order_edges(edge_pair[1], edge_pair[0]) for edge_pair in connected_edge_pairs
        ]
        # Convert set of tuples to a list of lists
        return [list(pair) for pair in connected_edge_pairs]
