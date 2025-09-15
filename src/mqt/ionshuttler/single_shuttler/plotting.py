import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .graph_utils import create_idc_dictionary, get_idc_from_idx, get_idx_from_idc


# plotting
def plot_state(graph, ion_moves, labels, plot_ions=True, show_plot=False, save_plot=False, filename=""):
    idc_dict = create_idc_dictionary(graph)
    pos = {(x, y): (y, -x) for i, (x, y) in enumerate(list(graph.nodes()))}
    if plot_ions is True:
        pass
        # edge_labels = nx.get_edge_attributes(graph,'ion_chain')
    else:
        edge_labels = {}
        for idc in graph.edges():
            # pass
            edge_labels[idc] = f"$e_{{{get_idx_from_idc(idc_dict, idc)}}}$"

    for edge_idc in graph.edges():
        # color all edges black
        graph.add_edge(edge_idc[0], edge_idc[1], color="k")

        ion_holder = {}
        colors = []
        np.random.seed(0)
        for _ in range(len(ion_moves)):
            r = np.round(np.random.rand(), 1)
            g = np.round(np.random.rand(), 1)
            b = np.round(np.random.rand(), 1)

            colors.append((r, g, b))
        np.random.seed()

    for i, ion_place in enumerate(ion_moves):
        ion_edge_idc = get_idc_from_idx(idc_dict, ion_place)
        try:
            ion_holder[ion_place].append(i)
        except KeyError:
            ion_holder[ion_place] = [i]
    for i, ion_place in enumerate(ion_moves):
        ion_edge_idc = get_idc_from_idx(idc_dict, ion_place)
        graph.add_edge(ion_edge_idc[0], ion_edge_idc[1], ion_chain=ion_holder[ion_place], color=colors[i])

    edge_color = nx.get_edge_attributes(graph, "color").values()
    node_color = list(nx.get_node_attributes(graph, "color").values())
    edge_labels = nx.get_edge_attributes(graph, "ion_chain")

    # plt.figure(figsize=(25, 15))
    plt.figure(
        figsize=(max(pos.keys())[0] * 3, max(pos.keys())[1] * 3)
    )  # self.n * self.ion_chain_size_horizontal, self.m * self.ion_chain_size_vertical))
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=True,
        node_size=300,
        node_color=node_color,
        width=8,
        edge_color=edge_color,
        font_size=6,
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)

    # reset edge labels
    for i, ion in enumerate(ion_moves):
        ion_edge_idc = get_idc_from_idx(idc_dict, ion)
        graph.add_edge(ion_edge_idc[0], ion_edge_idc[1], ion_chain="", color=colors[i])

    labels0, labels1 = labels
    plt.plot([], [], label=labels0)
    plt.plot([], [], label=labels1)
    plt.legend()

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(filename)
    plt.close()
