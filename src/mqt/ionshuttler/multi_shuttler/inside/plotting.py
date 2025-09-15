import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .graph_utils import get_idx_from_idc


# Plotting function
def plot_state(
    graph,
    labels,
    plot_ions=True,
    show_plot=False,
    save_plot=False,
    plot_cycle=False,
    plot_pzs=False,
    filename="graph.pdf",
):
    idc_dict = graph.idc_dict
    pos = {(x, y): (y, -x) for i, (x, y) in enumerate(list(graph.nodes()))}
    if plot_ions is True:
        pass
        # edge_labels = nx.get_edge_attributes(graph,'ions')
    else:
        edge_labels = {}
        for idc in graph.edges():
            edge_labels[idc] = f"$e_{{{get_idx_from_idc(idc_dict, idc)}}}$"

    for edge_idc in graph.edges():
        # color all edges black
        graph.add_edge(edge_idc[0], edge_idc[1], color="k")

        ion_holder = {}
        colors = []
        np.random.seed(0)
        for _ in range(len(graph.edges)):
            r = np.round(np.random.rand(), 1)
            g = np.round(np.random.rand(), 1)
            b = np.round(np.random.rand(), 1)

            colors.append((r, g, b))
        np.random.seed()

    # populate ion_holder (saves colors of edges with ions in next loop)
    for edge in graph.edges:
        ions = graph.edges[edge]["ions"]
        for ion in ions:
            try:
                ion_holder[edge].append(ion)
            except KeyError:
                ion_holder[edge] = [ion]

    for edge in graph.edges:
        if edge in ion_holder:
            graph.add_edge(
                edge[0],
                edge[1],
                ions=ion_holder[edge],
                color=colors[ion_holder[edge][0]],
            )

    if plot_cycle is not False:
        for edge in plot_cycle:
            graph.add_edge(edge[0], edge[1], color="r")

    if plot_pzs is not False:
        for pz in graph.pzs:
            graph.add_edge(pz.edge_idc[0], pz.edge_idc[1], color="g")

    edge_color = nx.get_edge_attributes(graph, "color").values()
    node_color = list(nx.get_node_attributes(graph, "color").values())
    edge_labels = nx.get_edge_attributes(graph, "ions")
    # node_size = list(nx.get_node_attributes(graph, "node_size").values())

    plt.figure(figsize=(max(pos.keys())[1] * 2, max(pos.keys())[0] * 2))
    print(graph.nodes())
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=True,
        # node_size=node_size,
        node_color=node_color,
        width=8,
        edge_color=edge_color,
        font_size=6,
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)

    # # reset edge labels for following iterations?
    # for edge in graph.edges:
    #     if edge in ion_holder:
    #         graph.add_edge(edge[0], edge[1], ions=[], color="k")

    labels0, labels1 = labels
    plt.plot([], [], label=labels0)
    plt.plot([], [], label=labels1)
    # plt.legend()

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(filename)
    plt.close()
