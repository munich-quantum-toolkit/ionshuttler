from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .graph_utils import get_idc_from_idx

if TYPE_CHECKING:
    from .graph import Graph
    from .ion_types import Edge


# Plotting function
def plot_state(
    graph: Graph,
    labels: tuple[str, str | None],
    plot_ions: bool = True,
    show_plot: bool = False,
    save_plot: bool = False,
    plot_cycle: list[int] | bool = False,
    plot_pzs: bool = False,
    filename: Path | None = None,
) -> None:
    if filename is None:
        filename = Path("graph.pdf")

    plot_paper = False

    # --- Node positions: assumes nodes are (x,y) int tuples ---
    # Original intent: flip axes / mirror y; keep exactly that behavior
    pos = {n: (n[1], -n[0]) for n in graph.nodes()}

    # Optional: if you have stub nodes, place them near their neighbor for nicer plots
    # (requires that stubs have attribute is_stub=True and degree 1)
    # Optional: place stub nodes near their neighbor for nicer plots
    for node, node_data in graph.nodes(data=True):
        if node_data.get("is_stub", False):
            for _inner_node, inner_data in graph.nodes(data=True):
                if inner_data.get("is_stub", False):
                    try:
                        u = inner_data.get("stub_neighbor")

                        # Only use u if it still exists in pos
                        if u is None or u not in pos:
                            nbrs = [v for v in graph.neighbors(node) if v in pos]
                            u = nbrs[0] if nbrs else None

                        parent = inner_data.get("failed_parent")

                        # If no valid live anchor exists, fall back to parent position
                        if u is None:
                            if parent is not None:
                                pos[node] = (parent[1], -parent[0])
                            continue

                        x_u, y_u = pos[u]

                        if parent is not None:
                            x_p, y_p = (parent[1], -parent[0])
                            dx, dy = (x_p - x_u), (y_p - y_u)
                        else:
                            dx, dy = (0.0, 0.0)

                        if dx == 0.0 and dy == 0.0:
                            h = hash(node) % 4
                            if h == 0:
                                dx, dy = (1.0, 0.0)
                            elif h == 1:
                                dx, dy = (-1.0, 0.0)
                            elif h == 2:
                                dx, dy = (0.0, 1.0)
                            else:
                                dx, dy = (0.0, -1.0)

                        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
                        pos[node] = (x_u + 0.35 * dx / norm, y_u + 0.35 * dy / norm)

                    except Exception:
                        pass
    if plot_ions is True:
        nx.get_edge_attributes(graph, "ions")

    # Color all edges black by default
    for edge_idc in graph.edges():
        graph.add_edge(edge_idc[0], edge_idc[1], color="k")

    # Prepare ion color palette (deterministic across calls)
    ion_holder: dict[Edge, list[int]] = {}
    colors = []
    np.random.seed(0)
    for _ in range(len(graph.edges)):
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        colors.append((r, g, b))
    np.random.seed()

    if plot_paper is False:
        # populate ion_holder (saves colors of edges with ions in next loop)
        for edge in graph.edges:
            ions = graph.edges[edge].get("ions", [])
            for ion in ions:
                ion_holder.setdefault(edge, []).append(ion)

        # color edges that carry ions
        for edge in graph.edges:
            if edge in ion_holder:
                # use color of first ion on that edge
                ion_id = ion_holder[edge][0]
                # be safe if ion_id exceeds palette size
                c = colors[ion_id % len(colors)] if colors else "k"
                graph.add_edge(edge[0], edge[1], ions=ion_holder[edge], color=c)

    if plot_cycle is not False:
        assert isinstance(plot_cycle, list)
        for edge_idx in plot_cycle:
            edge = get_idc_from_idx(graph.idc_dict, edge_idx)
            graph.add_edge(edge[0], edge[1], color="r")
            for node in edge:
                if nx.get_node_attributes(graph, "node_type").get(node) != "junction_node":
                    graph.add_node(node, color="r")

    if plot_pzs:
        for pz in graph.pzs:
            graph.add_edge(pz.parking_edge[0], pz.parking_edge[1], color="r")

    # --- IMPORTANT FIX: build arrays aligned with explicit node/edge lists ---
    nodelist = list(graph.nodes())
    edgelist = list(graph.edges())

    edge_color_attr = nx.get_edge_attributes(graph, "color")
    edge_color = [edge_color_attr.get(e, "k") for e in edgelist]

    node_color_attr = nx.get_node_attributes(graph, "color")
    node_size_attr = nx.get_node_attributes(graph, "node_size")

    # defaults prevent crashes when new nodes (e.g., stubs) lack attributes
    node_color = [node_color_attr.get(n, "k") for n in nodelist]
    node_size = [node_size_attr.get(n, 300) for n in nodelist]

    if plot_paper is False:
        nx.get_edge_attributes(graph, "ions")

    plt.figure(figsize=(20, 9))
    with_labels = not plot_paper

    nx.draw_networkx(
        graph,
        pos=pos,
        nodelist=nodelist,
        edgelist=edgelist,
        with_labels=with_labels,
        node_size=node_size,
        node_color=node_color,
        width=8,
        edge_color=edge_color,
        font_size=8,
        font_color="white",
    )

    if plot_ions:
        edge_labels = {}
        for e in edgelist:
            ions = graph.edges[e].get("ions", [])
            if ions:
                # show ion ids on occupied edges
                edge_labels[e] = ",".join(map(str, ions))
        nx.draw_networkx_edge_labels(
            graph,
            pos=pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="black",
            rotate=False,
        )

    labels0, labels1 = labels
    plt.plot([], [], label=labels0)
    plt.plot([], [], label=labels1)
    plt.legend()

    if show_plot:
        backend = plt.get_backend().lower()
        if "agg" not in backend:
            plt.show()

    if save_plot is True:
        plt.savefig(filename)

    plt.close()
