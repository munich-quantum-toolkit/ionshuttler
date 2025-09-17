from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import Graph
    from .types import Edge


def create_idc_dictionary(graph: Graph) -> dict[int, Edge]:
    edge_dict = {}
    for edge_idx, edge_idc in enumerate(graph.edges()):
        edge_dict[edge_idx] = tuple(sorted(edge_idc, key=sum))
    return edge_dict


def get_idx_from_idc(edge_dictionary: dict[int, Edge], idc: Edge) -> int:
    node1, node2 = tuple(sorted(idc, key=sum))
    return list(edge_dictionary.values()).index((node1, node2))


def get_idc_from_idx(edge_dictionary: dict[int, Edge], idx: int) -> Edge:
    return edge_dictionary[idx]
