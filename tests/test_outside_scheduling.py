from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.ionshuttler.multi_shuttler.outside import scheduling
from mqt.ionshuttler.multi_shuttler.outside.graph import Graph

if TYPE_CHECKING:
    from mqt.ionshuttler.multi_shuttler.outside.ion_types import Edge


def test_cost_function_hybrid_first_example() -> None:
    g: Graph = Graph()  # type: ignore[no-untyped-call]
    g.add_node((0.0, 0.0), node_type="junction_node")
    g.add_node((0.0, 1.0), node_type="junction_node")
    g.add_node((1.0, 1.0), node_type="junction_node")
    g.add_node((1.0, 0.0), node_type="junction_node")
    g.add_node((1.0, 2.0), node_type="junction_node")

    cycle: list[Edge] = [
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
    ]
    path: list[Edge] = [
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (1.0, 2.0)),
    ]

    assert scheduling.cost_function_hybrid(g, [1.0, 1.0], cycle, path) == 1


def test_cost_function_hybrid_prefers_cycle_when_cheaper() -> None:
    g: Graph = Graph()  # type: ignore[no-untyped-call]
    g.add_node((0.0, 0.0), node_type="junction_node")
    g.add_node((0.0, 1.0), node_type="trap_node")
    g.add_node((1.0, 1.0), node_type="trap_node")
    g.add_node((1.0, 2.0), node_type="trap_node")

    cycle: list[Edge] = [((0.0, 0.0), (0.0, 1.0))]
    path: list[Edge] = [((0.0, 1.0), (1.0, 1.0)), ((1.0, 1.0), (1.0, 2.0))]

    # Choose weights so cycle is cheaper:
    # cycle_cost = alpha*1 + beta*1
    # path_cost  = alpha*2 + beta*0

    assert scheduling.cost_function_hybrid(g, [1.0, 0.5], cycle, path) == 0


def test_cost_function_hybrid_prefers_path_when_cheaper() -> None:
    g: Graph = Graph()  # type: ignore[no-untyped-call]
    g.add_node((0.0, 0.0), node_type="junction_node")
    g.add_node((0.0, 1.0), node_type="trap_node")
    g.add_node((1.0, 1.0), node_type="trap_node")
    g.add_node((1.0, 2.0), node_type="trap_node")

    cycle: list[Edge] = [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (1.0, 1.0))]
    path: list[Edge] = [((0.0, 1.0), (1.0, 1.0))]

    # Make junctions expensive, so cycle loses:
    # cycle_cost = 1*2 + 10*1 = 12
    # path_cost  = 1*1 + 10*0 = 1

    assert scheduling.cost_function_hybrid(g, [1.0, 10.0], cycle, path) == 1


def test_cost_function_hybrid_returns_0_on_tie() -> None:
    g: Graph = Graph()  # type: ignore[no-untyped-call]
    g.add_node((0.0, 0.0), node_type="junction_node")
    g.add_node((0.0, 1.0), node_type="trap_node")

    cycle: list[Edge] = [((0.0, 0.0), (0.0, 1.0))]
    path: list[Edge] = [((0.0, 0.0), (0.0, 1.0))]

    assert scheduling.cost_function_hybrid(g, [1.0, 1.0], cycle, path) == 0
