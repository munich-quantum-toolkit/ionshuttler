from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

from mqt.ionshuttler.multi_shuttler.outside import scheduling
from mqt.ionshuttler.multi_shuttler.outside.graph import Graph
from mqt.ionshuttler.multi_shuttler.outside.ion_types import Edge

if TYPE_CHECKING:
    from mqt.ionshuttler.multi_shuttler.outside.processing_zone import ProcessingZone

NextEdges = dict[int, tuple[Edge, Edge]]


def _add_edge_with_ions(g: Graph, edge: Edge, ions: list[int]) -> None:
    u, v = edge
    if not g.has_edge(u, v):
        g.add_edge(u, v, ions=ions)
    else:
        g.edges[u, v]["ions"] = ions


def test_cost_function_hybrid_first_example() -> None:
    g: Graph = Graph()
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

    _add_edge_with_ions(g, ((0.0, 0.0), (0.0, 1.0)), [0])
    _add_edge_with_ions(g, ((0.0, 1.0), (1.0, 1.0)), [1])
    _add_edge_with_ions(g, ((1.0, 1.0), (1.0, 0.0)), [2])
    _add_edge_with_ions(g, ((1.0, 0.0), (0.0, 0.0)), [3])
    _add_edge_with_ions(g, ((1.0, 1.0), (1.0, 2.0)), [4])

    next_edges: NextEdges = {
        0: (((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (1.0, 1.0))),
        1: (((0.0, 1.0), (1.0, 1.0)), ((1.0, 1.0), (1.0, 0.0))),
        3: (((1.0, 0.0), (0.0, 0.0)), ((1.0, 0.0), (0.0, 0.0))),
        4: (((1.0, 1.0), (1.0, 2.0)), ((1.0, 1.0), (1.0, 0.0))),
    }

    assert scheduling.cost_function_hybrid(g, cycle, path, next_edges) == 0


def test_cost_function_hybrid_prefers_cycle_when_cheaper() -> None:
    g: Graph = Graph()
    g.add_node((0.0, 0.0), node_type="junction_node")
    g.add_node((0.0, 1.0), node_type="trap_node")
    g.add_node((0.0, 2.0), node_type="trap_node")
    g.add_node((1.0, 0.0), node_type="trap_node")
    g.add_node((2.0, 0.0), node_type="trap_node")
    g.add_node((1.0, 2.0), node_type="trap_node")

    cycle: list[Edge] = [((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (2.0, 0.0))]
    path: list[Edge] = [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 2.0))]

    _add_edge_with_ions(g, cycle[0], [0])  # cycle ions = 1
    _add_edge_with_ions(g, path[0], [1])  # path ions includes 1,2
    _add_edge_with_ions(g, path[1], [2])
    _add_edge_with_ions(g, cycle[1], [3])

    next_edges: NextEdges = {0: (((1.0, 0.0), (2.0, 0.0)), ((1.0, 0.0), (2.0, 0.0)))}

    assert scheduling.cost_function_hybrid(g, cycle, path, next_edges) == 0


def test_cost_function_hybrid_prefers_path_when_cheaper() -> None:
    g: Graph = Graph()
    g.add_node((0.0, 0.0), node_type="junction_node")
    g.add_node((0.0, 1.0), node_type="trap_node")
    g.add_node((0.0, 2.0), node_type="trap_node")
    g.add_node((1.0, 0.0), node_type="trap_node")
    g.add_node((2.0, 0.0), node_type="trap_node")
    g.add_node((1.0, 2.0), node_type="trap_node")

    cycle: list[Edge] = [((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (2.0, 0.0))]
    path: list[Edge] = [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 2.0))]

    _add_edge_with_ions(g, cycle[0], [0])  # cycle ions = 1
    _add_edge_with_ions(g, path[0], [1])  # path ions includes 1,2
    _add_edge_with_ions(g, path[1], [2])
    _add_edge_with_ions(g, cycle[1], [3])

    next_edges: NextEdges = {
        3: (cycle[1], cycle[0]),
        1: (path[0], path[1]),
    }
    assert scheduling.cost_function_hybrid(g, cycle, path, next_edges) == 1


def test_cost_function_hybrid_returns_0_on_tie() -> None:
    g: Graph = Graph()
    g.add_node((0.0, 0.0), node_type="junction_node")
    g.add_node((0.0, 1.0), node_type="trap_node")

    cycle: list[Edge] = [((0.0, 0.0), (0.0, 1.0))]
    path: list[Edge] = [((0.0, 0.0), (0.0, 1.0))]

    _add_edge_with_ions(g, cycle[0], [0])  # both costs identical
    next_edges: NextEdges = {}
    assert scheduling.cost_function_hybrid(g, cycle, path, next_edges) == 0


def test_create_move_list_prioritizes_entry_and_exit_ions() -> None:
    pz = cast(
        "ProcessingZone",
        SimpleNamespace(
            path_to_pz_idxs=[20],
            path_from_pz_idxs=[10],
            parking_edge=((9.0, 9.0), (9.0, 10.0)),
        ),
    )

    graph = SimpleNamespace(
        idc_dict={},
        get_edge_data=lambda _u, _v: {"edge_type": "trap"},
    )

    # Input order from partitioned priority queue
    partitioned_priority_queue = [1, 2, 3]

    # Fake ion positions: ion 2 in exit, ion 3 in entry, ion 1 in normal trap
    ions_state = {
        1: ("e1a", "e1b"),
        2: ("e2a", "e2b"),
        3: ("e3a", "e3b"),
    }

    edge_idx_map = {
        ("e1a", "e1b"): 30,  # normal MZ trap
        ("e2a", "e2b"): 20,  # exit
        ("e3a", "e3b"): 10,  # entry
    }

    with (
        patch(
            "mqt.ionshuttler.multi_shuttler.outside.scheduling.get_ions",
            return_value=ions_state,
        ),
        patch(
            "mqt.ionshuttler.multi_shuttler.outside.scheduling.get_idx_from_idc",
            side_effect=lambda _idc_dict, edge: edge_idx_map[edge],
        ),
        patch(
            "mqt.ionshuttler.multi_shuttler.outside.scheduling.find_path_edge_to_edge",
            return_value=[("x", "y")],  # length 1 path for normal ion
        ),
    ):
        move_list = scheduling.create_move_list(
            cast("Graph", graph),
            partitioned_priority_queue,
            pz,
        )

    # entry + exit ions should be in front (both before normal ion)
    assert move_list.index(2) < move_list.index(1)
    assert move_list.index(3) < move_list.index(1)
    assert set(move_list) == {1, 2, 3}


def _mk_graph(state: dict[int, Edge], edge_types: dict[Edge, str]) -> Graph:
    def get_edge_data(u: Any, v: Any) -> dict[str, str]:
        return {"edge_type": edge_types[u, v]}

    return cast(
        "Graph",
        SimpleNamespace(
            state=state,
            get_edge_data=get_edge_data,
        ),
    )


def test_calculate_next_edges_trap_edge_uses_find_next_edge_and_ordering() -> None:
    ion_edge: Edge = ((0.0, 0.0), (0.0, 1.0))
    candidate_next: Edge = ((0.0, 1.0), (1.0, 1.0))
    ordered_next: Edge = ((0.0, 1.0), (1.0, 1.0))

    graph = _mk_graph(
        state={1: ion_edge},
        edge_types={ion_edge: "trap", candidate_next: "trap"},
    )

    pz = cast("ProcessingZone", SimpleNamespace(parking_edge=((9.0, 9.0), (9.0, 10.0))))

    with (
        patch(
            "mqt.ionshuttler.multi_shuttler.outside.scheduling.find_next_edge",
            return_value=candidate_next,
        ),
        patch(
            "mqt.ionshuttler.multi_shuttler.outside.scheduling.find_ordered_edges",
            return_value=(ion_edge, ordered_next),
        ),
    ):
        out = scheduling.calculate_next_edges_for_moves(
            graph,
            [1],
            pz,
        )

    assert out == {1: (ion_edge, ordered_next)}


def test_calculate_next_edges_non_trap_edge_stays_put_and_does_not_route() -> None:
    ion_edge: Edge = ((2.0, 0.0), (2.0, 1.0))
    graph = _mk_graph(
        state={2: ion_edge},
        edge_types={ion_edge: "entry"},
    )
    pz = cast("ProcessingZone", SimpleNamespace(parking_edge=((9.0, 9.0), (9.0, 10.0))))

    with (
        patch("mqt.ionshuttler.multi_shuttler.outside.scheduling.find_next_edge") as p_find_next,
        patch("mqt.ionshuttler.multi_shuttler.outside.scheduling.find_ordered_edges") as p_order,
    ):
        out = scheduling.calculate_next_edges_for_moves(graph, [2], pz)

    assert out == {2: (ion_edge, ion_edge)}
    p_find_next.assert_not_called()
    p_order.assert_not_called()


def test_split_ions_by_direction_on_move_single_example() -> None:
    # Move with consecutive directed edge-pairs:
    # (e1->e2), (e2->e3), (e3->e4)
    move_edges: list[Edge] = [
        ((0.0, 0.0), (0.0, 1.0)),  # e1
        ((0.0, 1.0), (1.0, 1.0)),  # e2
        ((1.0, 1.0), (1.0, 2.0)),  # e3
        ((1.0, 2.0), (2.0, 2.0)),  # e4
    ]

    # next_edges maps ion -> (current_edge, next_edge)
    next_edges: NextEdges = {
        10: ((move_edges[0][0], move_edges[0][1]), (move_edges[1][0], move_edges[1][1])),  # same direction
        11: ((move_edges[1][0], move_edges[1][1]), (move_edges[2][0], move_edges[2][1])),  # same direction
        12: ((move_edges[2][1], move_edges[2][0]), (move_edges[1][1], move_edges[1][0])),  # opposite direction
        13: (
            (move_edges[3][0], move_edges[3][1]),
            (move_edges[0][0], move_edges[0][1]),
        ),  # not adjacent in move -> ignored
        14: ((move_edges[0][0], move_edges[0][1]), (move_edges[0][0], move_edges[0][1])),  # stop-like -> ignored
    }

    same, opposite = scheduling.split_ions_by_direction_on_move(move_edges, next_edges)

    assert set(same) == {10, 11}
    assert set(opposite) == {12}


def test_calculate_next_edges_for_moves_mixed_trap_and_non_trap() -> None:
    trap_edge: Edge = ((0.0, 0.0), (0.0, 1.0))
    trap_next: Edge = ((0.0, 1.0), (1.0, 1.0))
    non_trap_edge: Edge = ((2.0, 2.0), (2.0, 3.0))

    graph = _mk_graph(
        state={1: trap_edge, 2: non_trap_edge},
        edge_types={
            trap_edge: "trap",
            trap_next: "trap",
            non_trap_edge: "entry",
        },
    )
    pz = cast("ProcessingZone", SimpleNamespace(parking_edge=((3.0, 3.0), (4.0, 3.0))))

    with (
        patch(
            "mqt.ionshuttler.multi_shuttler.outside.scheduling.find_next_edge",
            return_value=trap_next,
        ) as p_find_next,
        patch(
            "mqt.ionshuttler.multi_shuttler.outside.scheduling.find_ordered_edges",
            return_value=(trap_edge, trap_next),
        ) as p_order,
    ):
        out = scheduling.calculate_next_edges_for_moves(graph, [1, 2], pz)

    assert out == {
        1: (trap_edge, trap_next),
        2: (non_trap_edge, non_trap_edge),
    }
    p_find_next.assert_called_once_with(graph, trap_edge, pz.parking_edge)
    p_order.assert_called_once_with(graph, trap_edge, trap_next)
