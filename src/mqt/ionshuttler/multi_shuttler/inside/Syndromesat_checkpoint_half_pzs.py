# So it found a solution in 32 time steps.
# now this is the code i used
from __future__ import annotations

import json
import pathlib
import sys
from itertools import pairwise, permutations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
from z3 import And, AtLeast, AtMost, Bool, If, Not, Or, Solver, Sum, sat

if TYPE_CHECKING:
    Edge = tuple[tuple[int, int], tuple[int, int]]
    Node = tuple[int, int]
    Graph = nx.Graph

# ==========================================
# 1. Visualization Generator
# ==========================================


def generate_visualization_json(solver, model, filename, m, n, v, h):
    """
    Generates visualization.
    Now detects gates based on ADJACENCY in the split PZ edges.
    """

    def fmt_coord(node):
        return f"({node[0]}, {node[1]})"

    idx_to_edge = dict(enumerate(solver.edge_list))

    # Architecture: Draw the "PZ Area" (The middle two edges)
    # We will draw the full grid lines based on physical edges
    inner_pz_edges_block = []
    # Just draw the middle 2 edges of every vertical chain to represent PZs visually
    # Filter solver pz_pairs to get raw edges
    for pz_pair in solver.pz_pairs_indices:
        e1 = idx_to_edge[pz_pair[0]]
        e2 = idx_to_edge[pz_pair[1]]
        for edge in [e1, e2]:
            u, v_node = sorted(edge)
            inner_pz_edges_block.append([fmt_coord(u), fmt_coord(v_node)])

    architecture_block = {
        "grid": {"rows": m, "cols": n},
        "sites": {"vertical": v, "horizontal": h},
        "pzs": {"top": False, "right": False, "bottom": False, "left": False},
        "innerPZEdges": inner_pz_edges_block,
    }

    timeline_data = []
    print(f"Generating timeline for {solver.timesteps} steps...")

    for t in range(solver.timesteps):
        frame_ions = []
        frame_gates = []

        # A. Ions
        for ion in solver.ions:
            for edge_idx in range(len(solver.edge_list)):
                if bool(model.evaluate(solver.states[t, edge_idx, ion])):
                    edge = idx_to_edge[edge_idx]
                    u, v_node = sorted(edge)
                    frame_ions.append({"id": f"$q_{{{ion}}}$", "edge": [fmt_coord(u), fmt_coord(v_node)]})
                    break

        # B. Gates (Based on Solver Gate Variables)
        # We query the 'gate_start' variables to see active gates
        # This is more robust than re-detecting positions
        # However, we need access to the gate definitions.
        # Alternatively, check adjacency in PZs manually:

        for pz_L_idx, pz_R_idx in solver.pz_pairs_indices:
            # Check if any pair of ions is here
            ions_L = [i for i in solver.ions if bool(model.evaluate(solver.states[t, pz_L_idx, i]))]
            ions_R = [i for i in solver.ions if bool(model.evaluate(solver.states[t, pz_R_idx, i]))]

            if ions_L and ions_R:
                # We have a pair!
                ionA = ions_L[0]
                ionB = ions_R[0]

                # Visual location: Draw box around both edges?
                # Or just pick the "Middle" node shared by them.
                eL = idx_to_edge[pz_L_idx]
                eR = idx_to_edge[pz_R_idx]

                # Format for viz
                # Use the outer nodes to span the whole PZ area
                nodes = sorted(set(eL + eR))  # Should be 3 nodes
                u, v_node = nodes[0], nodes[-1]

                frame_gates.append({
                    "id": f"t{t}_pz_{pz_L_idx}",
                    "type": "OP",
                    "qubits": [f"$q_{{{ionA}}}$", f"$q_{{{ionB}}}$"],
                    "edge": [fmt_coord(u), fmt_coord(v_node)],
                    "duration": 1,
                    "pz": "gate",
                })

        timeline_data.append({"t": t, "ions": frame_ions, "gates": frame_gates})

    full_payload = {
        "architecture": architecture_block,
        "grid": architecture_block["grid"],
        "sites": architecture_block["sites"],
        "pzs": architecture_block["pzs"],
        "innerPZEdges": inner_pz_edges_block,
        "timeline": timeline_data,
    }

    pathlib.Path(pathlib.Path(filename).parent).mkdir(exist_ok=True, parents=True)
    with pathlib.Path(filename).open("w", encoding="utf-8") as f:
        json.dump(full_payload, f, separators=(",", ":"))
    print(f"JSON Visualization written to: {filename}")


# ==========================================
# 2. Graph & Helpers
# ==========================================


def _canon(edge):
    return tuple(sorted(edge, key=lambda x: (x[0], x[1])))


def create_graph(m: int, n: int, v_size: int, h_size: int) -> Graph:
    # Generic grid generator
    m_ext = m + (v_size - 1) * (m - 1)
    n_ext = n + (h_size - 1) * (n - 1)
    g = nx.grid_2d_graph(m_ext, n_ext)

    # Prune horizontal edges
    for i in range(0, m_ext - v_size, v_size):
        for k in range(1, v_size):
            for j in range(n_ext - 1):
                if ((i + k, j), (i + k, j + 1)) in g.edges():
                    g.remove_edge((i + k, j), (i + k, j + 1))
    # Prune vertical edges
    for i in range(0, n_ext - h_size, h_size):
        for k in range(1, h_size):
            for j in range(m_ext - 1):
                if ((j, i + k), (j + 1, i + k)) in g.edges():
                    g.remove_edge((j, i + k), (j + 1, i + k))
    # Prune nodes
    for i in range(0, m_ext - v_size, v_size):
        for k in range(1, v_size):
            for j in range(0, n_ext - h_size, h_size):
                for p in range(1, h_size):
                    if (i + k, j + p) in g.nodes():
                        g.remove_node((i + k, j + p))

    nx.set_edge_attributes(g, "trap", "edge_type")
    for n in g.nodes():
        g.nodes[n]["node_type"] = "trap_node"
    # Junctions
    for i in range(0, m_ext, v_size):
        for j in range(0, n_ext, h_size):
            g.add_node((i, j), node_type="junction_node")
    return g


def map_logical_to_physical_split(edge, step=4):
    """
    Maps logical edge to the TWO middle physical edges.
    Assumes step=4 (4 edges, 5 nodes).
    Chain: 0 -- 1 -- 2 -- 3 -- 4
    Edges: (0,1), (1,2), (2,3), (3,4)
    PZ Left: (1,2), PZ Right: (2,3)
    """
    u, v = sorted(edge)
    y1, x1 = u[0] * step, u[1] * step
    y2, _x2 = v[0] * step, v[1] * step

    if y1 == y2:  # Horizontal
        # Chain along x
        e1 = ((y1, x1 + 1), (y1, x1 + 2))
        e2 = ((y1, x1 + 2), (y1, x1 + 3))
        return (e1, e2)
    # Vertical
    e1 = ((y1 + 1, x1), (y1 + 2, x1))
    e2 = ((y1 + 2, x1), (y1 + 3, x1))
    return (e1, e2)


def create_idc_dict(g):
    return {i: _canon(e) for i, e in enumerate(g.edges())}


def get_idx(d, e):
    target = _canon(e)
    for idx, edge in d.items():
        if edge == target:
            return idx
    msg = f"Edge {e} not found"
    raise ValueError(msg)


def get_idc(d, i):
    return d[i]


def get_path_between(g, e1, e2):
    path_nodes = nx.shortest_path(g, e1[0], e2[0])
    path_edges = list(pairwise(path_nodes))
    return [e for e in path_edges if _canon(e) != _canon(e1) and _canon(e) != _canon(e2)]


def get_moves_through_node(g, d, node):
    conn_edges = g.edges(node)
    conn_indices = [get_idx(d, e) for e in conn_edges]
    return list(permutations(conn_indices, 2))


def get_junctions(g, n1, n2, h, v):
    if g.nodes[n1]["node_type"] == "junction_node" and g.nodes[n2]["node_type"] == "junction_node":
        return [n1, n2]
    js = []
    limit = max(h, v)
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for k in range(1, limit + 1):
            nxt = (n1[0] + dy * k, n1[1] + dx * k)
            if nxt not in g:
                break
            if g.nodes[nxt]["node_type"] == "junction_node":
                js.append(nxt)
                break
    return js


def get_possible_moves_over_junction(nx_g, edge, h_size, v_size):
    n1, n2 = edge
    if nx_g.nodes[n1]["node_type"] != "junction_node":
        jcts = get_junctions(nx_g, n1, n2, h_size, v_size)
    else:
        jcts = get_junctions(nx_g, n2, n1, h_size, v_size)

    poss = []
    for j in jcts:
        for e in nx_g.edges(j):
            poss.append(e)

    for j in jcts:
        path_nodes = nx.shortest_path(nx_g, n1, j)
        path_edges = list(pairwise(path_nodes))
        for e_betw in path_edges:
            if _canon(e_betw) in [_canon(p) for p in poss]:
                poss = [p for p in poss if _canon(p) != _canon(e_betw)]

    return poss


def get_possible_previous_edges_from_junction_move(nx_g, edge, h_size, v_size):
    n1, n2 = edge
    target_jct = n1 if nx_g.nodes[n1]["node_type"] == "junction_node" else n2
    junction_neighbors = list(nx_g.neighbors(target_jct))
    current_arm_node = n2 if n1 == target_jct else n1
    possible_previous_edges = []

    for neighbor in junction_neighbors:
        if neighbor == current_arm_node:
            continue
        chain_edges = []
        curr = neighbor
        prev = target_jct
        chain_edges.append(_canon((prev, curr)))

        while nx_g.nodes[curr]["node_type"] != "junction_node":
            neighbors = list(nx_g.neighbors(curr))
            if len(neighbors) == 1:
                break
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            chain_edges.append(_canon((curr, next_node)))
            prev = curr
            curr = next_node

        possible_previous_edges.extend(chain_edges)
    return possible_previous_edges


def create_graph_dict(nx_g, func, h_size, v_size, edges="all"):
    d = {}
    if edges == "all":
        edges = list(nx_g.edges())
    for e in edges:
        d[e] = func(nx_g, e, h_size, v_size)
        d[tuple(reversed(e))] = func(nx_g, tuple(reversed(e)), h_size, v_size)
    return d


# ==========================================
# Hook Error Scheduling
def get_hook_error_dependencies(all_gates, ion_locations_dict):
    """
    Groups gates by Ancilla and enforces "N-shape" or "Z-shape" ordering
    to suppress hook errors.
    """
    # 1. Group gates by Ancilla (the ion connected to 3 or 4 others)
    ion_usage = {}
    for k, (u, v) in enumerate(all_gates):
        if u not in ion_usage:
            ion_usage[u] = []
        if v not in ion_usage:
            ion_usage[v] = []
        ion_usage[u].append(k)
        ion_usage[v].append(k)

    dependencies = []

    # 2. Sort interactions for each Ancilla
    for ion, gate_indices in ion_usage.items():
        if len(gate_indices) < 3:
            continue  # Ignore data qubits (usually 2 connections)

        # Get coordinates of the partner ion for each gate
        gates_with_coords = []
        for k in gate_indices:
            g = all_gates[k]
            partner = g[1] if g[0] == ion else g[0]

            # Use logical coordinates for sorting
            # ion_locations_dict should map ion_ID -> (row, col)
            if partner not in ion_locations_dict:
                continue
            r, c = ion_locations_dict[partner]
            gates_with_coords.append((k, r, c))

        # SORTING RULE: "N-Shape" (Row first, then Column)
        # Sorts Top-Left -> Top-Right -> Bottom-Left -> Bottom-Right
        gates_with_coords.sort(key=lambda x: (x[1], x[2]))

        # Create the chain: Gate 0 -> Gate 1 -> Gate 2 -> Gate 3
        sorted_indices = [x[0] for x in gates_with_coords]
        for i in range(len(sorted_indices) - 1):
            dependencies.append((sorted_indices[i], sorted_indices[i + 1]))

    return dependencies


def get_commutation_dependencies(all_gates, gate_types):
    """
    Ensures that for shared Data qubits, X-type gates happen before Z-type gates.
    Requires 'gate_types' dict: {gate_index: 'X' or 'Z'}
    """
    dependencies = []
    qubit_history = {}  # Ion -> list of (gate_index, type)

    for k, (u, v) in enumerate(all_gates):
        g_type = gate_types.get(k)
        if not g_type:
            continue

        # Record this gate for both ions
        for ion in [u, v]:
            if ion not in qubit_history:
                qubit_history[ion] = []
            qubit_history[ion].append((k, g_type))

    for ion, history in qubit_history.items():
        has_X = [h[0] for h in history if h[1] == "X"]
        has_Z = [h[0] for h in history if h[1] == "Z"]

        # If ion is touched by both types, force X -> Z
        if has_X and has_Z:
            for x_k in has_X:
                for z_k in has_Z:
                    dependencies.append((x_k, z_k))

    return dependencies


def classify_gates_by_type(all_gates, ion_locations_dict):
    """
    Returns a dict: {gate_index: 'X' or 'Z'}
    Based on the checkerboard pattern of the Ancilla ion.
    """
    gate_types = {}

    # 1. Identify which ion is the Ancilla for each gate
    # (The ancilla is the one with multiple gates attached to it)
    ion_counts = {}
    for u, v in all_gates:
        ion_counts[u] = ion_counts.get(u, 0) + 1
        ion_counts[v] = ion_counts.get(v, 0) + 1

    for k, (u, v) in enumerate(all_gates):
        # The ancilla is the ion with more connections (usually 4)
        # In a tie (rare/boundary), pick the one with higher connectivity
        ancilla = u if ion_counts[u] >= ion_counts[v] else v

        # 2. Get coordinates (row, col)
        if ancilla not in ion_locations_dict:
            continue
        r, c = ion_locations_dict[ancilla]

        # 3. Checkerboard Rule:
        # If (row + col) is Even -> Z-type. If Odd -> X-type.
        # (You can swap this definition, as long as it's consistent)
        # We define coordinates as integers for parity check
        r_int, c_int = round(r), round(c)

        if (r_int + c_int) % 2 == 0:
            gate_types[k] = "Z"
        else:
            gate_types[k] = "X"

    return gate_types


def get_consistency_dependencies(all_gates, gate_types):
    """
    Identifies pairs of Ancillas that share TWO data qubits.
    Returns a list of tuples: [ ((X1, Z1), (X2, Z2)), ... ]

    Logic:
    If Plaquette A (X) and Plaquette B (Z) share Data Qubits Q1 and Q2...
    Then the order of operations must match:
    (Time(X1) < Time(Z1)) == (Time(X2) < Time(Z2))
    """

    # 1. Identify Ancilla for every gate (needed to group boundaries)
    gate_to_ancilla = {}
    ion_counts = {}
    for u, v in all_gates:
        ion_counts[u] = ion_counts.get(u, 0) + 1
        ion_counts[v] = ion_counts.get(v, 0) + 1

    for k, (u, v) in enumerate(all_gates):
        # Ancilla has higher connectivity
        gate_to_ancilla[k] = u if ion_counts[u] >= ion_counts[v] else v

    # 2. Map Data Qubits to the gates acting on them
    # Format: data_id -> list of (gate_idx, type, ancilla_id)
    qubit_history = {}

    for k, (u, v) in enumerate(all_gates):
        g_type = gate_types.get(k)
        if not g_type:
            continue

        # Determine which ion is Data (the one with lower connectivity or just not the ancilla)
        ancilla = gate_to_ancilla[k]
        data = v if u == ancilla else u

        if data not in qubit_history:
            qubit_history[data] = []
        qubit_history[data].append((k, g_type, ancilla))

    # 3. Group by Ancilla Pair
    # Key: Tuple(Ancilla_ID_1, Ancilla_ID_2)
    # Value: List of gate pairs [(Gate_X, Gate_Z), (Gate_X, Gate_Z)]
    boundary_map = {}

    for _data_ion, history in qubit_history.items():
        x_interactions = [h for h in history if h[1] == "X"]
        z_interactions = [h for h in history if h[1] == "Z"]

        # We are looking for data qubits touched by exactly one X and one Z (bulk shared)
        if len(x_interactions) == 1 and len(z_interactions) == 1:
            gx, _type_x, and_x = x_interactions[0]
            gz, _type_z, and_z = z_interactions[0]

            # Create a unique key for this pair of Ancillas
            # Sort IDs to ensure (A,B) is same as (B,A)
            and_pair = tuple(sorted((and_x, and_z)))

            if and_pair not in boundary_map:
                boundary_map[and_pair] = []
            boundary_map[and_pair].append((gx, gz))

    # 4. Generate Constraints
    constraints = []
    for and_pair, gate_pairs in boundary_map.items():
        # We only care if they share 2 (or more) qubits.
        # If they share 1, order doesn't matter relative to another qubit.
        if len(gate_pairs) >= 2:
            # We enforce consistency between the first shared qubit and the second.
            # gate_pairs[0] is (X1, Z1)
            # gate_pairs[1] is (X2, Z2)
            constraints.append((gate_pairs[0], gate_pairs[1]))

    return constraints


# ==========================================
# 3. Solver Class
# ==========================================


class OptimalSyndromeSAT:
    def __init__(self, graph, h_size, v_size, ions, timesteps, pz_physical_pairs, gate_duration=2):
        """
        pz_physical_pairs: List of tuples [(edgeL, edgeR), ...]
        """
        self.graph = graph
        self.ions = ions
        self.timesteps = timesteps
        self.idc = create_idc_dict(graph)
        self.edge_list = [self.idc[i] for i in range(len(self.idc))]

        # Store Pairs of indices for Gate Logic
        self.pz_pairs_indices = []
        for eL, eR in pz_physical_pairs:
            self.pz_pairs_indices.append((get_idx(self.idc, eL), get_idx(self.idc, eR)))

        # For simple drawing, flatten these
        self.pz_edges = {_canon(e) for pair in pz_physical_pairs for e in pair}

        self.s = Solver()
        self.states = {}
        for t in range(timesteps):
            for e in range(len(graph.edges())):
                for i in ions:
                    self.states[t, e, i] = Bool(f"s_{t}_{e}_{i}")

        self.h_size = h_size
        self.v_size = v_size
        self.gate_duration = gate_duration

    def create_constraints(self, start_pos):
        junction_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]["node_type"] == "junction_node"]
        junction_edges = [list(self.graph.edges(n)) for n in junction_nodes]
        junction_edges_flat = [(sorted(e)[0], sorted(e)[1]) for sub in junction_edges for e in sub]

        junction_move_dict = create_graph_dict(self.graph, get_possible_moves_over_junction, self.h_size, self.v_size)
        prev_junction_move_dict = create_graph_dict(
            self.graph,
            get_possible_previous_edges_from_junction_move,
            self.h_size,
            self.v_size,
            edges=junction_edges_flat,
        )

        # 1. Start Positions
        for e_idx in range(len(self.edge_list)):
            edge = self.idc[e_idx]
            # Check if any ion starts here
            ions_here = [i for i, pos in start_pos.items() if _canon(pos) == edge]
            for i in self.ions:
                if i in ions_here:
                    self.s.add(self.states[0, e_idx, i])
                else:
                    self.s.add(Not(self.states[0, e_idx, i]))

        # 2. Conservation
        for t in range(1, self.timesteps):
            for i in self.ions:
                self.s.add(AtMost(*[self.states[t, e, i] for e in range(len(self.edge_list))], 1))
                self.s.add(AtLeast(*[self.states[t, e, i] for e in range(len(self.edge_list))], 1))

        # 3. Movement
        for t in range(self.timesteps - 1):
            for i in self.ions:
                for e_idx in range(len(self.edge_list)):
                    edge = self.idc[e_idx]
                    possible_next = junction_move_dict[edge].copy()
                    for neighbor_edge in self.graph.edges(edge):
                        possible_next.append(neighbor_edge)

                    next_conds = []
                    for n_edge in possible_next:
                        n_idx = get_idx(self.idc, n_edge)
                        path_edges = get_path_between(self.graph, edge, n_edge)
                        path_clear = And(*[
                            Not(self.states[t, get_idx(self.idc, pe), oi]) for pe in path_edges for oi in self.ions
                        ])
                        next_conds.append(And(self.states[t + 1, n_idx, i], path_clear))

                    self.s.add(Or(Not(self.states[t, e_idx, i]), And(self.states[t, e_idx, i], Or(*next_conds))))

        # 4. Junction Capacity
        for t in range(1, self.timesteps):
            for node in junction_nodes:
                self.s.add(
                    AtMost(
                        *[
                            And(
                                self.states[t, get_idx(self.idc, je), i],
                                Or(*[
                                    self.states[t - 1, get_idx(self.idc, prev), i]
                                    for prev in prev_junction_move_dict[je]
                                ]),
                            )
                            for je in self.graph.edges(node)
                            for i in self.ions
                        ],
                        1,
                    )
                )

        # 5. Node Capacity (Anti-Swap)
        for t in range(1, self.timesteps):
            for n in self.graph.nodes():
                moves = get_moves_through_node(self.graph, self.idc, n)
                if moves:
                    self.s.add(
                        AtMost(
                            *[
                                And(self.states[t, m[1], i], self.states[t - 1, m[0], i])
                                for i in self.ions
                                for m in moves
                            ],
                            1,
                        )
                    )

        # 6. Edge Capacity - STRICT 1 EVERYWHERE
        for t in range(1, self.timesteps):
            for e_idx in range(len(self.edge_list)):
                self.s.add(AtMost(*[self.states[t, e_idx, i] for i in self.ions], 1))

    def evaluate_free_optimization(self, all_gates, dependencies=None, consistency_deps=None):
        """
        dependencies: Strict order [(pre, post), ...] -> Time(post) >= Time(pre) + duration
        consistency_deps: Coupled order [((x1, z1), (x2, z2)), ...] -> (tx1 < tz1) == (tx2 < tz2)
        """
        if dependencies is None:
            dependencies = []
        if consistency_deps is None:
            consistency_deps = []

        num_gates = len(all_gates)
        gate_start = [[Bool(f"start_gate_{t}_{k}") for k in range(num_gates)] for t in range(self.timesteps)]

        # 1. Start Time Constraints
        for k in range(num_gates):
            valid_start = range(1, self.timesteps - self.gate_duration + 1)
            self.s.add(AtMost(*[gate_start[t][k] for t in range(self.timesteps)], 1))
            self.s.add(AtLeast(*[gate_start[t][k] for t in valid_start], 1))
            for t in [t for t in range(self.timesteps) if t not in valid_start]:
                self.s.add(Not(gate_start[t][k]))

        # 2. Gate Physics (Adjacency in Split PZs)
        for t in range(self.timesteps):
            for k, (ionA, ionB) in enumerate(all_gates):
                if t <= self.timesteps - self.gate_duration:
                    possible_locations = []
                    for idxL, idxR in self.pz_pairs_indices:
                        stay_LR = And(*[
                            And(self.states[t + dt, idxL, ionA], self.states[t + dt, idxR, ionB])
                            for dt in range(self.gate_duration)
                        ])
                        stay_RL = And(*[
                            And(self.states[t + dt, idxL, ionB], self.states[t + dt, idxR, ionA])
                            for dt in range(self.gate_duration)
                        ])
                        possible_locations.append(stay_LR)
                        possible_locations.append(stay_RL)
                    self.s.add(Or(Not(gate_start[t][k]), Or(*possible_locations)))

        # 3. Conflict Optimization
        ion_to_gate_indices = {}
        for k, (ionA, ionB) in enumerate(all_gates):
            if ionA not in ion_to_gate_indices:
                ion_to_gate_indices[ionA] = []
            if ionB not in ion_to_gate_indices:
                ion_to_gate_indices[ionB] = []
            ion_to_gate_indices[ionA].append(k)
            ion_to_gate_indices[ionB].append(k)

        for t in range(self.timesteps):
            for _ion, gate_indices in ion_to_gate_indices.items():
                if len(gate_indices) > 1:
                    active = []
                    for k in gate_indices:
                        relevant = []
                        for dt in range(self.gate_duration):
                            if t - dt >= 0:
                                relevant.append(gate_start[t - dt][k])
                        if relevant:
                            active.append(Or(*relevant))
                    self.s.add(AtMost(*active, 1))

        # 4. Strict Dependencies (Hook Errors)
        for pre, post in dependencies:
            t_pre = Sum([If(gate_start[t][pre], t, 0) for t in range(self.timesteps)])
            t_post = Sum([If(gate_start[t][post], t, 0) for t in range(self.timesteps)])
            self.s.add(t_post >= t_pre + self.gate_duration)

        # 5. Consistency Dependencies (Coupled Ordering)
        # Enforces: Order on Qubit 1 == Order on Qubit 2
        for pair_a, pair_b in consistency_deps:
            x1, z1 = pair_a
            x2, z2 = pair_b

            t_x1 = Sum([If(gate_start[t][x1], t, 0) for t in range(self.timesteps)])
            t_z1 = Sum([If(gate_start[t][z1], t, 0) for t in range(self.timesteps)])
            t_x2 = Sum([If(gate_start[t][x2], t, 0) for t in range(self.timesteps)])
            t_z2 = Sum([If(gate_start[t][z2], t, 0) for t in range(self.timesteps)])

            # Using Boolean Equality in Z3: (cond1) == (cond2)
            self.s.add((t_x1 < t_z1) == (t_x2 < t_z2))

        self.check = self.s.check()
        if self.check == sat:
            self.model = self.s.model()
            return True
        return False


# ==========================================
# 4. Debug Plotting
# ==========================================


def plot_results(solver: OptimalSyndromeSAT, model, title_prefix=""):
    timesteps = solver.timesteps
    graph = solver.graph
    ions = solver.ions
    pos = {node: (node[1], -node[0]) for node in graph.nodes()}
    all_edges = list(graph.edges())
    # Color PZ pairs
    pz_indices_flat = set()
    for l, r in solver.pz_pairs_indices:
        pz_indices_flat.add(l)
        pz_indices_flat.add(r)

    bg_pz = [solver.idc[i] for i in pz_indices_flat]
    bg_norm = [e for e in all_edges if _canon(e) not in [_canon(x) for x in bg_pz]]
    node_colors = [
        "#77DD77" if graph.nodes[n].get("node_type") == "junction_node" else "#AEC6CF" for n in graph.nodes()
    ]

    _fig, axes = plt.subplots(1, timesteps, figsize=(timesteps * 4, 5))
    if timesteps == 1:
        axes = [axes]

    for t in range(timesteps):
        ax = axes[t]
        ax.set_title(f"{title_prefix} T={t}")
        edge_labels, active_edges = {}, []
        for e_idx in range(len(solver.edge_list)):
            edge = solver.idc[e_idx]
            ions_on = [ion for ion in ions if bool(model.evaluate(solver.states[t, e_idx, ion]))]
            if ions_on:
                edge_labels[edge] = str(ions_on)
                active_edges.append(edge)

        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=bg_norm, edge_color="lightgray", width=2)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=bg_pz, edge_color="#FFB6C1", width=3)
        if active_edges:
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=active_edges, edge_color="orange", width=4)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=8)
        ax.axis("off")
    plt.tight_layout()
    # plt.show()


# ==========================================
# 5. Execution
# ==========================================


if __name__ == "__main__":
    m, n = 5, 5
    # CHANGED: Increased physical chain size to 4 to allow 2 separate PZ spots
    v_size, h_size = 4, 4
    print(f"--- Setting up Distance 3 Optimization (Grid {m}x{n}, Chain {v_size}) ---")

    x0, x1, y0, y1 = 1, n - 1, 1, m - 2
    pz_edges_logical = (
        [((y, x), (y + 1, x)) for x in range(x0, x1) for y in range(y0, y1)]
        + [((0, x), (1, x)) for x in range(x0, x1) if x % 2 == 0]
        + [((y1, x - 1), (y1 + 1, x - 1)) for x in range(x0, x1) if x % 2 == 0]
        + [((y - 1, 0), (y, 0)) for y in range(y0, y1) if y % 2 == 0]
        + [((y, n - 1), (y + 1, n - 1)) for y in range(y0, y1) if y % 2 == 0]
    )
    data_edges_logical = [((y, x), (y, x + 1)) for y in range(1, m - 1) for x in range(n - 1)]
    all_qubits_logical = {_canon(e) for e in pz_edges_logical + data_edges_logical}

    ion_index = 0
    ions_dict = {}
    for y in range(m):
        for x in range(n):
            if x < n - 1:
                e = _canon(((y, x), (y, x + 1)))
                if e in all_qubits_logical:
                    ions_dict[ion_index] = e
                    ion_index += 1
            if y < m - 1:
                e = _canon(((y, x), (y + 1, x)))
                if e in all_qubits_logical:
                    ions_dict[ion_index] = e
                    ion_index += 1

    all_ions = list(ions_dict.keys())
    gates = []
    edge_to_ion = {v: k for k, v in ions_dict.items()}
    for pz_e in pz_edges_logical:
        pz_e = _canon(pz_e)
        if pz_e not in edge_to_ion:
            continue
        pz_ion = edge_to_ion[pz_e]
        for node in pz_e:
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adj_edge = _canon((node, (node[0] + dy, node[1] + dx)))
                if adj_edge in edge_to_ion and adj_edge != pz_e:
                    g = tuple(sorted((pz_ion, edge_to_ion[adj_edge])))
                    if g not in gates:
                        gates.append(g)

    # gates = gates[:18] # Test larger batch
    print(f"Scheduling {len(gates)} gates for {len(all_ions)} ions.")

    # Create larger graph
    nx_g = create_graph(m, n, v_size, h_size)

    # Map to Split PZs (Left/Right)
    # Start positions: Pick Left side of the split PZ for consistency? Or distribute?
    # Let's say all ions start in the "Left" slot of their PZ area.
    start_pos = {}
    pz_pairs_phys = []  # List of (e1, e2)

    for e in pz_edges_logical:
        (e1, e2) = map_logical_to_physical_split(e, 4)
        pz_pairs_phys.append((e1, e2))

    # Map starts for ALL ions (Data and PZ)
    # Since all logical qubits are edges, they all map to split pairs.
    for i, e_log in ions_dict.items():
        (e1, e2) = map_logical_to_physical_split(e_log, 4)
        start_pos[i] = e1  # Start in slot 1

    # Dependency Generation #####
    # 1. PREPARE COORDINATES FOR HOOK SORTING
    # Map ion_ID back to logical coordinates (row, col)
    # We use the center of the edge they live on as a proxy
    ion_coords = {}
    for ion_id, edge in ions_dict.items():
        # edge is ((r,c), (r,c)), average them
        r_avg = (edge[0][0] + edge[1][0]) / 2.0
        c_avg = (edge[0][1] + edge[1][1]) / 2.0
        ion_coords[ion_id] = (r_avg, c_avg)

    gate_types = classify_gates_by_type(gates, ion_coords)

    # Debug print to verify checkboard
    x_count = sum(1 for t in gate_types.values() if t == "X")
    z_count = sum(1 for t in gate_types.values() if t == "Z")
    print(f"Detected {x_count} X-type gates and {z_count} Z-type gates.")

    # # 2. GENERATE DEPENDENCIES
    # # A. Hook Errors (Local N-shape)
    # hook_deps = get_hook_error_dependencies(gates, ion_coords)

    # # B. Commutation (X before Z for shared data qubits)
    # # Now this works because gate_types is populated!
    # comm_deps = get_commutation_dependencies(gates, gate_types)

    # print(f"Constraints: {len(hook_deps)} Hook rules, {len(comm_deps)} Commutation rules.")

    # all_deps = hook_deps + comm_deps

    # ###### Test for cycles in dependencies ######
    # print(f"Checking for dependency cycles in {len(all_deps)} constraints...")
    # # Build a simple graph of dependencies
    # dep_graph = nx.DiGraph()
    # dep_graph.add_edges_from(all_deps)

    # try:
    #     cycle = nx.find_cycle(dep_graph)
    #     print("\nCRITICAL ERROR: A dependency cycle was detected!")
    #     print(f"The solver will ALWAYS return UNSAT because of this loop:")
    #     print(cycle)
    #     print("You must relax constraints (likely the X->Z strict ordering).")
    #     exit() # Stop the script
    # except nx.NetworkXNoCycle:
    #     print("Dependency graph is Acyclic (Valid). Proceeding to solver...")

    # 1. GENERATE GATE TYPES
    gate_types = classify_gates_by_type(gates, ion_coords)

    # 2. GENERATE DEPENDENCIES
    # A. Hook Errors (KEEP THIS - Local N-shape is vital)
    hook_deps = get_hook_error_dependencies(gates, ion_coords)

    # B. Consistency (NEW - Replaces strict commutation)
    consist_deps = get_consistency_dependencies(gates, gate_types)

    # NOTE: strict comm_deps are removed to prevent cycles.
    all_deps = hook_deps

    print(f"Constraints: {len(hook_deps)} Hook rules, {len(consist_deps)} Consistency rules.")

    # Cycle check applies ONLY to strict dependencies (hook_deps)
    print("Checking for cycles in strict dependencies...")
    dep_graph = nx.DiGraph()
    dep_graph.add_edges_from(all_deps)
    if not nx.is_directed_acyclic_graph(dep_graph):
        print("CRITICAL ERROR: Hook dependencies form a cycle. Check geometry.")
        sys.exit()
    else:
        print("Dependency graph is Acyclic. Proceeding.")

    found = False
    # Start search around T=20
    for t in range(16, 40):
        print(f"Checking T={t}...")
        solver = OptimalSyndromeSAT(nx_g, h_size, v_size, all_ions, t, pz_physical_pairs=pz_pairs_phys, gate_duration=2)
        solver.create_constraints(start_pos)

        import time

        ts = time.time()

        # PASS BOTH SETS OF DEPENDENCIES
        if solver.evaluate_free_optimization(gates, dependencies=all_deps, consistency_deps=consist_deps):
            print(f"MINIMUM FOUND: T={t}, Time: {time.time() - ts:.2f}s")
            plot_results(solver, solver.model)
            generate_visualization_json(solver, solver.model, "benchmarks/solution_d3_final.json", m, n, v_size, h_size)
            found = True
            break

    if not found:
        print(f"No solution found. Time: {time.time() - ts:.2f}s")
