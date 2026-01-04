from __future__ import annotations

import itertools
import json
import os
from itertools import pairwise
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from z3 import And, AtLeast, AtMost, Bool, Not, Or, Solver, sat

if TYPE_CHECKING:
    from collections.abc import Callable
    Edge = tuple[tuple[int, int], tuple[int, int]]
    Node = tuple[int, int]
    Graph = nx.Graph

# Viz helper not used anymore (only for SyndromSAT_dev)

# ==========================================
# 1. Visualization Generator (Embedded)
# ==========================================

def generate_visualization_json(solver, model, filename, m, n, v, h):
    """
    Generates the visualization JSON. 
    Detects gates when 2 ions overlap on an edge marked in solver.data_edges.
    """
    
    # Helper to format coordinates for the JS visualizer
    def fmt_coord(node): return f"({node[0]}, {node[1]})"

    # Map index back to edge tuple
    idx_to_edge = {i: e for i, e in enumerate(solver.edge_list)}

    # 1. Architecture: Draw PZs (Stabilizers) as the "Background"
    inner_pz_edges_block = []
    for edge in solver.pz_edges:
        u, v_node = sorted(edge) # Sort nodes for consistent drawing
        inner_pz_edges_block.append([fmt_coord(u), fmt_coord(v_node)])

    architecture_block = {
        "grid": {"rows": m, "cols": n},
        "sites": {"vertical": v, "horizontal": h}, # Physical spacing
        "pzs": {"top": False, "right": False, "bottom": False, "left": False},
        "innerPZEdges": inner_pz_edges_block
    }

    timeline_data = []
    print(f"Generating timeline for {solver.timesteps} steps...")

    for t in range(solver.timesteps):
        frame_ions = []
        frame_gates = []

        # A. Ions
        for ion in solver.ions:
            for edge_idx in range(len(solver.edge_list)):
                if bool(model.evaluate(solver.states[(t, edge_idx, ion)])):
                    edge = idx_to_edge[edge_idx]
                    u, v_node = sorted(edge)
                    frame_ions.append({
                        "id": f"$q_{{{ion}}}$", 
                        "edge": [fmt_coord(u), fmt_coord(v_node)]
                    })
                    break 

        # B. Gates (Operations)
        # Check edges defined as "Gate Locations" (passed as data_edges to solver)
        for edge_idx, edge in enumerate(solver.edge_list):
            # We canonicalize before checking membership
            canon_edge = tuple(sorted(edge, key=lambda x: (x[0], x[1])))
            
            if canon_edge in solver.data_edges:
                ions_here = []
                for ion in solver.ions:
                    if bool(model.evaluate(solver.states[(t, edge_idx, ion)])):
                        ions_here.append(f"$q_{{{ion}}}$")
                
                # If 2 ions are together, it's a GATE
                if len(ions_here) >= 2:
                    u, v_node = sorted(edge)
                    frame_gates.append({
                        "id": f"t{t}_e{edge_idx}",
                        "type": "OP",
                        "qubits": ions_here,
                        "edge": [fmt_coord(u), fmt_coord(v_node)],
                        "duration": 1,
                        "pz": f"pz_{u[0]}_{u[1]}"
                    })

        timeline_data.append({"t": t, "ions": frame_ions, "gates": frame_gates})

    full_payload = {
        "architecture": architecture_block,
        "grid": architecture_block["grid"],
        "sites": architecture_block["sites"],
        "pzs": architecture_block["pzs"],
        "innerPZEdges": inner_pz_edges_block,
        "timeline": timeline_data
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(full_payload, f, separators=(',', ':'))
    print(f"JSON Visualization written to: {filename}")


# ==========================================
# 2. Graph & Helpers
# ==========================================

def _canon(edge):
    """Sorts edge nodes by (row, col) to ensure consistent tuple representation."""
    return tuple(sorted(edge, key=lambda x: (x[0], x[1])))

def create_graph(m: int, n: int, v_size: int = 3, h_size: int = 3) -> Graph:
    m_ext = m + (v_size - 1) * (m - 1)
    n_ext = n + (h_size - 1) * (n - 1)
    g = nx.grid_2d_graph(m_ext, n_ext)
    
    # Prune to skeleton
    for i in range(0, m_ext - v_size, v_size):
        for k in range(1, v_size):
            for j in range(n_ext - 1):
                if ((i + k, j), (i + k, j + 1)) in g.edges(): g.remove_edge((i + k, j), (i + k, j + 1))
    for i in range(0, n_ext - h_size, h_size):
        for k in range(1, h_size):
            for j in range(m_ext - 1):
                if ((j, i + k), (j + 1, i + k)) in g.edges(): g.remove_edge((j, i + k), (j + 1, i + k))
    for i in range(0, m_ext - v_size, v_size):
        for k in range(1, v_size):
            for j in range(0, n_ext - h_size, h_size):
                for p in range(1, h_size):
                    if (i + k, j + p) in g.nodes(): g.remove_node((i + k, j + p))
                    
    nx.set_edge_attributes(g, "trap", "edge_type")
    for n in g.nodes(): g.nodes[n]["node_type"] = "trap_node"
    for i in range(0, m_ext, v_size):
        for j in range(0, n_ext, h_size):
            g.add_node((i, j), node_type="junction_node")
    return g

def map_logical_to_physical_middle(edge, step=3):
    u, v = sorted(edge)
    y1, x1 = u[0]*step, u[1]*step
    y2, x2 = v[0]*step, v[1]*step
    if y1 == y2: return ((y1, x1+1), (y1, x1+2))
    return ((y1+1, x1), (y1+2, x1))

def create_idc_dict(g): 
    # Use _canon to ensure the dictionary keys match the solver sets
    return {i: _canon(e) for i, e in enumerate(g.edges())}

def get_idx(d, e): 
    # Search by value (canonical tuple)
    target = _canon(e)
    for idx, edge in d.items():
        if edge == target:
            return idx
    raise ValueError(f"Edge {e} not found in dictionary")

def get_idc(d, i): return d[i]

def get_path_between(g, e1, e2):
    path_nodes = nx.shortest_path(g, e1[0], e2[0])
    path_edges = list(pairwise(path_nodes))
    path_edges = [e for e in path_edges if _canon(e)!=_canon(e1) and _canon(e)!=_canon(e2)]
    return path_edges

def get_moves_through_node(g, d, node):
    # Retrieve indices for all edges connected to this node
    conn_edges = g.edges(node)
    conn_indices = [get_idx(d, e) for e in conn_edges]
    return list(itertools.permutations(conn_indices, 2))

def get_junctions(g, n1, n2, h, v):
    if g.nodes[n1]['node_type'] == 'junction_node' and g.nodes[n2]['node_type'] == 'junction_node':
        return [n1, n2]
    js = []
    limit = max(h, v)
    for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
        for k in range(1, limit+1):
            nxt = (n1[0]+dy*k, n1[1]+dx*k)
            if nxt not in g: break
            if g.nodes[nxt]['node_type'] == 'junction_node': js.append(nxt); break
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
                # Remove carefully
                poss = [p for p in poss if _canon(p) != _canon(e_betw)]
            
    return poss

def get_possible_previous_edges_from_junction_move(nx_g, edge, h_size, v_size):
    n1, n2 = edge
    target_jct = n1 if nx_g.nodes[n1]["node_type"] == "junction_node" else n2
    junction_neighbors = list(nx_g.neighbors(target_jct))
    current_arm_node = n2 if n1 == target_jct else n1
    possible_previous_edges = []

    for neighbor in junction_neighbors:
        if neighbor == current_arm_node: continue
        chain_edges = []
        curr = neighbor
        prev = target_jct
        chain_edges.append(_canon((prev, curr)))
        
        while nx_g.nodes[curr]["node_type"] != "junction_node":
            neighbors = list(nx_g.neighbors(curr))
            if len(neighbors) == 1: break 
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            chain_edges.append(_canon((curr, next_node)))
            prev = curr
            curr = next_node
            
        possible_previous_edges.extend(chain_edges)
    return possible_previous_edges

def create_graph_dict(nx_g, func, h_size, v_size, edges="all"):
    d = {}
    if edges == "all": edges = list(nx_g.edges())
    for e in edges:
        d[e] = func(nx_g, e, h_size, v_size)
        d[tuple(reversed(e))] = func(nx_g, tuple(reversed(e)), h_size, v_size)
    return d


# ==========================================
# 3. Solver Class
# ==========================================

class OptimalSyndromeSAT:
    def __init__(self, graph, h_size, v_size, ions, timesteps, pz_physical_edges, data_physical_edges=None, gate_duration=2):
        self.graph = graph
        self.ions = ions
        self.timesteps = timesteps
        self.idc = create_idc_dict(graph)
        
        # Helper Compatibility
        self.edge_list = [self.idc[i] for i in range(len(self.idc))]
        self.pz_edges = set([_canon(e) for e in pz_physical_edges])
        
        # CRITICAL: Since gates happen in PZs, we treat PZs as "Data Edges" for the visualizer
        # We merge logical data edges (if mapped) and pz edges so the viz detects overlap in either
        input_data_edges = set([_canon(e) for e in (data_physical_edges or [])])
        self.data_edges = self.pz_edges.union(input_data_edges)
        
        self.pz_indices = [get_idx(self.idc, e) for e in pz_physical_edges]
        
        self.s = Solver()
        self.states = {}
        for t in range(timesteps):
            for e in range(len(graph.edges())):
                for i in ions:
                    self.states[(t, e, i)] = Bool(f"s_{t}_{e}_{i}")
        
        self.h_size = h_size
        self.v_size = v_size
        self.gate_duration = gate_duration

    def create_constraints(self, start_pos):
        # Pre-calc dictionaries
        junction_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]["node_type"] == "junction_node"]
        junction_edges = [list(self.graph.edges(n)) for n in junction_nodes]
        junction_edges_flat = [(sorted(e)[0], sorted(e)[1]) for sub in junction_edges for e in sub]

        junction_move_dict = create_graph_dict(self.graph, get_possible_moves_over_junction, self.h_size, self.v_size)
        prev_junction_move_dict = create_graph_dict(self.graph, get_possible_previous_edges_from_junction_move, self.h_size, self.v_size, edges=junction_edges_flat)

        # 1. Start Positions
        for e_idx in range(len(self.edge_list)):
            edge = self.idc[e_idx]
            start_ions = [i for i, pos in start_pos.items() if _canon(pos) == edge]
            for i in self.ions:
                if i in start_ions: self.s.add(self.states[(0, e_idx, i)])
                else: self.s.add(Not(self.states[(0, e_idx, i)]))

        # 2. Conservation
        for t in range(1, self.timesteps):
            for i in self.ions:
                self.s.add(AtMost(*[self.states[(t, e, i)] for e in range(len(self.edge_list))], 1))
                self.s.add(AtLeast(*[self.states[(t, e, i)] for e in range(len(self.edge_list))], 1))

        # 3. Movement
        for t in range(self.timesteps - 1):
            for i in self.ions:
                for e_idx in range(len(self.edge_list)):
                    edge = self.idc[e_idx]
                    possible_next = junction_move_dict[edge].copy()
                    for neighbor_edge in self.graph.edges(edge): possible_next.append(neighbor_edge)
                        
                    next_conds = []
                    for n_edge in possible_next:
                        n_idx = get_idx(self.idc, n_edge)
                        path_edges = get_path_between(self.graph, edge, n_edge)
                        path_clear = And(*[Not(self.states[(t, get_idx(self.idc, pe), oi)]) for pe in path_edges for oi in self.ions])
                        next_conds.append(And(self.states[(t+1, n_idx, i)], path_clear))
                    
                    self.s.add(Or(Not(self.states[(t, e_idx, i)]), And(self.states[(t, e_idx, i)], Or(*next_conds))))

        # 4. Junction Capacity
        for t in range(1, self.timesteps):
            for node in junction_nodes:
                self.s.add(AtMost(*[And(self.states[(t, get_idx(self.idc, je), i)], Or(*[self.states[(t-1, get_idx(self.idc, prev), i)] for prev in prev_junction_move_dict[je]])) for je in self.graph.edges(node) for i in self.ions], 1))

        # 5. Node Capacity (Anti-Swap)
        for t in range(1, self.timesteps):
            for n in self.graph.nodes():
                moves = get_moves_through_node(self.graph, self.idc, n)
                if moves:
                    self.s.add(AtMost(*[And(self.states[(t, m[1], i)], self.states[(t-1, m[0], i)]) for i in self.ions for m in moves], 1))

        # 6. Edge Capacity
        for t in range(1, self.timesteps):
            for e_idx in range(len(self.edge_list)):
                limit = 2 if e_idx in self.pz_indices else 1
                self.s.add(AtMost(*[self.states[(t, e_idx, i)] for i in self.ions], limit))

    def evaluate_free_optimization(self, all_gates):
            num_gates = len(all_gates)
            gate_start = [[Bool(f"start_gate_{t}_{k}") for k in range(num_gates)] for t in range(self.timesteps)]
            
            # 1. Gate Start Time Validity (Must happen once)
            for k in range(num_gates):
                valid_start = range(1, self.timesteps - self.gate_duration + 1)
                self.s.add(AtMost(*[gate_start[t][k] for t in range(self.timesteps)], 1))
                self.s.add(AtLeast(*[gate_start[t][k] for t in valid_start], 1))
                for t in [t for t in range(self.timesteps) if t not in valid_start]: 
                    self.s.add(Not(gate_start[t][k]))

            # 2. Gate Location Constraints (THE MISSING PART)
            # This links the abstract "gate start" to physical ion positions
            for t in range(self.timesteps):
                for k, (ionA, ionB) in enumerate(all_gates):
                    if t <= self.timesteps - self.gate_duration:
                        possible_locations = []
                        # The ions must be together at ANY valid PZ edge for the duration
                        for pz_idx in self.pz_indices:
                            stay_at_pz = And(*[
                                And(self.states[(t + dt, pz_idx, ionA)], 
                                    self.states[(t + dt, pz_idx, ionB)]) 
                                for dt in range(self.gate_duration)
                            ])
                            possible_locations.append(stay_at_pz)
                        
                        # If gate K starts at T, the ions MUST be at a PZ
                        self.s.add(Or(Not(gate_start[t][k]), Or(*possible_locations)))

            # 3. New Optimization: Interaction Conflicts
            ion_to_gate_indices = {}
            for k, (ionA, ionB) in enumerate(all_gates):
                if ionA not in ion_to_gate_indices: ion_to_gate_indices[ionA] = []
                if ionB not in ion_to_gate_indices: ion_to_gate_indices[ionB] = []
                ion_to_gate_indices[ionA].append(k)
                ion_to_gate_indices[ionB].append(k)

            for t in range(self.timesteps):
                for ion, gate_indices in ion_to_gate_indices.items():
                    if len(gate_indices) > 1:
                        active_conditions = []
                        for k in gate_indices:
                            relevant_starts = []
                            for dt in range(self.gate_duration):
                                if t - dt >= 0:
                                    relevant_starts.append(gate_start[t - dt][k])
                            if relevant_starts:
                                active_conditions.append(Or(*relevant_starts))
                        
                        self.s.add(AtMost(*active_conditions, 1))

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
    pz_canon = {_canon(solver.idc[idx]) for idx in solver.pz_indices}
    all_edges = list(graph.edges())
    bg_pz = [e for e in all_edges if _canon(e) in pz_canon]
    bg_norm = [e for e in all_edges if _canon(e) not in pz_canon]
    node_colors = ['#77DD77' if graph.nodes[n].get('node_type') == 'junction_node' else '#AEC6CF' for n in graph.nodes()]

    fig, axes = plt.subplots(1, timesteps, figsize=(timesteps * 4, 5))
    if timesteps == 1: axes = [axes]

    for t in range(timesteps):
        ax = axes[t]
        ax.set_title(f"{title_prefix} T={t}")
        edge_labels, active_edges = {}, []
        for e_idx in range(len(solver.edge_list)):
            edge = solver.idc[e_idx]
            ions_on = [ion for ion in ions if bool(model.evaluate(solver.states[(t, e_idx, ion)]))]
            if ions_on:
                edge_labels[edge] = str(ions_on)
                active_edges.append(edge)
        
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=bg_norm, edge_color='lightgray', width=2)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=bg_pz, edge_color='#FFB6C1', width=3)
        if active_edges: nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=active_edges, edge_color='orange', width=4)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. Execution
# ==========================================

if __name__ == "__main__":
    m, n = 5, 5
    v_size, h_size = 3, 3
    print(f"--- Setting up Distance 3 Optimization (Grid {m}x{n}) ---")
    
    x0, x1, y0, y1 = 1, n - 1, 1, m - 2
    pz_edges_logical = [((y, x), (y + 1, x)) for x in range(x0, x1) for y in range(y0, y1)] + \
                       [((0, x), (1, x)) for x in range(x0, x1) if x % 2 == 0] + \
                       [((y1, x - 1), (y1 + 1, x - 1)) for x in range(x0, x1) if x % 2 == 0] + \
                       [((y - 1, 0), (y, 0)) for y in range(y0, y1) if y % 2 == 0] + \
                       [((y, n - 1), (y + 1, n - 1)) for y in range(y0, y1) if y % 2 == 0]
    data_edges_logical = [((y, x), (y, x + 1)) for y in range(1, m - 1) for x in range(0, n - 1)]
    all_qubits_logical = set([_canon(e) for e in pz_edges_logical + data_edges_logical])
    
    ion_index = 0
    ions_dict = {}
    for y in range(m):
        for x in range(n):
            if x < n - 1:
                e = _canon(((y, x), (y, x + 1)))
                if e in all_qubits_logical: ions_dict[ion_index] = e; ion_index += 1
            if y < m - 1:
                e = _canon(((y, x), (y + 1, x)))
                if e in all_qubits_logical: ions_dict[ion_index] = e; ion_index += 1
    
    all_ions = list(ions_dict.keys())
    gates = []
    edge_to_ion = {v: k for k, v in ions_dict.items()}
    for pz_e in pz_edges_logical:
        pz_e = _canon(pz_e)
        if pz_e not in edge_to_ion: continue
        pz_ion = edge_to_ion[pz_e]
        for node in pz_e:
            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                adj_edge = _canon((node, (node[0]+dy, node[1]+dx)))
                if adj_edge in edge_to_ion and adj_edge != pz_e:
                    g = tuple(sorted((pz_ion, edge_to_ion[adj_edge])))
                    if g not in gates: gates.append(g)
    
    # Test a batch of gates
    #gates = gates[:30]
    print(f"Scheduling {len(gates)} gates for {len(all_ions)} ions.")

    nx_g = create_graph(m, n, v_size, h_size)
    start_pos = {i: map_logical_to_physical_middle(e, 3) for i, e in ions_dict.items()}
    pz_phys = [map_logical_to_physical_middle(e, 3) for e in pz_edges_logical]
    
    # NOTE: We treat PZs as valid locations for gates (capacity=2). 
    # The solver class will now use these for viz detection automatically.
    
    found = False
    for t in range(14, 30):
        print(f"Checking T={t}...")
        solver = OptimalSyndromeSAT(
            nx_g, h_size, v_size, all_ions, t, 
            pz_physical_edges=pz_phys,
            data_physical_edges=None, # Viz will default to using PZ edges, which is correct
            gate_duration=2
        )
        solver.create_constraints(start_pos)
        import time
        time_start = time.time()
        if solver.evaluate_free_optimization(gates):
            print(f"MINIMUM FOUND: T={t}")
            time_end = time.time()
            print(f"Solved in {time_end - time_start:.2f} seconds.")
            plot_results(solver, solver.model)
            generate_visualization_json(solver, solver.model, "benchmarks/solution_d3_viz.json", m, n, v_size, h_size)
            found = True
            break
            
    if not found:
        print("No solution found.")