from __future__ import annotations
import networkx as nx
from z3 import And, AtMost, AtLeast, Bool, Not, Or, Implies, Solver, sat, unsat
import itertools

from viz_helper import generate_visualization_json

# ==========================================
# 1. Graph Construction
# ==========================================

def _canon(edge):
    return tuple(sorted(edge))

def build_user_graph(m, n):
    x0, x1, y0, y1 = 1, n - 1, 1, m - 2
    
    # PZ edges (Vertical)
    pz_edges = [((y, x), (y + 1, x)) for x in range(x0, x1) for y in range(y0, y1)]
    pz_edges.extend([((0, x), (1, x)) for x in range(x0, x1) if x % 2 == 0])
    pz_edges.extend([((y1, x - 1), (y1 + 1, x - 1)) for x in range(x0, x1) if x % 2 == 0])
    pz_edges.extend([((y - 1, 0), (y, 0)) for y in range(y0, y1) if y % 2 == 0])
    pz_edges.extend([((y, n - 1), (y + 1, n - 1)) for y in range(y0, y1) if y % 2 == 0])

    # Data edges (Horizontal)
    data_qubits = [((y, x), (y, x + 1)) for y in range(1, m - 1) for x in range(0, n - 1)]

    pz_edges = [_canon(e) for e in pz_edges]
    data_qubits = [_canon(e) for e in data_qubits]
    
    nx_g = nx.Graph()
    for e in pz_edges:
        nx_g.add_edge(e[0], e[1], edge_type="pz_trap", capacity=1)
    for e in data_qubits:
        nx_g.add_edge(e[0], e[1], edge_type="data_trap", capacity=2)

    for node in nx_g.nodes():
        degree = nx_g.degree(node)
        nx_g.nodes[node]['node_type'] = 'junction_node' if degree > 2 else 'trap_node'

    ion_index = 0
    ions_start_dict = {}
    for e in data_qubits:
        ions_start_dict[ion_index] = e
        ion_index += 1
    for e in pz_edges:
        ions_start_dict[ion_index] = e
        ion_index += 1

    return nx_g, pz_edges, data_qubits, ions_start_dict

def get_possible_moves(nx_g, edge):
    neighbors = []
    u, v = edge
    for n in nx_g.neighbors(u):
        if n != v: neighbors.append(_canon((u, n)))
    for n in nx_g.neighbors(v):
        if n != u: neighbors.append(_canon((v, n)))
    return neighbors

def generate_hook_error_schedule(pz_edges, data_edges):
    schedule = {}
    data_edge_set = set(data_edges)
    
    for pz in pz_edges:
        u, v = sorted(pz)
        y, x = u
        is_type_z = (y + x) % 2 == 0
        
        tl = _canon(((y, x-1), (y, x)))
        tr = _canon(((y, x), (y, x+1)))
        bl = _canon(((y+1, x-1), (y+1, x)))
        br = _canon(((y+1, x), (y+1, x+1)))
        
        targets_map = {'TL': tl, 'TR': tr, 'BL': bl, 'BR': br}
        order = ['TL', 'TR', 'BL', 'BR'] if is_type_z else ['TL', 'BL', 'TR', 'BR']
            
        final_sequence = []
        for pos in order:
            edge_cand = targets_map[pos]
            if edge_cand in data_edge_set:
                final_sequence.append(edge_cand)
        if final_sequence:
            schedule[pz] = final_sequence
    return schedule

# ==========================================
# 3. The SAT Solver Class
# ==========================================

class SyndromeSAT:
    def __init__(self, graph, ions_dict, timesteps, data_edges, pz_edges, gate_duration=1):
        self.graph = graph
        self.ions_dict = ions_dict
        self.ions = list(ions_dict.keys())
        self.timesteps = timesteps
        self.data_edges = set([_canon(e) for e in data_edges])
        self.pz_edges = set([_canon(e) for e in pz_edges])
        self.gate_duration = gate_duration
        
        self.edge_list = [_canon(e) for e in graph.edges()]
        self.edge_to_idx = {e: i for i, e in enumerate(self.edge_list)}

        self.s = Solver()
        self.states = {}
        for t in range(timesteps):
            for idx in range(len(self.edge_list)):
                for ion in self.ions:
                    self.states[(t, idx, ion)] = Bool(f"q_{t}_e{idx}_i{ion}")
        
        # Store start variables for verification later
        # Key: (ion_id, step_index, t) -> Bool
        self.gate_start_vars = {} 

    def create_constraints(self):
        # 1. Initial Positions
        for ion, start_edge in self.ions_dict.items():
            start_idx = self.edge_to_idx[_canon(start_edge)]
            self.s.add(self.states[(0, start_idx, ion)])
            for idx in range(len(self.edge_list)):
                if idx != start_idx:
                    self.s.add(Not(self.states[(0, idx, ion)]))

        # 2. Movement
        for t in range(self.timesteps - 1):
            for ion in self.ions:
                for idx, edge in enumerate(self.edge_list):
                    current_pos = self.states[(t, idx, ion)]
                    neighbors = get_possible_moves(self.graph, edge)
                    neighbor_indices = [self.edge_to_idx[n] for n in neighbors if n in self.edge_to_idx]
                    next_allowed = [self.states[(t+1, idx, ion)]] + \
                                   [self.states[(t+1, n_idx, ion)] for n_idx in neighbor_indices]
                    self.s.add(Or(Not(current_pos), Or(*next_allowed)))

        # 3. Capacity
        for t in range(1, self.timesteps):
            for ion in self.ions:
                self.s.add(AtMost(*[self.states[(t, idx, ion)] for idx in range(len(self.edge_list))], 1))
            for idx, edge in enumerate(self.edge_list):
                cap = 2 if edge in self.data_edges else 1
                ions_on_edge = [self.states[(t, idx, ion)] for ion in self.ions]
                self.s.add(AtMost(*ions_on_edge, cap))
        
        # 4. Data Ions Static
        for ion_id, start_edge in self.ions_dict.items():
            if start_edge in self.data_edges:
                start_idx = self.edge_to_idx[start_edge]
                for t in range(self.timesteps):
                    self.s.add(self.states[(t, start_idx, ion_id)])

    def add_physical_constraints(self):
        adj_map = {}
        for idx, edge in enumerate(self.edge_list):
            neighbors = get_possible_moves(self.graph, edge)
            adj_map[idx] = [self.edge_to_idx[n] for n in neighbors if n in self.edge_to_idx]

        junction_map = {} 
        for node in self.graph.nodes():
            if self.graph.degree(node) > 2:
                incident_edges = []
                for idx, edge in enumerate(self.edge_list):
                    if node in edge:
                        incident_edges.append(idx)
                junction_map[node] = incident_edges

        for t in range(self.timesteps - 1):
            # No Swapping
            processed_pairs = set()
            for u_idx, neighbors in adj_map.items():
                for v_idx in neighbors:
                    pair_key = tuple(sorted((u_idx, v_idx)))
                    if pair_key in processed_pairs: continue
                    processed_pairs.add(pair_key)
                    traffic_u_v = Or([And(self.states[(t, u_idx, ion)], self.states[(t+1, v_idx, ion)]) for ion in self.ions])
                    traffic_v_u = Or([And(self.states[(t, v_idx, ion)], self.states[(t+1, u_idx, ion)]) for ion in self.ions])
                    self.s.add(Not(And(traffic_u_v, traffic_v_u)))

            # Junction Capacity = 1
            for node, incident_indices in junction_map.items():
                junction_moves = []
                for e_start in incident_indices:
                    for e_end in incident_indices:
                        if e_start == e_end: continue
                        move_occurs = Or([And(self.states[(t, e_start, ion)], self.states[(t+1, e_end, ion)]) for ion in self.ions])
                        junction_moves.append(move_occurs)
                self.s.add(AtMost(*junction_moves, 1))

    def add_ordering_constraints(self):
        # LIFO Tunneling
        data_indices = [self.edge_to_idx[e] for e in self.data_edges if e in self.edge_to_idx]
        for t in range(1, self.timesteps - 1):
            for current_idx in data_indices:
                neighbors = get_possible_moves(self.graph, self.edge_list[current_idx])
                neighbor_indices = [self.edge_to_idx[n] for n in neighbors if n in self.edge_to_idx]
                
                for n_in, n_out in itertools.permutations(neighbor_indices, 2):
                    for ion_m in self.ions:
                        traversal = And(
                            self.states[(t-1, n_in, ion_m)],
                            self.states[(t, current_idx, ion_m)],
                            self.states[(t+1, n_out, ion_m)]
                        )
                        blocker_present = Or([self.states[(t, current_idx, ion_d)] for ion_d in self.ions if ion_d != ion_m])
                        self.s.add(Not(And(traversal, blocker_present)))

    def enforce_syndrome_schedule(self, schedules):
        start_edge_to_ion = {v: k for k, v in self.ions_dict.items()}
        duration = self.gate_duration

        for pz_edge, sequence in schedules.items():
            if pz_edge not in start_edge_to_ion: continue
            ion_id = start_edge_to_ion[pz_edge]
            
            # These variables track the *Start* of a gate
            gate_starts = [[Bool(f"start_{ion_id}_s{k}_t{t}") for t in range(self.timesteps)] 
                           for k in range(len(sequence))]

            for k, target in enumerate(sequence):
                target_idx = self.edge_to_idx[target]
                
                # Store for verification later
                for t in range(self.timesteps):
                    self.gate_start_vars[(ion_id, k, t)] = gate_starts[k][t]

                # 1. Gate must start exactly once
                valid_range = self.timesteps - duration + 1
                self.s.add(AtMost(*gate_starts[k], 1))
                self.s.add(AtLeast(*gate_starts[k][:valid_range], 1))
                for t in range(valid_range, self.timesteps):
                    self.s.add(Not(gate_starts[k][t]))

                # 2. Dwell: If Start(t) -> Occupy [t, t+dur-1]
                for t in range(valid_range):
                    dwell_conditions = []
                    for d in range(duration):
                        dwell_conditions.append(self.states[(t + d, target_idx, ion_id)])
                    self.s.add(Implies(gate_starts[k][t], And(*dwell_conditions)))

            # 3. Sequence: Start(k+1) at t2 => Start(k) at t1 where t1 + dur <= t2
            for k in range(len(sequence) - 1):
                for t_curr in range(self.timesteps):
                    possible_prev_starts = []
                    for t_prev in range(t_curr):
                        if t_prev + duration <= t_curr:
                            possible_prev_starts.append(gate_starts[k][t_prev])
                    
                    if not possible_prev_starts:
                        self.s.add(Not(gate_starts[k+1][t_curr]))
                    else:
                        self.s.add(Implies(gate_starts[k+1][t_curr], Or(*possible_prev_starts)))

    def add_no_traffic_constraints(self, schedules):
        """
        NEW: Forbids a measure ion from entering a Data Edge 
        UNLESS it is actively starting or performing a scheduled gate there.
        Prevents 'drive-by' traversals that mess up the schedule order.
        """
        start_edge_to_ion = {v: k for k, v in self.ions_dict.items()}
        
        for pz_edge, sequence in schedules.items():
            if pz_edge not in start_edge_to_ion: continue
            ion_id = start_edge_to_ion[pz_edge]
            
            # Map target edge -> list of step indices that use it
            # (An ion might visit the same edge twice in different steps, though unlikely in surface code)
            edge_to_steps = {}
            for k, target in enumerate(sequence):
                t_idx = self.edge_to_idx[target]
                if t_idx not in edge_to_steps: edge_to_steps[t_idx] = []
                edge_to_steps[t_idx].append(k)
            
            # Iterate over ALL Data Edges (potential illegal zones)
            for d_edge_idx in [self.edge_to_idx[e] for e in self.data_edges]:
                
                # Is this edge in the schedule?
                if d_edge_idx in edge_to_steps:
                    # Allowed if it is part of a valid gate execution
                    allowed_steps = edge_to_steps[d_edge_idx]
                    
                    for t in range(self.timesteps):
                        # Construct "Is Active Gate Here?" condition
                        # Active if Start(t) OR Start(t-1) ... OR Start(t - duration + 1)
                        is_active = []
                        for k in allowed_steps:
                            # Check valid start times relative to current time t
                            for d in range(self.gate_duration):
                                start_time = t - d
                                if 0 <= start_time < self.timesteps:
                                    is_active.append(self.gate_start_vars[(ion_id, k, start_time)])
                        
                        # Constraint: If Present at t -> Must be Active Gate
                        self.s.add(Implies(self.states[(t, d_edge_idx, ion_id)], Or(*is_active)))
                
                else:
                    # Edge not in schedule at all? FORBIDDEN completely.
                    for t in range(self.timesteps):
                        self.s.add(Not(self.states[(t, d_edge_idx, ion_id)]))

    def add_hook_error_ordering_constraints(self, schedules):
        start_edge_to_ion = {v: k for k, v in self.ions_dict.items()}
        edge_visitors = {}
        for pz_edge, sequence in schedules.items():
            if pz_edge not in start_edge_to_ion: continue
            ion_id = start_edge_to_ion[pz_edge]
            for step_idx, target_edge in enumerate(sequence):
                e_key = tuple(sorted(target_edge))
                if e_key not in edge_visitors: edge_visitors[e_key] = []
                edge_visitors[e_key].append((ion_id, step_idx))

        pair_map = {}
        for e_key, visitors in edge_visitors.items():
            if len(visitors) == 2:
                ion_a, step_a = visitors[0]
                ion_b, step_b = visitors[1]
                if ion_a > ion_b: ion_a, ion_b = ion_b, ion_a; step_a, step_b = step_b, step_a
                
                pair_key = (ion_a, ion_b)
                if pair_key not in pair_map: pair_map[pair_key] = []
                pair_map[pair_key].append({'edge_idx': self.edge_to_idx[e_key]})

        for (ion_a, ion_b), shared_list in pair_map.items():
            if len(shared_list) < 2: continue 
            order_bools = []
            for item in shared_list:
                edge_idx = item['edge_idx']
                a_before_b = Bool(f"order_{ion_a}_{ion_b}_{edge_idx}")
                for t_a in range(self.timesteps):
                    for t_b in range(self.timesteps):
                        if t_a < t_b:
                            self.s.add(Implies(And(self.states[(t_a, edge_idx, ion_a)], self.states[(t_b, edge_idx, ion_b)]), a_before_b))
                        elif t_b < t_a:
                            self.s.add(Implies(And(self.states[(t_a, edge_idx, ion_a)], self.states[(t_b, edge_idx, ion_b)]), Not(a_before_b)))
                order_bools.append(a_before_b)
            
            first = order_bools[0]
            for other in order_bools[1:]:
                self.s.add(first == other)

    def solve(self):
        print(f"Checking SAT for T={self.timesteps}, Gate Duration={self.gate_duration}...")
        if self.s.check() == sat:
            return self.s.model()
        return None

def verify_schedule_execution(solver, model, schedules):
    print("\n=== GATE EXECUTION LOG ===")
    start_edge_to_ion = {v: k for k, v in solver.ions_dict.items()}
    
    for pz_start_edge, target_sequence in schedules.items():
        if pz_start_edge not in start_edge_to_ion: continue
        ion_id = start_edge_to_ion[pz_start_edge]
        print(f"\nMeasure Ion {ion_id}:")
        
        for step_idx, target_edge in enumerate(target_sequence):
            found_start_time = None
            
            # Check the logical START variable, not physical presence
            for t in range(solver.timesteps):
                if (ion_id, step_idx, t) in solver.gate_start_vars:
                    if bool(model.evaluate(solver.gate_start_vars[(ion_id, step_idx, t)])):
                        found_start_time = t
                        break
            
            if found_start_time is not None:
                # Also check physical presence to be sure
                target_idx = solver.edge_to_idx[target_edge]
                is_physically_there = bool(model.evaluate(solver.states[(found_start_time, target_idx, ion_id)]))
                status = "SUCCESS" if is_physically_there else "LOGIC ERR"
                print(f"  Step {step_idx+1}: Gate STARTED at {target_edge} at t={found_start_time} [{status}]")
            else:
                print(f"  Step {step_idx+1}: Gate at {target_edge} NEVER STARTED [FAILURE]")

if __name__ == "__main__":
    m, n = 5, 5
    print(f"--- Building Graph for {m}x{n} ---")
    nx_g, pz_edges, data_edges, ions_dict = build_user_graph(m, n)
    schedules = generate_hook_error_schedule(pz_edges, data_edges)
    
    DURATION = 2 # Gate takes 2 steps (results in 1 time step gate duration)

    for t in range(6, 20): 
        solver = SyndromeSAT(nx_g, ions_dict, t, data_edges, pz_edges, gate_duration=DURATION)
        
        solver.create_constraints()
        solver.add_physical_constraints() # Strict Junction Capacity = 1
        solver.add_ordering_constraints()
        solver.enforce_syndrome_schedule(schedules)
        solver.add_hook_error_ordering_constraints(schedules)
        solver.add_no_traffic_constraints(schedules) # <--- NEW: Forces clean pathing
        
        model = solver.solve()
        
        if model:
            print(f"Found Solution with {t} timesteps!")
            verify_schedule_execution(solver, model, schedules)
            out_file = f"benchmarks/sat_solution_{m}x{n}_t{t}_d{DURATION}.json"
            generate_visualization_json(solver, model, out_file, m, n)
            break
        else:
            print(f"No Solution for {t} timesteps.\n")