# make_timeline.py
# Create an HTML-player-friendly timeline JSON by piggybacking on the shuttler's plotting calls.

from __future__ import annotations
from pathlib import Path
import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any

# --- project imports (new layout) ---
from mqt.ionshuttler.multi_shuttler.outside.graph_creator import (
    GraphCreator,
    PZCreator,
)
from mqt.ionshuttler.multi_shuttler.outside.processing_zone import ProcessingZone
from mqt.ionshuttler.multi_shuttler.outside.cycles import (
    create_starting_config,
    get_state_idxs,
)
from mqt.ionshuttler.multi_shuttler.outside.scheduling import get_ions
from mqt.ionshuttler.multi_shuttler.outside.compilation import (
    create_initial_sequence,
    create_dag,
    create_updated_sequence_destructive,
    get_front_layer_non_destructive,
    map_front_gates_to_pzs,
    create_dist_dict,
    update_distance_map,
)
from mqt.ionshuttler.multi_shuttler.outside.partition import get_partition
from mqt.ionshuttler.multi_shuttler.outside.graph_utils import (
    create_idc_dictionary,
    get_idx_from_idc,
)
from mqt.ionshuttler.multi_shuttler.outside.shuttle import main as run_shuttle_main
from mqt.ionshuttler.multi_shuttler.outside import plotting as plotting_mod
from mqt.ionshuttler.multi_shuttler.outside import shuttle as shuttle_mod
print("USING SHUTTLE:", shuttle_mod.__file__)

# ===========================
# Config (mirrors run_benchmarks defaults)
# ===========================
PLOT_IMAGES = True       # we only capture frames; no PDFs
SAVE_IMAGES = False
USE_DAG = False
PARTITIONING = True
ALGORITHM = "random_connecting"#"full_register_access"  # or "transpiled_surface" "qft_no_swaps_nativegates_quantinuum_tket"
OUTPUT_JSON = "timeline.json"

GATE_TIME_1Q = 1
GATE_TIME_2Q = 1

# Paths/Cycles mode string for new main signature
CYCLE_OR_PATHS_STR = "Cycles"  # set to "Cycles"/"Paths" to switch behavior


# ===========================
# Helpers for conversion
# ===========================
def fmt_int_tuple(p: Tuple[float, float]) -> str:
    r, c = p
    return f"({int(round(r))}, {int(round(c))})"


def infer_pz_side(pz: ProcessingZone, maxR: int, maxC: int) -> str:
    r, c = pz.processing_zone
    if r < 0:
        return "top"
    if r > maxR:
        return "bottom"
    if c < 0:
        return "left"
    if c > maxC:
        return "right"
    # fallback: nearest border
    dists = {
        "top": abs(r - 0),
        "bottom": abs(r - maxR),
        "left": abs(c - 0),
        "right": abs(c - maxC),
    }
    return min(dists, key=dists.get)


def build_pz_index(G, idc_dict) -> Dict[str, Dict[str, Any]]:
    """
    Build an index for mapping edge indices to PZ-related specs.

    Tries (in order):
      1) Use explicit attributes if they exist on the ProcessingZone:
         - path_to_pz_idxs / path_from_pz_idxs
         - connection_into_pz_idxs / connection_out_of_pz_idxs
      2) Otherwise, split pz.pz_edges_idx around the parking edge index:
         everything BEFORE parking -> "into", everything AFTER -> "out".
    """
    pzi: Dict[str, Dict[str, Any]] = {}
    maxR = (G.m * G.v) - G.v
    maxC = (G.n * G.h) - G.h

    for pz in G.pzs:
        side = infer_pz_side(pz, maxR, maxC)
        parking_idx = get_idx_from_idc(idc_dict, pz.parking_edge)

        # Prefer explicit attributes if available (back-compat with older code)
        if hasattr(pz, "path_to_pz_idxs") and hasattr(pz, "path_from_pz_idxs"):
            into_idxs = list(pz.path_to_pz_idxs)
            out_idxs = list(pz.path_from_pz_idxs)
        elif hasattr(pz, "connection_into_pz_idxs") and hasattr(pz, "connection_out_of_pz_idxs"):
            into_idxs = list(pz.connection_into_pz_idxs)
            out_idxs = list(pz.connection_out_of_pz_idxs)
        else:
            # Fallback: split pz.pz_edges_idx around parking edge
            all_idxs = list(getattr(pz, "pz_edges_idx", []))
            if all_idxs and parking_idx in all_idxs:
                k = all_idxs.index(parking_idx)
                into_idxs = all_idxs[:k]      # heading INTO the PZ core
                out_idxs = all_idxs[k + 1:]   # heading OUT into the MZ
            else:
                # Last resort: at least show "out" edges so exits appear
                into_idxs = []
                out_idxs = all_idxs

        pzi[side] = {
            "name": pz.name,
            "connection_into_pz_idxs": into_idxs,
            "connection_out_of_pz_idxs": out_idxs,
            "parking_idx": parking_idx,
            "pz_obj": pz,
        }

    return pzi


def edge_idx_to_spec(idx: int, G, pz_index):
    """
    Convert a global edge index back to a user-facing spec used in the timeline JSON.

    For PZ-associated edges we emit string tags:
      - "PZ(<side>):parking"
      - "PZ(<side>):connection_into_pz:<i>"
      - "PZ(<side>):connection_out_of_pz:<i>"

    For MZ grid edges we emit the tuple endpoints:
      - ["(r1, c1)", "(r2, c2)"]
    """
    for side, info in pz_index.items():
        if idx == info["parking_idx"]:
            return (f"PZ({side}):parking", None)

        # INTO the PZ
        if idx in info["connection_into_pz_idxs"]:
            i = info["connection_into_pz_idxs"].index(idx)
            return (f"PZ({side}):connection_into_pz:{i}", None)

        # OUT OF the PZ into the MZ
        if idx in info["connection_out_of_pz_idxs"]:
            i = info["connection_out_of_pz_idxs"].index(idx)
            return (f"PZ({side}):connection_out_of_pz:{i}", None)

    # Otherwise, map the grid edge index back to its endpoints
    if not hasattr(G, "_rev_idc"):
        G._rev_idc = {val: key for key, val in G.idc_dict.items()}
    try:
        u, v = G._rev_idc[idx]
    except KeyError:
        return (f"PZ(top):parking", None)  # harmless fallback
    return (fmt_int_tuple(u), fmt_int_tuple(v))


def make_q_ids(ions: List[str]) -> Dict[str, str]:
    ids = sorted(ions)
    return {ion: f"$q_{i}$" for i, ion in enumerate(ids)}


# ===========================
# Frame capture by monkey-patching plotting.plot_state
# ===========================
class FrameCollector:
    def __init__(self):
        self.frames: List[Dict[str, Any]] = []
        self.q_map: Dict[str, str] | None = None
        self.pz_index: Dict[str, Dict[str, Any]] | None = None
        self.name_to_side: Dict[str, str] = {}
    
    def attach_to_graph(self, G):
        self.pz_index = build_pz_index(G, G.idc_dict)
        # Build reverse map from PZ name -> side
        try:
            self.name_to_side = {info.get("name"): side for side, info in (self.pz_index or {}).items() if isinstance(info, dict) and info.get("name")}
        except Exception:
            self.name_to_side = {}

    def capture(self, graph, *args, **kwargs):
        if self.pz_index is None:
            self.attach_to_graph(graph)

        state = getattr(graph, "state", None)
        if state is None:
            try:
                graph.state = get_ions(graph)
                state = graph.state
            except Exception:
                state = {}

        if self.q_map is None and state:
            self.q_map = make_q_ids(list(state.keys()))

        # NEW: attach executed gates from previous timestep (buffered on graph)
        pending = getattr(graph, "executed_gates_next", None)
        if pending and self.frames:
            def _fmt_node(p):
                r, c = p
                return f"({int(round(r))}, {int(round(c))})"
            t_end = len(self.frames) - 1
            for i, g in enumerate(pending):
                try:
                    gid = str(g.get("id", f"g_{i}"))
                    gtype = str(g.get("type", "OP")).upper()
                    qubits_raw = g.get("qubits", [])
                    qubits = qubits_raw
                    if self.q_map:
                        qubits = [q if str(q).startswith("$q_") else self.q_map.get(q, f"$q_{q}$") for q in qubits_raw]
                    edge = None
                    if "edge" in g and g["edge"]:
                        edge = g["edge"]
                    elif "edge_idc" in g and g["edge_idc"]:
                        try:
                            a, b = g["edge_idc"]
                            edge = [_fmt_node(a), _fmt_node(b)]
                        except Exception:
                            edge = None
                    try:
                        duration = int(g.get("duration")) if g.get("duration") is not None else (3 if len(qubits_raw) >= 2 else 1)
                    except Exception:
                        duration = 3 if len(qubits_raw) >= 2 else 1
                    t0 = max(0, t_end - (duration - 1))
                    # Resolve pz side from provided pz name or side
                    side = None
                    pz_in = g.get("pz")
                    if isinstance(pz_in, str):
                        if pz_in.lower() in ("top","right","bottom","left"):
                            side = pz_in.lower()
                        else:
                            side = self.name_to_side.get(pz_in)
                    for j in range(max(1, duration)):
                        idx = t_end - j
                        if idx < 0:
                            break
                        fr = self.frames[idx]
                        if "gates" not in fr:
                            fr["gates"] = []
                        fr["gates"].append({
                            "id": gid,
                            "type": gtype,
                            "qubits": qubits,
                            **({"edge": edge} if edge else {}),
                            "duration": duration,
                            "t0": t0,
                            **({"pz": side} if side else {}),
                        })
                except Exception:
                    continue
            try:
                graph.executed_gates_next = []
            except Exception:
                pass

        # time index from graph (shuttler increments this)
        t = getattr(graph, "timesteps", len(self.frames))

        ions_out: List[Dict[str, Any]] = []
        for ion, edge_idc in state.items():
            try:
                idx = get_idx_from_idc(graph.idc_dict, edge_idc)
                a, b = edge_idx_to_spec(idx, graph, self.pz_index)
                edge_spec = a if b is None else [a, b]
                ions_out.append(
                    {
                        "id": self.q_map.get(
                            ion, ion if str(ion).startswith("$q_") else f"${ion}$"
                        ),
                        "edge": edge_spec,
                    }
                )
            except Exception:
                continue

        self.frames.append({"t": t, "ions": ions_out})

        # Optionally call the real plotter (if user wants figures)
        show_plot = kwargs.get("show_plot", False)
        save_plot = kwargs.get("save_plot", False)
        if show_plot or save_plot:
            return _REAL_PLOT_STATE(graph, *args, **kwargs)
        # Otherwise, swallow the call.


# keep a handle to the original, then patch
_REAL_PLOT_STATE = plotting_mod.plot_state
_COLLECTOR = FrameCollector()


def _patched_plot_state(graph, *args, **kwargs):
    return _COLLECTOR.capture(graph, *args, **kwargs)


# ===========================
# Main runner (one configuration -> one timeline)
# ===========================
def run_single(
    m: int, n: int, v: int, h: int, number_of_pz: int, seed: int = 0
) -> Dict[str, Any]:
    height = -4.5

    # Renamed for clarity; order preserved for ProcessingZone constructor compatibility.
    into_anchor_1 = (float((m - 1) * v), float((n - 1) * h))  # was exit1
    out_anchor_1 = (float((m - 1) * v), float(0))  # was entry1
    processing_zone1 = (
        float((m - 1) * v - height),
        float((n - 1) * h / 2),
    )

    into_anchor_2 = (0.0, float(0))  # was exit2
    out_anchor_2 = (0.0, float((n - 1) * h))  # was entry2
    processing_zone2 = (float(height), float((n - 1) * h / 2))

    into_anchor_3 = (float((m - 1) * v), float(0))  # was exit3
    out_anchor_3 = (float(0), float(0))  # was entry3
    processing_zone3 = (float((m - 1) * v / 2), float(height))

    into_anchor_4 = (float(0), float((n - 1) * h))  # was exit4
    out_anchor_4 = (float((m - 1) * v), float((n - 1) * h))  # was entry4
    processing_zone4 = (
        float((m - 1) * v / 2),
        float((n - 1) * h - height),
    )

    # Preserve original positional order: [into, out, pz_core]
    pz1 = ProcessingZone("pz1", [into_anchor_1, out_anchor_1, processing_zone1])
    pz2 = ProcessingZone("pz2", [into_anchor_2, out_anchor_2, processing_zone2])
    pz3 = ProcessingZone("pz3", [into_anchor_3, out_anchor_3, processing_zone3])
    pz4 = ProcessingZone("pz4", [into_anchor_4, out_anchor_4, processing_zone4])
    pzs = [pz1, pz2, pz3, pz4][0:number_of_pz]

    # graphs
    basegraph_creator = GraphCreator(m, n, v, h, failing_junctions=0, pz_info=pzs)
    MZ_graph = basegraph_creator.get_graph()

    pzgraph_creator = PZCreator(m, n, v, h, failing_junctions=0, pzs=pzs)
    G = pzgraph_creator.get_graph()
    G.mz_graph = MZ_graph

    # handy attrs for side inference
    G.m, G.n, G.v, G.h = m, n, v, h

    # init as in benchmarks
    G.seed = seed
    _ = G.idc_dict  # ensure lazy build; keep locally via G.idc_dict when needed
    G.max_num_parking = 2  # set before assigning pzs
    G.pzs = pzs

    # Override gate times in outside shuttle module
    try:
        from mqt.ionshuttler.multi_shuttler.outside import shuttle as sh_out
        sh_out.GATE_TIME_1Q = GATE_TIME_1Q
        sh_out.GATE_TIME_2Q = GATE_TIME_2Q
    except Exception:
        pass

    G.plot = PLOT_IMAGES
    G.save = SAVE_IMAGES
    G.arch = str([m, n, v, h])

    number_of_mz_edges = len(MZ_graph.edges())
    print(f"Number of MZ edges: {number_of_mz_edges}")
    number_of_chains = 22#math.ceil(1.0 * len(MZ_graph.edges()))#50

    # starting ions and state
    create_starting_config(G, number_of_chains, seed=seed)
    G.state = get_ions(G)

    # sequence + partitioning/DAG
    qasm_file_path = Path(
        f"QASM_files/development/{ALGORITHM}/{ALGORITHM}_{number_of_chains}.qasm"
    )
    assert qasm_file_path.is_file(), f"QASM file not found: {qasm_file_path}"
    G.sequence = create_initial_sequence(qasm_file_path)

    if PARTITIONING:
        part = get_partition(str(qasm_file_path), len(G.pzs))
        partition = {pz.name: part[i] for i, pz in enumerate(G.pzs)}
    else:
        raise NotImplementedError("Non-partitioning branch not implemented here.")

    G.map_to_pz = {
        element: pz for pz, elements in partition.items() for element in elements
    }

    dag = None
    if USE_DAG:
        for pz in G.pzs:
            pz.getting_processed = []
        dag = create_dag(qasm_file_path)
        G.locked_gates = {}
        front_layer_nodes = get_front_layer_non_destructive(
            dag, virtually_processed_nodes=[]
        )
        pz_info_map = map_front_gates_to_pzs(G, front_layer_nodes=front_layer_nodes)
        _ = {value: key for key, values in pz_info_map.items() for value in values}
        G.dist_dict = create_dist_dict(G)
        state_idxs = get_state_idxs(G)
        G.dist_map = update_distance_map(G, state_idxs)
        sequence, _, dag = create_updated_sequence_destructive(
            G, qasm_file_path, dag, use_dag=USE_DAG
        )
        G.sequence = sequence

    # ---- Patch plotting: definition AND the symbol shuttle calls ----
    plotting_mod.plot_state = _patched_plot_state
    shuttle_mod.plot_state = _patched_plot_state
    _COLLECTOR.attach_to_graph(G)

    # run the shuttler (new signature: graph, dag, cycle_or_paths_str, use_dag=...)
    try:
        _ = run_shuttle_main(G, dag, CYCLE_OR_PATHS_STR, use_dag=USE_DAG)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial timeline...")
    
    # sides present (booleans for external PZs)
    maxR = (m - 1) * v
    maxC = (n - 1) * h
    sides_present = {}
    for pz in G.pzs:
        side = infer_pz_side(pz, maxR, maxC)
        sides_present[side] = True
    for s in ["top", "right", "bottom", "left"]:
        sides_present.setdefault(s, False)

    out = {
        "grid": {"rows": m, "cols": n},
        "sites": {"vertical": v, "horizontal": h},
        "pzs": sides_present,  # booleans indicating which external PZs exist
        "meta": {
            "algorithm": ALGORITHM,
            "seed": seed,
            "arch": [m, n, v, h],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "timeline": _COLLECTOR.frames,
    }
    return out


def main():
    # Example run
    m, n, v, h = 5, 4, 1, 1
    number_of_pz = 2
    seed = 0

    result = run_single(m, n, v, h, number_of_pz, seed)

    # ---------- JSON output ----------
    # Build viewer-friendly payload:
    # - Include "architecture" with grid/sites/pzs and (for outside shuttler) empty innerPZEdges.
    # - Duplicate the same fields at top level for compatibility.
    # - Keep the collected timeline as-is.
    architecture = {
        "grid": result["grid"],
        "sites": result["sites"],
        "pzs": result["pzs"],          # booleans for external PZs (top/right/bottom/left)
        "innerPZEdges": []             # outside shuttler: no inner PZ-marked segments
    }
    payload = {
        "architecture": architecture,
        "grid": architecture["grid"],
        "sites": architecture["sites"],
        "pzs": architecture["pzs"],
        "innerPZEdges": architecture["innerPZEdges"],
        "timeline": result["timeline"]
    }

    # Minified JSON (no extra spaces)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)

    print(f"Wrote {OUTPUT_JSON} with {len(result['timeline'])} frames.")


if __name__ == "__main__":
    main()
