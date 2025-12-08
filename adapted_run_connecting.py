# run_benchmarks.py
from __future__ import annotations
import pathlib
import time
from datetime import datetime
from typing import Any

# Adjust this import if your runner module has a different name/path
from mqt.ionshuttler.multi_shuttler.main import main as run_main

# ---- Experiment grid ----
ARCHS: list[list[int]] = [
     [3, 3, 1, 1],
    # [3, 3, 3, 3],
    # [3, 3, 5, 5],
    # [4, 4, 1, 1],
    #[4, 4, 3, 3],
    # [4, 4, 5, 5],
    # [5, 5, 1, 1],
]
SEEDS = [0]
NUM_PZS_LIST = [2]  # add 2,3,4 if you want to sweep
USE_DAG = False
USE_PATHS = False  # True -> "Paths", False -> "Cycles"
FAILING_JUNCTIONS = 0
PLOT = False
SAVE = False
# Force interactive backend when plotting
if PLOT:
    import matplotlib
    # On macOS
    matplotlib.use("MacOSX", force=True)

# Algorithms live under: QASM_files/development/<algorithm>/<algorithm>_<num_ions>.qasm
ALGORITHM = "ghz_nativegates_quantinuum_tket"#"qv_test_transpiled"#"random_connecting"#"qft_no_swaps_nativegates_quantinuum_tket"ghz_nativegates_quantinuum_tket

# Base dir for the “development” QASM set
QASM_BASE_DIR = pathlib.Path.cwd() / "QASM_files" / "development"

# Where to store a tiny txt summary (optional)
BENCH_DIR = pathlib.Path("benchmarks")
BENCH_DIR.mkdir(exist_ok=True)
STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BENCH_FILE = BENCH_DIR / f"exit_{STAMP}_{ALGORITHM}.txt"

# Allow overriding gate times for 1q/2q gates in outside.shuttle
GATE_TIME_1Q = 1
GATE_TIME_2Q = 3

# ---- Move counter monkey-patch ----
class _MoveCounter:
    def __init__(self) -> None:
        self.prev: dict[int, tuple[tuple[int, int], tuple[int, int]]] | None = None
        self.moves: int = 0

    def _snapshot_map(self, graph) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
        curr: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        try:
            for e, data in graph.edges.items():
                for ion in data.get("ions", []) or []:
                    curr[ion] = e
        except Exception:
            for e in graph.edges():
                data = graph.get_edge_data(*e) or {}
                for ion in data.get("ions", []) or []:
                    curr[ion] = e
        return curr

    def count_from_graph(self, graph) -> None:
        curr = self._snapshot_map(graph)
        if self.prev is not None:
            for ion, edge in curr.items():
                prev_edge = self.prev.get(ion)
                if prev_edge is not None and prev_edge != edge:
                    self.moves += 1
        self.prev = curr

    def count_delta(self, prev_map, curr_map) -> None:
        for ion, edge in curr_map.items():
            prev_edge = prev_map.get(ion)
            if prev_edge is not None and prev_edge != edge:
                self.moves += 1


def _patch_plotting(counter: _MoveCounter):
    """Patch inside/outside plotting.plot_state to count moves per frame."""
    patched = []
    try:
        from mqt.ionshuttler.multi_shuttler.outside import plotting as plotting_out
        _orig_out = plotting_out.plot_state

        def _wrap_out(graph, labels, plot_ions=True, show_plot=False, save_plot=False, plot_cycle=False, plot_pzs=False, filename=None):
            counter.count_from_graph(graph)
            return _orig_out(graph, labels, plot_ions, show_plot, save_plot, plot_cycle, plot_pzs, filename)

        plotting_out.plot_state = _wrap_out  # type: ignore
        patched.append((plotting_out, _orig_out))
    except Exception:
        pass
    try:
        from mqt.ionshuttler.multi_shuttler.inside import plotting as plotting_in
        _orig_in = plotting_in.plot_state

        def _wrap_in(graph, labels, plot_ions=True, show_plot=False, save_plot=False, plot_cycle=False, plot_pzs=False, filename=None):
            counter.count_from_graph(graph)
            return _orig_in(graph, labels, plot_ions, show_plot, save_plot, plot_cycle, plot_pzs, filename)

        plotting_in.plot_state = _wrap_in  # type: ignore
        patched.append((plotting_in, _orig_in))
    except Exception:
        pass
    return patched


def _unpatch_plotting(patched):
    for mod, orig in patched:
        try:
            mod.plot_state = orig  # type: ignore
        except Exception:
            pass


def _patch_shuttle(counter: _MoveCounter):
    """Patch inside/outside shuttle.shuttle to count moves per timestep."""
    patched = []
    # Outside
    try:
        from mqt.ionshuttler.multi_shuttler.outside import shuttle as sh_out
        _orig_sh_out = sh_out.shuttle

        def _wrap_sh_out(graph, priority_queue, timestep, cycle_or_paths, unique_folder):
            prev_map = counter._snapshot_map(graph)
            res = _orig_sh_out(graph, priority_queue, timestep, cycle_or_paths, unique_folder)
            curr_map = counter._snapshot_map(graph)
            counter.count_delta(prev_map, curr_map)
            return res

        sh_out.shuttle = _wrap_sh_out  # type: ignore
        patched.append((sh_out, _orig_sh_out, 'shuttle'))
    except Exception:
        pass
    # Inside
    try:
        from mqt.ionshuttler.multi_shuttler.inside import shuttle as sh_in
        _orig_sh_in = sh_in.shuttle

        def _wrap_sh_in(graph, priority_queue, timestep, cycle_or_paths, unique_folder):
            prev_map = counter._snapshot_map(graph)
            res = _orig_sh_in(graph, priority_queue, timestep, cycle_or_paths, unique_folder)
            curr_map = counter._snapshot_map(graph)
            counter.count_delta(prev_map, curr_map)
            return res

        sh_in.shuttle = _wrap_sh_in  # type: ignore
        patched.append((sh_in, _orig_sh_in, 'shuttle'))
    except Exception:
        pass
    return patched


def _unpatch_shuttle(patched):
    for mod, orig, attr in patched:
        try:
            setattr(mod, attr, orig)
        except Exception:
            pass


def run_one(config: dict[str, Any]) -> float:
    """Run one config and return wall-clock seconds."""
    # Install move counter patches
    counter = _MoveCounter()
    patched_plot = _patch_plotting(counter)
    patched_shuttle = _patch_shuttle(counter)

    # Override gate times in outside shuttle module
    try:
        from mqt.ionshuttler.multi_shuttler.outside import shuttle as sh_out
        sh_out.GATE_TIME_1Q = GATE_TIME_1Q
        sh_out.GATE_TIME_2Q = GATE_TIME_2Q
    except Exception:
        pass

    t0 = time.perf_counter()
    try:
        timesteps = run_main(config)   # prints progress + results internally
    finally:
        t1 = time.perf_counter()
        _unpatch_plotting(patched_plot)
        _unpatch_shuttle(patched_shuttle)
        print(f"Total ion moves counted: {counter.moves}")
    return t1 - t0, counter.moves, timesteps


def main() -> None:
    for m, n, v, h in ARCHS:
        for num_pzs in NUM_PZS_LIST:
            for seed in SEEDS:

                num_ions = int(((m-1)*v*n + (n-1)*h*m))
                print(f"max num_ions: {(m-1)*v*n + (n-1)*h*m}")

                cfg = {
                    "arch": [m, n, v, h],        # [m, n, v, h]
                    "num_pzs": num_pzs,          # how many PZs to enable
                    "seed": seed,                # RNG seed
                    "algorithm_name": ALGORITHM, # folder + filename prefix in qasm_base_dir
                    "num_ions": num_ions,        # selects .../<algorithm>_<num_ions>.qasm
                    "use_dag": USE_DAG,
                    "use_paths": USE_PATHS,
                    "plot": PLOT,
                    "save": SAVE,
                    "failing_junctions": FAILING_JUNCTIONS,
                    # Point at .../QASM_files/development
                    "qasm_base_dir": str(QASM_BASE_DIR),
                    # Optionally expose a max steps limit (your runner reads it but doesn’t use it yet)
                    "max_timesteps": 100_000,
                }

                print("\n" + "=" * 80)
                print(
                    f"Run: arch={cfg['arch']}  #pzs={num_pzs}  seed={seed}  "
                    f"algo={ALGORITHM}  ions={num_ions}  "
                    f"DAG={USE_DAG}  mode={'Paths' if USE_PATHS else 'Cycles'}"
                )
                print("=" * 80)

                wall, moves_count, timesteps = run_one(cfg)

                # Append a tiny row to a benchmarks file (runner already prints details)
                try:
                    with BENCH_FILE.open("a") as f:
                        f.write(
                            f"{datetime.now().isoformat()} "
                            f"arch={cfg['arch']} pzs={num_pzs} seed={seed} "
                            f"algo={ALGORITHM} ions={num_ions} "
                            f"dag={USE_DAG} paths={USE_PATHS} "
                            f"wall_s={wall:.2f} "
                            f"moves={moves_count} "
                            f"time steps={timesteps}\n"
                        )
                except Exception as e:
                    print(f"Warning: could not write {BENCH_FILE}: {e}")

    print(f"\nDone. Summary (wall time only) in: {BENCH_FILE}")
    print("ran shuttle scheduling file: ")
    import os
    import mqt.ionshuttler.multi_shuttler.outside.scheduling
    print(os.path.abspath(mqt.ionshuttler.multi_shuttler.outside.scheduling.__file__))

if __name__ == "__main__":
    main()

# for connecting: place ions in pz, gate times to 1