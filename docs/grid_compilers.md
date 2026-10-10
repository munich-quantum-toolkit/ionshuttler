---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Grid Compilers

Grid compilers route ions between processing and memory segments, connected by
junctions. Use the {doc}`Linear compiler <linear_compiler>` when compilation
must resolve individual sites and local transport within a segment. The
{doc}`hardware model overview <hardware_models>` compares both abstraction
levels.

## Grid compiler API

{py:class}`~mqt.ionshuttler.grid.compiler.GridCompiler` implements the shared
compiler contract. It accepts a supported circuit input and returns a replayable
{py:class}`~mqt.ionshuttler.core.result.CompilationResult`. Use
{py:func}`~mqt.ionshuttler.grid.layouts.square_grid`,
{py:func}`~mqt.ionshuttler.grid.layouts.rectangular_grid`,
{py:func}`~mqt.ionshuttler.grid.layouts.hexagonal_grid`, or
{py:func}`~mqt.ionshuttler.grid.layouts.from_networkx` to construct common
layouts.

```{code-cell} ipython3
from mqt.ionshuttler.grid import (
    GridArchitecture,
    GridCompiler,
    Junction,
    ProcessingZone,
    Segment,
)

memory = Segment("memory", capacity=2)
processing = Segment("processing", capacity=2, processing_zones=(ProcessingZone("pz"),))
architecture = GridArchitecture(
    segments=(memory, processing),
    junctions=(Junction("memory-processing", (memory.end, processing.start)),),
)
circuit = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rxx(0.5) q[0],q[1];
"""

result = GridCompiler(architecture).compile(
    circuit,
    initial_placement={"memory": (0, 1)},
)
result.validate()
```

The default strategy is a deterministic greedy scheduler. It looks ahead over
upcoming gates, routes several ions concurrently, and uses cycles to clear
blocked paths. Independent gates and transport actions can overlap when they use
different resources.

By default, the compiler also:

- uses commutation-aware circuit dependencies;
- partitions gates between suitable processing zones before scheduling;
- runs ten lookahead rounds, each selecting at most one gate per target
  processing-zone segment, and prioritizes the ions used by those gates.

Set `pre_partition=False` in
{py:meth}`~mqt.ionshuttler.grid.compiler.GridCompiler.compile` to select
processing zones while scheduling. Use `gate_to_processing_zone` to assign
individual gate IDs to processing-zone IDs explicitly. An explicit mapping
replaces pre-partitioning, even when `pre_partition=True`. Unassigned gates
select a processing zone while scheduling.

### Configuration

{py:class}`~mqt.ionshuttler.grid.config.GridCompilerConfig` provides the main
compiler settings:

- {py:attr}`~mqt.ionshuttler.grid.config.GridCompilerStrategy.GREEDY` is the
  default and suits larger inputs.
- {py:attr}`~mqt.ionshuttler.grid.config.GridCompilerStrategy.BREADTH_FIRST`
  performs bounded routing search and is intended for small architectures.
- {py:attr}`~mqt.ionshuttler.grid.config.GridDependencyMode.CONSERVATIVE`
  preserves the dependencies produced by the shared circuit parser instead of
  reordering commuting gates.
- {py:attr}`~mqt.ionshuttler.grid.config.GridCompilerConfig.allowed_junction_crossings`
  can restrict routing to selected directed segment pairs without changing the
  hardware topology.
- {py:attr}`~mqt.ionshuttler.grid.config.GridCompilerConfig.greedy_lookahead_rounds`
  controls how much future work the greedy strategy considers. Larger values can
  increase both congestion and compilation time.

```{code-cell} ipython3
from mqt.ionshuttler.grid import GridCompilerConfig, GridCompilerStrategy

config = GridCompilerConfig(
    strategy=GridCompilerStrategy.BREADTH_FIRST,
    max_routing_states=1_000,
)
result = GridCompiler(architecture, config).compile(circuit)
```

### Model limits

- The Grid model has no segment-reordering action or reordering cost.
  `UNORDERED` segments leave their internal ion order unspecified, so the Grid
  model permits reordering without a separate action or duration.
- The compiler uses bounded heuristics and does not guarantee a minimum
  makespan. A failed compilation does not prove that the input is infeasible.
- Grid timing does not include any transport or reordering that a future
  refinement to a Linear architecture might require.

### Planning trace

Set `record_planning_trace=True` to inspect failed or slow compilations. The
compiler stores one snapshot per scheduling decision in
`result.diagnostics.planning_trace`. Each snapshot summarizes the selected
gates, target segments, ion priorities, transport actions, and any reason for
making no progress.

```{code-cell} ipython3
config = GridCompilerConfig(record_planning_trace=True)
result = GridCompiler(architecture, config).compile(circuit)
for snapshot in result.diagnostics.planning_trace or ():
    print(snapshot.time, snapshot.selected_gates, snapshot.idle_reason)
```

Tracing is disabled by default because long compilations can produce many
snapshots. The schedule remains the authoritative record of executed actions.

## Visualize a result

{py:func}`mqt.ionshuttler.visualize` returns an interactive
{py:class}`~mqt.ionshuttler.visualization.GridView` for a Grid result. In a
notebook, the view shows itself as the cell output. Elsewhere, save it as an
HTML file and open it in a browser. The file contains all data and needs no
network access.

```python
from mqt.ionshuttler import visualize

view = visualize(result)
view.save("schedule.html")
```

The view draws the schedule on a canvas. Use **Play**, **Previous layer**,
**Next layer**, and the time slider to move through the schedule. Ions that
cross a junction travel from their segment through the junction into the
destination segment. Simultaneous actions move together. Running gates highlight
their ions and processing zone. The header shows the time and the actions of the
running layer.

### Result viewer

The result viewer is one browser page for all results. It runs in your Python
process and listens only on this computer. Python replays each result; the page
draws it.

```python
from mqt.ionshuttler.visualization import GridVisualizer, open_viewer

visualizer = GridVisualizer(theme="dark")
visualizer.open(result)  # show one result
visualizer.compare({"first": first, "second": second}).open()  # show a comparison
open_viewer()  # open the viewer without a result
```

Use **Open result…** in the page, or drop a file on it, to show a result saved
with `result.save("result.json")`. The **View** menu switches between all
results of the session. In a notebook, `open` returns at once and the viewer
runs as long as the kernel. In a plain script, `open` waits until you press
Ctrl+C, so that the viewer keeps running. Under WSL, the viewer opens in the
Windows browser. The returned address also opens the viewer by hand. The viewer
shows Grid results; Linear results are not supported yet.

### Settings

{py:class}`~mqt.ionshuttler.visualization.GridVisualizer` holds the display
settings. The browser controls start from these values.

```python
from mqt.ionshuttler.visualization import GridVisualizer

visualizer = GridVisualizer(
    theme="dark",
    show_ion_labels=True,
    show_processing_zone_labels=False,
    show_hardware_ids=True,
    timesteps_per_second=8,
)
view = visualizer.visualize(result)
```

`theme` is `"dark"` (the default), `"light"`, or `"auto"`, which follows the
browser. Figures and videos use the same theme; pass `theme="light"` for print.
`show_hardware_ids` writes segment and junction IDs. `timesteps_per_second` sets
the playback speed. `width` and `height` set the view and video size in pixels.

The rectangular and square generators use junction IDs of the form
`j:row:column`. The visualizer uses those IDs to recover the generated lattice.
For other architectures, including hexagonal grids, it first attempts a planar
layout. A non-planar graph uses a deterministic force-directed layout instead.
To control the drawing, pass explicit coordinates. The mapping must contain each
junction exactly once. Keys can be junction IDs or `Junction` values.

```python
visualizer = GridVisualizer(junction_coordinates={"memory-processor": (0.0, 0.0)})
```

Coordinates belong to the visualization, not `GridArchitecture`. Changing a
drawing therefore does not change compilation, replay, or serialized hardware.

To arrange a drawing by hand, click **Edit layout** in the view and drag
junctions. Segments and ions follow at once, also during playback.
**Copy coordinates** copies the current positions as Python source for
`junction_coordinates`. For a comparison, it copies one mapping per panel for
the `junction_coordinates` argument of `compare`. **Reset** restores the
original layout.

### Ion and processing-zone colors

Each ion is drawn as a disk with a colored ring. By default, every ion has its
own ring color. Each processing zone also has its own color. A running gate
draws a thin outer ring around its ions in that color and thickens its zone.
`ion_colors="single"` gives all ions one ring color. A mapping sets the colors
of chosen ions, for example to show the role of each ion. A color can change at
a schedule time, and a label adds a legend entry:

```python
from mqt.ionshuttler.visualization import GridVisualizer, IonColor

data = IonColor(border="tab:blue", label="data")
idle = IonColor(border="#94a3b8", label="ancilla")
active = IonColor(border="crimson", fill="#fee2e2", label="active ancilla")
visualizer = GridVisualizer(
    ion_colors={0: data, 1: data, 2: [(0, idle), (120, active)]},
    processing_zone_colors={"pz": "teal"},
)
```

Colors accept any Matplotlib color. Ions without an entry have a gray ring. The
**Ion colors** switch in the view changes between distinct, single, and custom
colors. In **Edit layout** mode, click an ion or a processing zone to change its
color. **Copy settings** then also copies the changed colors.

### Video export

Open **Export video…** in the view and drag the two handles to select the first
and last timestep. The export uses the current theme, labels, speed, and frame
rate. The video shows `timesteps_per_second` timesteps per second, so the frame
rate changes only smoothness. The browser draws each frame at a fixed schedule
time and does not wait for playback. It writes a WebM file and needs a browser
with WebCodecs video encoding, such as a current Chromium-based browser.

Scripts can export a video without a browser. This uses Matplotlib with the same
layout, colors, and labels:

```python
view = GridVisualizer(theme="dark", video_start_time=315, video_end_time=420).visualize(result)
view.export_video("interesting.gif", frames_per_second=30)
view.export_video("interesting.mp4", start_time=315, end_time=360)
```

A `.gif` file needs no further software. The `.mp4`, `.m4v`, `.mov`, `.mkv`, and
`.webm` formats need the FFmpeg program.

### Static figures and comparisons

{py:meth}`~mqt.ionshuttler.visualization.GridVisualizer.plot` draws one schedule
time as a Matplotlib figure, for example for a paper:

```python
visualizer.plot(result, 120).savefig("t120.png", dpi=200)
```

{py:meth}`~mqt.ionshuttler.visualization.GridVisualizer.compare` plays several
results side by side with one clock. The clock uses absolute schedule time. A
result that ends earlier keeps showing its final state.

```python
view = visualizer.compare({"breadth-first": first, "greedy": second})
```

Results on different architectures can use their own explicit coordinates:

```python
view = visualizer.compare(
    {"loop": loop_result, "square": square_result},
    junction_coordinates={"loop": loop_coordinates},
)
```

## Established command-line compilers

The established exact and heuristic compilers remain available during the
migration to the shared API.

### Exact compilation

The exact compiler searches for a minimum-cost solution on small architectures
with one processing zone.

```console
mqt-ionshuttler-exact --help
mqt-ionshuttler-exact inputs/algorithms_exact/qft_06.json
```

Example inputs are available in
[`inputs/algorithms_exact`](https://github.com/munich-quantum-toolkit/ionshuttler/tree/main/inputs/algorithms_exact).

### Heuristic compilation

The heuristic compiler supports larger circuits and multiple processing zones.

```console
mqt-ionshuttler-heuristic --help
mqt-ionshuttler-heuristic inputs/algorithms_heuristic/qft_60_4pzs.json
```

Example inputs are available in
[`inputs/algorithms_heuristic`](https://github.com/munich-quantum-toolkit/ionshuttler/tree/main/inputs/algorithms_heuristic).

These tools retain their existing command-line and JSON interfaces. They are
separate from the common `GridCompiler` API.

## See also

- {doc}`hardware_models` — compare the Linear and Grid abstractions
- {doc}`references` — publications describing the exact and heuristic methods
