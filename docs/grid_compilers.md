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

Use `target_placement` to request a final placement after all gates. The mapping
uses the same format as `initial_placement`. A target that contains every ion is
exact, and omitted segments are empty. A target that contains a subset
constrains only those ions; all other ions and segments remain unconstrained.

```{code-cell} ipython3
result = GridCompiler(architecture).compile(
    circuit,
    initial_placement={"memory": (0, 1)},
    target_placement={"memory": (0, 1)},
)
```

```{code-cell} ipython3
result = GridCompiler(architecture).compile(
    circuit,
    initial_placement={"memory": (0, 1)},
    target_placement={"processor": (0,)},
)
```

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
