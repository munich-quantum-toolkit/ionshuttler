# Grid Compilers

The grid compilers route ions through graph-based memory and processing zones.
They provide the original MQT IonShuttler workflows for exact compilation of
small instances and heuristic compilation of larger devices.

Use the {doc}`Linear compiler <linear_compiler>` when the compilation task must
resolve individual sites and local transport within a trap segment. The
{doc}`hardware model overview <hardware_models>` compares the two abstraction
levels.

## Exact compilation

The exact compiler targets small architectures with one processing zone. It
searches for a minimum-cost shuttling solution and is most useful as a reference
for compact instances.

```console
mqt-ionshuttler-exact --help
mqt-ionshuttler-exact inputs/algorithms_exact/qft_06.json
```

Pass `--plot` to visualize the result. Example architecture and algorithm files
are available in
[`inputs/algorithms_exact`](https://github.com/munich-quantum-toolkit/ionshuttler/tree/main/inputs/algorithms_exact).

## Heuristic compilation

The heuristic compiler scales to larger circuits and supports one or several
processing zones. It trades an optimality guarantee for practical runtime.

```console
mqt-ionshuttler-heuristic --help
mqt-ionshuttler-heuristic inputs/algorithms_heuristic/qft_60_4pzs.json
```

Example inputs are available in
[`inputs/algorithms_heuristic`](https://github.com/munich-quantum-toolkit/ionshuttler/tree/main/inputs/algorithms_heuristic).

The optional dependency-aware mode can schedule ready gates according to the
current ion positions instead of following one fixed gate sequence. The shared
fine-grained tabu partitioner is available from
{py:mod}`mqt.ionshuttler.partitioning`.

## Interface status

These tools keep their established command-line and JSON interfaces. The shared
compiler contracts currently used by the Linear backend are intended to support
the grid abstraction in a later refactor. Existing grid workflows remain
available during that transition.

## See also

- {doc}`hardware_models` — compare the Linear and grid abstractions
- {doc}`references` — publications describing the exact and heuristic methods
