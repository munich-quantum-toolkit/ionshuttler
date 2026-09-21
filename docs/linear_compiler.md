---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Linear Compiler

The Linear compiler turns a quantum circuit into a timed sequence of gates and
ion movements for a one-dimensional array of sites. Processing zones mark the
sites where gates may run; the remaining sites can be used to store ions while
the compiler brings the next operands together.

The compiler is intended for Python workflows. Supply a hardware layout and a
Qiskit circuit, QASM text, or a QASM file, then inspect the returned
{py:class}`~mqt.ionshuttler.linear.CompilationResult`.

## Compile a circuit

The following example uses five sites and one two-site processing zone. The
compiler chooses the necessary shuttles, swaps, and gate start times.

```{code-cell} ipython3
from qiskit import QuantumCircuit

from mqt.ionshuttler.linear import Architecture, CompilationStatus, LinearCompiler

architecture = Architecture(
    num_sites=5,
    processing_zones={"gate_zone": [2, 3]},
)

circuit = QuantumCircuit(2)
circuit.rx(0.25, 0)
circuit.ry(0.5, 1)
circuit.rzz(0.75, 0, 1)

result = LinearCompiler(architecture).compile(circuit)

if result.status is CompilationStatus.SUCCESS:
    print(f"Compiled in {result.schedule.num_timesteps} timesteps")
    for action in result.schedule.path:
        print(action)
else:
    print(f"Compilation stopped with {result.status.value}")
```

By default, ions are distributed across the available sites. Pass one starting
site per circuit qubit when a particular loading is required:

```{code-cell} ipython3
result = LinearCompiler(architecture).compile(
    circuit,
    initial_positions=[0, 4],
)
```

The positions must be distinct, lie within the architecture, and match the
number of circuit qubits. See also the
{doc}`trapped-ion hardware model <linear_hardware_model>` for further details on
the hardware abstraction.

| Compile option      | User-visible effect                                                                                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `initial_positions` | Selects one distinct starting site per circuit qubit. When omitted, the compiler distributes the ions across the available sites.                                                                               |
| `pre_partition`     | On layouts with multiple processing zones, computes a fine-grained gate-to-zone preference once and uses it to guide search. It defaults to `False` and has no effect on layouts with only one processing zone. |

For example, enable the partition-guided heuristic for a multi-zone layout at
the compilation call:

```{code-cell} ipython3
multi_zone_architecture = Architecture(
    num_sites=7,
    processing_zones={"left": [1, 2], "right": [4, 5]},
)
partitioned_result = LinearCompiler(multi_zone_architecture).compile(
    circuit,
    pre_partition=True,
)
```

Pre-partitioning can substantially reduce compilation time when a circuit lacks
spatial locality matching the zone layout. It is not a strict improvement:
circuits whose interactions already match the layout may see little benefit or
an occasional small slowdown. Advanced users can tune the partitioner through
`SearchConfig.pre_partition_config` using a
{py:class}`~mqt.ionshuttler.partitioning.FineGrainedTabuConfig`.

### Circuit inputs

{py:meth}`~mqt.ionshuttler.linear.compiler.LinearCompiler.compile` accepts:

- a Qiskit {py:class}`qiskit.circuit.QuantumCircuit`;
- a string containing OpenQASM 2 or the supported subset of OpenQASM 3; or
- a {py:class}`pathlib.Path` to a UTF-8 QASM file.

Strings are always treated as QASM text. Use `Path("circuit.qasm")` for a file.
The default gate set is `rx`, `ry`, `rz`, and `rzz`, a common trapped-ion gate
set. Additional implemented gates may be enabled through the compiler's
`action_types`. Rotation parameters must be numeric when compilation begins.
Barriers and final measurements are accepted but do not become scheduled
hardware operations. Unsupported or malformed circuits raise an exception before
schedule search starts.

## Describe the architecture

Architectures can be created in Python, as above, or loaded from JSON:

```{code-cell} ipython3
from mqt.ionshuttler.linear import Architecture

architecture = Architecture.from_json(
    """{
      "num_sites": 7,
      "processing_zones": {
        "left_zone": [1, 2],
        "right_zone": [5, 6]
      }
    }"""
)
```

Use `Architecture.load("architecture.json")` to read the same JSON
representation from a file.

Site numbers start at zero. Each processing zone must contain one or more
contiguous sites, and processing zones may not overlap. A one-qubit gate needs
its ion in a processing zone. A two-qubit gate needs both ions in the same zone;
they do not need to occupy adjacent sites within that zone. If
`processing_zones` is omitted, all sites form one processing zone.

An architecture may also contain a site-dependent `field_profile`. The Linear
compiler carries this information forward for later analysis; it does not use
the profile to choose the base schedule and does not normalize its values.

## Hardware timing

The timing configuration tells the compiler how long the hardware operations
take. Defaults are measured in compiler timesteps:

| Operation                         | Default duration |
| --------------------------------- | ---------------: |
| Shuttle to an adjacent empty site |                1 |
| Swap adjacent ions                |                3 |
| `rx`, `ry`                        |                1 |
| `rz`                              |       0, virtual |
| `rzz`                             |                2 |
| Optional `rxx`, `ryy`             |                2 |

For example:

```{code-cell} ipython3
from mqt.ionshuttler.linear import (
    GateTiming,
    HardwareTiming,
    LinearCompiler,
    LinearCompilerConfig,
    TransportTiming,
)

timing = HardwareTiming(
    transport=TransportTiming(shuttle=2, swap=5),
    gates=GateTiming(rx=2, ry=2, rz=0, rxx=4, ryy=4, rzz=4),
)
config = LinearCompilerConfig(hardware_timing=timing)
compiler = LinearCompiler(architecture, config=config)
```

`rx`, `ry`, and `rz` may be selected as virtual single-qubit gates through
`GateTiming.virtual_single_qubit_gates`. Virtual rotations must have zero
duration and remain in the logical schedule without reserving hardware time.
`rz` is virtual by default, reflecting a common trapped-ion implementation. When
reproducing a schedule created with different timing assumptions, pass those
durations explicitly.

## Choose a search profile

The compiler uses A-star search: it compares the time already used with an
estimate of the work still needed and explores promising schedules first
{cite:p}`hart1968formal`. This estimate is called a *heuristic*. An admissible
heuristic never overestimates the remaining cost; with the usual unrestricted
A-star conditions, that property is what supports an optimality guarantee.

The Linear heuristic combines estimated ion movement with remaining gate depth.
Because movements and gates may overlap, this estimate is
**not guaranteed to be admissible**. The compiler should therefore be viewed as
a schedule-quality search, not as an exact optimizer. A successful result is a
valid complete schedule, but it is not a proof that no shorter schedule exists.

The defaults favor practical compilation time:

```{code-cell} ipython3
from mqt.ionshuttler.linear import LinearCompilerConfig, SearchConfig

config = LinearCompilerConfig(
    search=SearchConfig(
        horizon=3,
        committed_gates=2,
        iterative_diving_search=True,
        informed_action_prioritization=False,
        num_solutions=1,
        max_frontier_size=1000,
        max_compile_time=1800.0,
        use_dependencies=True,
    )
)
```

| Setting                          | User-visible effect                                                                                                                                                                                                                                                    |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `horizon`                        | Plans this many upcoming gates at a time. Small horizons are usually faster and use less memory, but make more local decisions. `None` considers the complete remaining circuit together; it can be much more expensive and still does not prove optimality.           |
| `committed_gates`                | Number of gates fixed before a rolling-horizon replan. A smaller value preserves more flexibility but replans more often. It must not exceed `horizon`.                                                                                                                |
| `iterative_diving_search`        | Immediately follows a promising branch while retaining alternatives. This often reaches a schedule sooner, but changes search order and tends to yield lower-quality results than standard branching. It independently breaks admissibility and optimality guarantees. |
| `informed_action_prioritization` | Tries a narrower set of promising movements before falling back to the full set. It may reduce work, especially under a time limit, but can change which schedule is found first.                                                                                      |
| `num_solutions`                  | Continues until this many distinct completed states have been found, then returns the best of them. Larger values can improve the returned schedule at additional cost, without proving global optimality.                                                             |
| `max_frontier_size`              | Limits stored alternatives. Smaller bounds reduce memory, but discarded candidates can contain the best or only solution. Use `None` to retain an unbounded frontier.                                                                                                  |
| `max_compile_time`               | Shared wall-clock budget in seconds. A finite value prevents unexpectedly long runs; `None` removes the time limit.                                                                                                                                                    |
| `use_dependencies`               | `True` respects circuit dependencies while allowing independent gates to overlap. `False` conservatively schedules gates in their input order and may increase the makespan.                                                                                           |
| `heuristic`                      | Remaining-cost estimate guiding the search. `None` uses the faster default, which is not guaranteed to be admissible. `zero_heuristic` uses no estimate; it is admissible, but usually explores many more states. Any other `HeuristicFn` may be supplied.             |
| `pre_partition_config`           | Optional fine-grained tabu settings used when `compile(..., pre_partition=True)` is selected. `None` uses the partitioner's defaults.                                                                                                                                  |

A useful, more thorough profile for small circuits is:

```{code-cell} ipython3
search = SearchConfig(
    horizon=None,
    committed_gates=1,
    iterative_diving_search=False,
    num_solutions=10,
    max_frontier_size=None,
    max_compile_time=None,
)
```

This removes the rolling horizon, frontier bound, and timeout, and asks for
several solutions. Runtime and memory can grow rapidly. Because the heuristic is
not guaranteed admissible and the search stops after the requested number of
solutions, this remains a quality-oriented search rather than an exact
optimality mode.

For small instances where a minimum-makespan result matters more than search
speed, all quality-oriented shortcuts can be disabled:

```{code-cell} ipython3
from mqt.ionshuttler.linear import zero_heuristic

exact_search = SearchConfig(
    horizon=None,
    committed_gates=1,
    iterative_diving_search=False,
    informed_action_prioritization=False,
    num_solutions=1,
    max_frontier_size=None,
    max_compile_time=None,
    use_dependencies=True,
    heuristic=zero_heuristic,
)
```

This turns the search into an unrestricted uniform-cost search. The first
complete schedule therefore has the minimum makespan among the schedules
represented by the configured hardware and compiler action model. Runtime and
memory can grow very quickly, so this profile is mainly useful for small
circuits, reference results, and comparisons. A timeout, frontier bound, rolling
horizon, iterative dive, or narrowed action generation removes that optimality
guarantee.

## Supplying a custom heuristic

`SearchConfig.heuristic` selects the remaining-cost estimate. Besides the
default (`None`) and :func:`~mqt.ionshuttler.linear.cost.zero_heuristic`, any
callable matching :class:`~mqt.ionshuttler.linear.cost.HeuristicFn` may be
supplied. It receives the current state, the architecture, the gates to
schedule, and the dependency map, and returns a nonnegative estimate of the
remaining schedule time. `gate_order` and `gates` also include gates that are
already completed or running, so you might want to filter them with
`state.completed_gates` and `state.in_progress_gates`.

Because a larger estimate marks a state as further from the goal, adding a
penalty steers the search away from the states it describes. This heuristic adds
one unit of estimated work for every pair of ions sitting on neighboring sites,
biasing the search toward schedules that keep ions spread out:

```{code-cell} ipython3
from itertools import combinations

from mqt.ionshuttler.linear import SearchConfig
from mqt.ionshuttler.linear.cost import heuristic as default_heuristic


def prefer_spread_out_ions(state, architecture, gate_order, gates, predecessors=None):
    """Penalize neighboring ions on top of the default estimate."""
    base = default_heuristic(state, architecture, gate_order, gates, predecessors)
    sites = [site for _, site in state.positions]
    crowding = sum(1 for left, right in combinations(sites, 2) if abs(left - right) <= 1)
    return base + crowding


custom_search = SearchConfig(heuristic=prefer_spread_out_ions)
```

The estimate steers the search only; it never changes which schedules are legal.
An estimate that overshoots the true remaining time gives up admissibility,
exactly as the default already does.

## Understand the result

The result status describes why compilation stopped:

| Status        | Meaning                                                                       |
| ------------- | ----------------------------------------------------------------------------- |
| `SUCCESS`     | Every circuit gate was scheduled.                                             |
| `TIMEOUT`     | The time budget expired; the result contains the best partial schedule found. |
| `INTERRUPTED` | Compilation was interrupted; the best available partial schedule is returned. |
| `FAILED`      | The search ran out of candidates before completing the circuit.               |

`result.schedule` is the immutable
{py:class}`~mqt.ionshuttler.linear.ActionSchedule` that downstream DD and
simulation stages consume. Its `path` contains the ordered gate, movement, and
time-advance actions, and its `num_timesteps` is the makespan. It also contains
the machine-only initial state and a stable identifier for every action. The
architecture is stored alongside it as `result.architecture`, allowing later
passes to supply a compatible architecture with additional capabilities.
Compiler search progress and control-pass provenance are deliberately absent
from this execution boundary.

The containing result owns compiler-only diagnostics: `result.wall_clock_s`,
`result.score`, `result.final_state`, and `result.explored_nodes`. Always check
`status` before treating its schedule as a complete circuit schedule.

Invalid input and configuration errors raise exceptions instead of returning
`FAILED`. Use `result.save(...)` to write beneath `outputs/results/json` in the
current working directory by default, or pass `directory` explicitly. Use
{py:meth}`~mqt.ionshuttler.linear.CompilationResult.load` for explicit JSON
output; compilation does not create caches or result files on its own.

## See also

- {doc}`linear_hardware_model` — sites, processing zones, timing, and physical
  assumptions
- {doc}`linear_dd` — dynamical-decoupling methods and comparison metrics
- {doc}`linear_design` — package architecture and custom extension points
- {doc}`api/mqt/ionshuttler/linear/index` — complete Linear Python API reference
