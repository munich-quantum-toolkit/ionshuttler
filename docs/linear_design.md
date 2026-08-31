---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Linear Compiler Architecture

This page gives a high-level view of the Linear compiler package: how its main
pieces fit together, where the hardware model ends and compiler policy begins,
and how the design can be extended. For instructions on compiling a circuit, see
the {doc}`Linear compiler guide <linear_compiler>`. The
{doc}`trapped-ion hardware model <linear_hardware_model>` describes the physical
abstractions behind the architecture.

## Overall structure

The design follows the same broad separation used in a physical control stack.
The hardware model describes what the device offers and what each control
operation does. The compiler decides when to use those operations to realize a
circuit.

<figure>
  <a href="_static/linear_compiler_sequence.png">
    <img src="_static/linear_compiler_sequence.png" width="100%" alt="Sequence
    diagram showing a client invoking the Linear compiler, which parses the
    circuit, searches for a schedule, and returns the compilation result">
  </a>
  <figcaption>The main compilation path from circuit input to a schedule and
  completion status.</figcaption>
</figure>

This direction of dependency is intentional. The compiler knows enough about the
hardware to construct a valid schedule, while the hardware model does not need
to know about A-star search, circuit dependencies, or compiler heuristics.
Likewise, later analysis and control passes consume a compiled schedule without
becoming dependencies of the base search.

## The main layers

Reading the figure from left to right, the public compiler coordinates circuit
preparation, schedule search, and result construction. The hardware model and
compiler state support that path without depending on the search strategy.

### Public compiler interface

{py:class}`~mqt.ionshuttler.linear.compiler.LinearCompiler` is the main entry
point. It holds an architecture and configuration, accepts a circuit, creates
its initial state, and returns a
{py:class}`~mqt.ionshuttler.linear.CompilationResult`.

The facade keeps these preparation steps consistent and keeps the lower-level
search machinery out of normal user code. Compilation itself has no output-file
side effects; saving a result is a separate, explicit operation.

### Hardware model

The hardware-facing part consists of three ideas:

- {py:class}`~mqt.ionshuttler.linear.Architecture` describes sites, processing
  zones, and optional site-dependent field information.
- {py:class}`~mqt.ionshuttler.linear.HardwareTiming` describes the duration and
  implementation of transport and gate operations.
- {py:class}`~mqt.ionshuttler.linear.actions.Action` represents a control
  primitive that may change the hardware state.

An action owns the rules and effects that are intrinsic to that operation:

```python
class Action:
    def is_valid(self, state, architecture) -> bool: ...
    def apply(self, state, architecture): ...
    def to_dict(self) -> dict[str, object]: ...
```

For example, a shuttle knows that it moves one ion between adjacent sites, and a
gate knows which ions and processing-zone resources it occupies. These rules
live with the action so adding a new hardware primitive does not require a
central list of every supported action type.

### Compiler state

{py:class}`~mqt.ionshuttler.linear.state.State` is the meeting point between the
hardware model and the scheduler. It records:

- the current ion positions, device time, and resource availability; and
- which circuit gates are complete or still running.

States are immutable and hashable. The search can therefore compare candidate
schedules and recognize a previously visited state without worrying that an
earlier candidate was modified in place.

Physical actions update only the hardware-related part of the state. The
compiler separately updates gate progress. This split is intentional to mirror
the relationship between an application-facing compiler and the
application-agnostic physical device controller.

### Circuit preparation

Circuit preparation translates Qiskit or QASM input into supported gate actions
and identifies which gates may run independently. It also applies the selected
hardware timing and virtual-gate choices. This layer sits above the hardware
primitives and immediately before schedule search. Circuit-level optimization is
deliberately outside the scope of the Linear compiler.

### Scheduling and search

The compiler owns decisions involving more than one action or the logical
circuit as a whole. This includes:

- deciding which gates are ready from their dependencies;
- proposing transport and gate actions;
- checking conflicts between simultaneous operations;
- advancing time and completing scheduled gates; and
- choosing which candidate schedule to explore next.

The current search uses elapsed schedule time and, by default, a routing/depth
heuristic to guide A-star search. Search settings control practical tradeoffs
such as heuristic choice, rolling horizons, frontier size, and time limits. A
zero heuristic supports an exact but potentially much slower search profile.
Hardware timing remains separate from these choices: changing a search
preference does not redefine how long the hardware takes to perform an
operation.

Individual action validity and scheduler policy are deliberately distinct. An
idle time advance, for example, can be a meaningful operation in the state
model, while the compiler may choose not to generate unhelpful idle branches.
This prevents a search preference from being mistaken for a physical rule.

### Compilation result and downstream work

The result contains the selected action sequence, completion status, makespan,
and supporting state and architecture information. It is the boundary between
base compilation and later consumers such as visualization, analysis, or
additional control passes.

Downstream passes sit on top of compilation. They consume an `ActionSchedule`
and produce a revised schedule together with method-specific diagnostics. The
schedule retains stable action identifiers, while provenance such as the IDs of
inserted local DD pulses belongs to the pass report.

Examples of downstream consumers include:

- shuttling-aware dynamical-decoupling passes; and
- visualization tools that consume the JSON representation.

## Extending the hardware model

The hardware model is designed with future additions and custom modifications in
mind. The following sections outline how to extend the abstraction.

### Define a custom action

A hardware-specific control primitive can subclass `Action` or one of its
physical-action bases. For example, a hardware model could expose a controlled-X
gate directly:

```{code-cell} ipython3
from dataclasses import dataclass

from mqt.ionshuttler.linear import DEFAULT_ACTION_TYPES, LinearCompiler
from mqt.ionshuttler.linear.actions import TwoQubitGate


@dataclass(frozen=True)
class CX(TwoQubitGate):
    circuit_name = "cx"
    duration: int = 2
```

The inherited behavior requires both ions to be free and located in the same
available processing zone. The inherited `to_dict` method also serializes the
gate without another central dispatch branch.

Pass the supported action types when constructing the `Architecture`:

```{code-cell} ipython3
from mqt.ionshuttler.linear import Architecture

architecture = Architecture(
    num_sites=5,
    processing_zones={"gate_zone": [2, 3]},
    supported_action_types=(*DEFAULT_ACTION_TYPES, CX),
)
compiler = LinearCompiler(architecture)
```

The architecture catalog describes the complete set of hardware operations. The
compiler uses that catalog by default, or it can receive an explicit supported
subset through `action_types`. It rejects a circuit immediately when the
selected subset lacks a required gate. For operations such as shuttling that are
proposed from the current hardware state, the action class additionally provides
its available instances. `AdvanceTime` remains part of the scheduler and does
not belong in the hardware catalog. The compiler recognizes the class's
`circuit_name` in either QASM or Qiskit input. `TwoQubitGate` checks the two
operands and constructs the custom gate; parameterized gate families can declare
their parameter names or override `from_instruction` when they need different
lowering behavior.

Note that CX is used here as a familiar example. Controlled-X gates are not
typically native in real-world trapped ion hardware. The default catalog uses
the common trapped-ion gate set composed of `Rx`, `Ry`, `Rz`, and `Rzz`.

If serialized results contain a custom action, provide its decoder explicitly
when restoring them:

```{code-cell} ipython3
from qiskit import QuantumCircuit

from mqt.ionshuttler.linear import CompilationResult

custom_circuit = QuantumCircuit(2)
custom_circuit.cx(0, 1)

serialized_result = compiler.compile(custom_circuit).to_json()
result = CompilationResult.from_json(serialized_result, action_types=(CX,))

next(action for action in result.schedule.path if isinstance(action, CX))
```

The architecture records the supported catalog alongside the action schedule in
the compilation result. Built-in actions decode automatically; supplying a
custom class lets both the architecture catalog and scheduled actions be
restored. Actions with nested or specialized serialized values can override
`from_dict`. Catalogs are supplied per call rather than installed in global
state, so separate applications can use different action sets without affecting
one another.

## See also

- {doc}`linear_compiler` — configure and run the Linear compiler
- {doc}`linear_dd` — apply and compare dynamical-decoupling methods
- {doc}`linear_hardware_model` — physical interpretation of the modeled device
- {doc}`api/mqt/ionshuttler/linear/index` — complete Linear Python API reference
