# Compiler Design

This page is the starting point for developers who want to understand or extend
MQT IonShuttler. It describes the implemented package boundaries, the contracts
shared by compiler backends, and the responsibilities that remain specific to a
backend. The Linear compiler serves as the current reference implementation.

For instructions on using that compiler, see the
{doc}`Linear compiler guide <linear_compiler>`. The
{doc}`trapped-ion hardware model <linear_hardware_model>` explains the physical
assumptions behind the Linear architecture.

## Overall structure

The separation has one central rule: An architecture or device defines the
available operations, their validity constraints and how they act on the
hardware state on the respective level of abstraction. A compiler reads these
operations and chooses among them based on internal heuristics.

## Shared packages

### Circuit frontend

The {py:mod}`mqt.ionshuttler.circuit` package converts Qiskit circuits, QASM
text, and QASM files into an immutable `Circuit`. A `Circuit` contains its ion
count, gate occurrences, stable gate identifiers, and dependency relationships.
It contains no hardware timing or transport policy.

This boundary lets a compiler consume one representation regardless of the input
format. Circuit-level optimization currently remains outside the scope of MQT
IonShuttler.

### Core contracts

The {py:mod}`mqt.ionshuttler.core` package defines the shared data exchanged by
compilers, architectures, and downstream tools:

- {py:class}`~mqt.ionshuttler.core.actions.Action` describes an operation
  without assigning it a time or hardware resource.
- Shared gates such as `Rx`, `Rzz`, and `GlobalGate` describe circuit-level
  operations without choosing backend resources.
- {py:class}`~mqt.ionshuttler.core.schedule.ScheduledAction` places one action
  on the timeline with an identity, start time, duration, and optional
  processing-zone identifier. The identifier records the resource selected by
  the backend without defining its geometry. A
  {py:class}`~mqt.ionshuttler.core.schedule.Schedule` collects these entries
  with the initial state and absolute end time. Its `start_time` comes from the
  initial state, and its `duration` is the difference between both times.
  Entries are stored in execution order: start times never decrease, and entries
  with the same start time execute in stored order. This order matters when an
  action depends on a zero-duration action that starts at the same time.
- {py:class}`~mqt.ionshuttler.core.result.CompilationResult` and
  {py:class}`~mqt.ionshuttler.core.result.CompilationStatus` describe the output
  of compilation.

These types are deliberately small. `core` records resolved timing and resource
identifiers, but it does not decide whether an action is valid, how long it
takes, what a resource means, or how an action changes machine state. Each
backend architecture owns those rules.

## Backend responsibilities

A compiler backend combines a compilation method, an architecture, backend
state, and backend-specific actions.

### Compiler

The compiler is the user-facing coordinator. It:

1. parses the input circuit;
2. creates the initial backend state;
3. selects or searches for hardware actions;
4. asks the architecture whether candidate actions are valid and applies them;
5. constructs an explicit schedule; and
6. returns a typed compilation result.

Candidate generation, heuristics, search limits, and decisions involving several
actions belong to compiler policy.

### Architecture

The architecture owns the hardware meaning of each supported action. A backend
architecture must define:

- which action types the hardware supports;
- the duration and occupied resources of an action;
- whether the action can start in a given state;
- how the action changes that state; and
- how an explicit schedule is validated and replayed.

This ownership keeps hardware facts out of the search algorithm. It also lets a
shared gate value retain its identity when different architecture levels assign
different resources or concurrency rules to it.

### Persistence

Core schedules and results accept explicit decoder functions. A backend owns the
loader that supplies its architecture, action, state, and diagnostics decoders.
Saved documents carry a schema name and version. Stable serialized action names
such as `gate.rx` and `linear.shuttle` identify action classes without depending
on Python class names.

## The Linear backend

The {py:mod}`mqt.ionshuttler.linear` package implements this design for a
site-based model of a linear architecture.

{py:class}`~mqt.ionshuttler.linear.compiler.LinearCompiler` accepts a circuit
and combines it with a {py:class}`~mqt.ionshuttler.linear.LinearArchitecture`
and {py:class}`~mqt.ionshuttler.linear.LinearCompilerConfig`. It returns a
`LinearCompilationResult`.

`LinearArchitecture` owns sites, maps processing-zone identifiers to contiguous
sites, and defines gate and transport timing, the supported action catalog, and
the rules used to validate and apply each action. Linear-specific transport
actions and timing live beside that backend; shared gates remain in `core`.

The private Linear search state records ion positions, device time, resource
availability, and circuit progress. Search advances time with an internal
`AdvanceTime` transition. Public schedules do not expose that search mechanism;
idle time appears implicitly as a gap between action intervals.

The Linear search proposes gates, shuttles, and physical swaps, then asks the
architecture to validate and apply them. Its heuristic, rolling horizon,
frontier limit, and time limit affect which valid schedule is found.

## Extending the package

### Change compiler policy

A new heuristic or search method should consume existing circuit, state, and
architecture contracts. It may change which valid candidate is explored first,
but it must not redefine action duration or validity.

### Change architecture behavior

Change the backend architecture when a device gives an existing action a
different duration, resource requirement, concurrency rule, or state effect. For
example, a stricter global-gate concurrency rule belongs in
`LinearArchitecture.is_action_valid` and `LinearArchitecture.apply_action`, not
in `GlobalGate`.

### Add an action type

An action class describes immutable operation data and its stable serialized
name. Supporting a new action also requires backend work:

1. add the action to the backend's implemented action map;
2. define its timing, resources, validity, and state transition;
3. teach the compiler when to propose it;
4. include it in backend decoding; and
5. test compilation, replay, and serialization.

`supported_action_types` is a capability switch over actions that a backend
already implements. It supports device configuration and architecture
comparisons; it is not a plugin registry.

### Add another backend

Another backend can reuse circuit normalization, shared gates, schedules, and
results while defining its own architecture, machine state, actions, compiler,
replay, diagnostics, and loader.

## Results and downstream tools

A compilation result is the boundary between base compilation and later work. It
contains the schedule, status, architecture, initial and final machine state,
and optional backend diagnostics. Downstream analysis or transformation passes
consume that result or its schedule without becoming dependencies of the base
compiler.

{py:func}`mqt.ionshuttler.visualize` selects the built-in visualizer for a
supported result. An explicit
{py:class}`~mqt.ionshuttler.visualization.Visualizer` instead provides its own
`visualize` method. The concrete
{py:class}`~mqt.ionshuttler.visualization.LinearVisualizer` implements that
contract. Backend loaders, such as
{py:func}`mqt.ionshuttler.linear.load_result`, restore persisted artifacts with
explicit backend knowledge.

## See also

- {doc}`linear_compiler` — configure and run the Linear compiler
- {doc}`linear_hardware_model` — Linear sites, processing zones, and timing
- {doc}`linear_dd` — downstream dynamical-decoupling passes
- {doc}`api/mqt/ionshuttler/index` — complete Python API reference
