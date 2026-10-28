---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Dynamic Programs

A dynamic program measures ions during execution and chooses later operations
from the outcomes. {py:func}`~mqt.ionshuttler.dynamic.compiler.compile_program`
compiles such a program for a Grid architecture. The result holds one
precompiled schedule for every straight-line part of the program and a decision
for every branch, so the control system only selects the next schedule when an
outcome arrives.

## Reading programs with MQT Core

Install the `dynamic` extra to read OpenQASM 3 programs with MQT Core:

```console
pip install "mqt.ionshuttler[dynamic]"
```

{py:func}`~mqt.ionshuttler.dynamic.qco.program_from_qasm` imports the program
with MQT Core, synthesizes the native gates `rx`, `ry`, `rz`, and `rxx` together
with measurement and reset, and reads the outcome.

```python
from mqt.ionshuttler.dynamic import program_from_qasm

program = program_from_qasm(
    """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[1] c;
h q[0];
c[0] = measure q[0];
if (c[0]) { x q[1]; }
"""
)
```

If you already have QCO text after native-gate synthesis, use
{py:func}`~mqt.ionshuttler.dynamic.qco.parse_program` instead. Programs can also
be built directly from circuits and branches:

```{code-cell} ipython3
from mqt.ionshuttler.circuit import Circuit
from mqt.ionshuttler.core.gates import Measure, Rx, Rxx
from mqt.ionshuttler.dynamic import Branch, Condition, Program, compile_program
from mqt.ionshuttler.grid import GridCompiler, square_grid

entangle_and_measure = Circuit(
    num_ions=2,
    gates=(Rxx(ion_a=0, ion_b=1, theta=0.5, gate_id=0), Measure(ion=0, outcome=0, gate_id=1)),
    predecessors=(frozenset(), frozenset({0})),
)
correction = Circuit(num_ions=2, gates=(Rx(ion=1, theta=0.5, gate_id=0),), predecessors=(frozenset(),))
program = Program(
    num_ions=2,
    steps=(entangle_and_measure, Branch(Condition(outcomes=(0,), true_for=frozenset({(1,)})), (correction,))),
)

architecture = square_grid(2, processing_zone_segments=("h:0:0",), segment_capacity=2)
result = compile_program(program, GridCompiler(architecture), feedback_delay=2)
result.validate()
for summary in result.path_summaries():
    print(summary)
```

## Execution model

- {py:class}`~mqt.ionshuttler.core.gates.Measure` and
  {py:class}`~mqt.ionshuttler.core.gates.Reset` are timed operations in a
  processing zone. Their durations come from
  {py:class}`~mqt.ionshuttler.core.gates.GateTiming`, and each processing zone
  declares whether it supports them.
- A {py:class}`~mqt.ionshuttler.dynamic.program.Condition` lists the measurement
  outcomes it reads and the outcome combinations for which the branch runs.
- A branch starts no earlier than `feedback_delay` after the last measurement
  that its condition reads.
- Both sides of a branch start from the placement before it. When later steps
  follow the branch, both sides return to that placement as the target placement
  of their last circuit, so the program continues from the same placement on
  every path.
- A {py:class}`~mqt.ionshuttler.dynamic.program.Loop` repeats its steps a fixed
  number of times (`for`) or while a condition holds (`while`). Its body is
  compiled once and returns to the placement in which the loop starts, so every
  iteration runs the same schedules. Outcomes measured inside a loop are keyed
  by outcome number and iteration, such as `(3, 0)`.
- {py:meth}`~mqt.ionshuttler.dynamic.compiler.ProgramResult.validate` checks
  each precompiled schedule against its circuit, replays it once, and checks
  that it starts from every state in which the schedules before it can end. This
  covers every path, also through many loop iterations.
- `paths` and `path_summaries` list paths explicitly. A `while` loop needs
  `max_iterations` there; longer paths are left out.

If a return finds no movement sequence within `max_routing_states`, the result
has status `FAILED` and names the failing branch.
