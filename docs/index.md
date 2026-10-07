# MQT IonShuttler

MQT IonShuttler compiles quantum circuits into ion movements and gate schedules
for quantum charge-coupled devices (QCCDs). Use it to explore how architecture,
routing, and control choices affect trapped-ion execution.

It is part of the {doc}`Munich Quantum Toolkit (MQT) <mqt:index>`.

## Choose a workflow

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Grid compilers
:link: grid_compilers
:link-type: doc

Route ions through networks of trap segments. Use exact compilation for small
single-zone instances or heuristic compilation for larger multi-zone devices.
:::

:::{grid-item-card} Linear compiler
:link: linear_compiler
:link-type: doc

Compile Qiskit or QASM circuits for an ordered array of sites. Configure
processing zones and operation timing, inspect explicit schedules, visualize ion
trajectories, and apply dynamical decoupling.
:::

::::

Start with the {doc}`installation guide <installation>`, then choose the
abstraction and compiler that match your task. The
{doc}`hardware model overview <hardware_models>` explains how both workflows
represent a QCCD.

## What you can study

- ion placement, transport, and gate scheduling;
- one or several processing zones;
- exact and heuristic compilation methods;
- schedule cost, makespan, and transport overhead; and
- dynamical decoupling for Linear schedules.

<p align="center">
  <a href="_static/qccd_device.pdf">
  <img src="_static/qccd_device.png" width="63%" alt="QCCD device with four processing zones">
  </a>
  <a href="_static/graph.pdf">
  <img src="_static/graph.png" width="33%" alt="Graph abstraction of the QCCD device">
  </a>
</p>
<p align="center"><b>QCCD layout and its scheduling abstraction.</b></p>

## Learn more

- {doc}`Grid compilers <grid_compilers>` — run the exact and heuristic tools
- {doc}`Linear compiler <linear_compiler>` — build and inspect Linear schedules
- {doc}`Dynamical decoupling <linear_dd>` — add DD to Linear schedules
- {doc}`Compiler design <design>` — understand and extend the software
- {doc}`References <references>` — cite the relevant methods

We welcome feedback and contributions. See the
{doc}`contribution guide <contributing>` or visit the
{doc}`support page <support>` if you need help.

```{toctree}
:hidden:

self
```

```{toctree}
:caption: User Guide
:glob:
:hidden:
:maxdepth: 1

installation
hardware_models
grid_compilers
linear_compiler
linear_dd
references
```

```{toctree}
:caption: Developers
:glob:
:hidden:
:maxdepth: 1

contributing
ai_usage
tooling
design
support
```

```{toctree}
:caption: Python API Reference
:glob:
:hidden:
:maxdepth: 6

api/mqt/ionshuttler/index
```

## Open source

MQT IonShuttler is free, open-source, and permissively licensed. It is developed
by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the
[Technical University of Munich](https://www.tum.de/) as part of the Munich
Quantum Toolkit.

Please consider starring the project, sharing your use case, contributing a
change, or citing the relevant work from the {doc}`reference list <references>`.
