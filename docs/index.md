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

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by
the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the
[Technical University of Munich](https://www.tum.de/) and supported by
[MQSC](https://mq.sc). Among others, it is part of the
[Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss)
ecosystem, which is being developed as part of the
[Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<div style="margin-top: 0.5em">
<div class="only-light" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Banner">
</div>
<div class="only-dark" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%" alt="MQT Banner">
</div>
</div>

Thank you to all the contributors who have helped make the MQT IonShuttler a
reality!

<p align="center">
<a href="https://github.com/munich-quantum-toolkit/ionshuttler/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/ionshuttler" />
</a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the
future. We are firmly committed to keeping it open and actively maintained for
the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories:
  <https://github.com/munich-quantum-toolkit>
- Contributing code, documentation, tests, or examples via issues and pull
  requests
- Citing the MQT in your publications (see {doc}`References <references>`)
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: <https://github.com/sponsors/munich-quantum-toolkit>

<p align="center">
<iframe src="https://github.com/sponsors/munich-quantum-toolkit/button" title="Sponsor munich-quantum-toolkit" height="32" width="114" style="border: 0; border-radius: 6px;"></iframe>
</p>
