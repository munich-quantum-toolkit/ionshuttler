[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mqt.ionshuttler.svg)](https://badge.fury.io/py/mqt.ionshuttler)

<p align="center">
  <a href="https://mqt.readthedocs.io">
   <picture>
     <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-dark.svg" width="60%">
     <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-light.svg" width="60%" alt="MQT Logo">
   </picture>
  </a>
</p>

# MQT IonShuttler

<i>MQT IonShuttler</i> is a tool for generating shuttling schedules for trapped-ion quantum computers with a grid-type Memory Zone based on the Quantum Charge Coupled Device (QCCD) architecture. It supports:

- **exact shuttling schedules** for small architectures with **a single processing zone (PZ)**, and
- **heuristic shuttling schedules** for larger devices with **one _or_ multiple processing zones**.

<p align="center">
  <a href="https://github.com/cda-tum/mqt-ion-shuttler/blob/main/docs/figures/QCCD_device.pdf">
    <img src="https://raw.githubusercontent.com/cda-tum/mqt-ion-shuttler/main/docs/figures/QCCD_device.png" width="63%" alt="(a) Potential QCCD device with four processing zones">
  </a>
  <a href="https://github.com/cda-tum/mqt-ion-shuttler/blob/main/docs/figures/graph.pdf">
    <img src="https://raw.githubusercontent.com/cda-tum/mqt-ion-shuttler/main/docs/figures/graph.png" width="33%" alt="(b) Corresponding interaction graph">
  </a>
</p>
<p align="center">
<b>Figure 1:</b> (<b>a</b>) Potential QCCD device with four processing zones; (<b>b</b>) corresponding graph abstraction.</p>

The exact solution guarantees optimality but is limited to a single PZ, while the heuristic method scales to many qubits and PZs. In the heuristic workflow, an optional **compilation** feature (`use_dag`) allows for dynamic rescheduling of gates based on the current ion positions and dependencies, potentially reducing shuttling overhead compared to executing a fixed sequence.

<i>MQT IonShuttler</i> is part of the [_Munich Quantum Toolkit_](https://mqt.readthedocs.io) (MQT) developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/).

If you have any questions, feel free to contact us via [quantum.cda@xcit.tum.de](mailto:quantum.cda@xcit.tum.de) or by creating an [issue](https://github.com/cda-tum/mqt-ion-shuttler/issues) on GitHub.

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
   <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

---

## Installation

We strongly recommend using [virtual environments](https://docs.python.org/3/library/venv.html).

To install MQT IonShuttler and its dependencies, you can use pip:

```bash
pip install mqt.ionshuttler
```

This will install the library components from [PyPI](https://pypi.org/project/mqt.ionshuttler/), making them importable in your Python projects.

## Getting Started with Example Scripts

The run_exact.py and run_heuristic.py scripts provide examples of how to use MQT IonShuttler. To run these scripts and access the example configuration files, it's best to clone the repository:

```bash
git clone https://github.com/cda-tum/mqt-ion-shuttler.git
cd mqt-ion-shuttler
```

Then, set up a virtual environment and install the package (which also installs dependencies):

```bash
python3 -m venv .venv
. .venv/bin/activate # Or .\.venv\Scripts\activate on Windows
pip install .        # Installs the package and its dependencies
# For development, you might prefer: pip install -e . (editable install)
```

Once this is done, you can run the example scripts as shown below. These scripts currently import modules using `from src...` and are designed to be run from the root of the cloned repository after the local installation.

### Exact Solution (single PZ)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
python3 run_exact.py --help
# Example
python3 run_exact.py algorithms_exact/qft_06.json
```

The script supports an additional `--plot` argument to visualise the result. All parameters are documented via `--help`.
Architectures and algorithms are specified in JSON files—see the examples in [`algorithms_exact/`](algorithms_exact/).

### Heuristic Solution (single & multiple PZs)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
python3 run_heuristic.py --help
# Example with 60 qubits executed on 4 PZs
python3 run_heuristic.py algorithms_heuristic/qft_60_4pzs.json
```

Architectures and algorithms are specified in JSON files—see the examples in [`algorithms_heuristic/`](algorithms_heuristic/).

---

## References

This implementation is based on the following publications:

1. D. Schoenberger, S. Hillmich, M. Brandl, and R. Wille, "Using Boolean Satisfiability for Exact Shuttling in Trapped-Ion Quantum Computers," _ASP-DAC_, 2024.
2. D. Schoenberger, S. Hillmich, M. Brandl, and R. Wille, "Shuttling for Scalable Trapped-Ion Quantum Computers," _IEEE TCAD_, 2024.
3. D. Schoenberger, R. Wille, "Orchestrating Multi-Zone Shuttling in Trapped-Ion Quantum Computers".

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European Union's Horizon 2020 research and innovation programme (DA QC, grant agreement No. 101001318 and MILLENION, grant agreement No. 101114305), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
