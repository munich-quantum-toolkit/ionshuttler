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

The MQT IonShuttler is a tool for generating shuttling schedules for trapped-ion quantum computers with a grid-type Memory Zone based on the Quantum Charge Coupled Device (QCCD) architecture.
It is part of the [_Munich Quantum Toolkit (MQT)_](https://mqt.readthedocs.io).

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/ionshuttler">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

## Key Features

- **Exact shuttling schedules** for small architectures with **a single processing zone (PZ)**
- **Heuristic shuttling schedules** for larger devices with **one _or_ multiple processing zones**

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

Thank you to all the contributors who have helped make the MQT IonShuttler a reality!

<p align="center">
  <a href="https://github.com/munich-quantum-toolkit/ionshuttler/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/ionshuttler" />
  </a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: https://github.com/munich-quantum-toolkit
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see [References](https://mqt.readthedocs.io/projects/ionshuttler/en/latest/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: https://github.com/sponsors/munich-quantum-toolkit

<p align="center">
  <a href="https://github.com/sponsors/munich-quantum-toolkit">
  <img width=20% src="https://img.shields.io/badge/Sponsor-white?style=for-the-badge&logo=githubsponsors&labelColor=black&color=blue" alt="Sponsor the MQT" />
  </a>
</p>

## Getting Started

`mqt.ionshuttler` is available via [PyPI](https://pypi.org/project/mqt.ionshuttler/).

```console
(.venv) $ pip install mqt.ionshuttler
```

To print information about the usage of the supported scripts, run:

```console
(.venv) $ mqt-ionshuttler-exact --help
(.venv) $ mqt-ionshuttler-heuristic --help
```

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/ionshuttler).**

## System Requirements

The MQT IonShuttler can be installed on all major operating systems with all [officially supported Python versions](https://devguide.python.org/versions/).
Building (and running) is continuously tested under Linux, macOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/runner-images).

## Cite This

Please cite the work that best fits your use case.

### MQT IonShuttler (the tool)

When citing the software itself or results produced with it, cite the MQT IonShuttler paper:

```bibtex
@article{schoenberger2024shuttling,
  title        = {Shuttling for Scalable Trapped-Ion Quantum Computers},
  author       = {Schoenberger, Daniel and Hillmich, Stefan and Brandl, Matthias and Wille, Robert},
  year         = 2024,
  journal      = {IEEE Trans. on CAD of Integrated Circuits and Systems},
  volume       = {44},
  number       = {6},
  pages        = {2144–2155},
  doi          = {10.1109/TCAD.2024.3513262},
  eprint       = {2402.14065},
  eprinttype   = {arXiv}
}
```

### The Munich Quantum Toolkit (the project)

When discussing the overall MQT project or its ecosystem, cite the MQT Handbook:

```bibtex
@inproceedings{mqt,
  title        = {The {{MQT}} Handbook: {{A}} Summary of Design Automation Tools and Software for Quantum Computing},
  shorttitle   = {{The MQT Handbook}},
  author       = {Wille, Robert and Berent, Lucas and Forster, Tobias and Kunasaikaran, Jagatheesan and Mato, Kevin and Peham, Tom and Quetschlich, Nils and Rovara, Damian and Sander, Aaron and Schmid, Ludwig and Schoenberger, Daniel and Stade, Yannick and Burgholzer, Lukas},
  year         = 2024,
  booktitle    = {IEEE International Conference on Quantum Software (QSW)},
  doi          = {10.1109/QSW62656.2024.00013},
  eprint       = {2405.17543},
  eprinttype   = {arxiv},
  addendum     = {A live version of this document is available at \url{https://mqt.readthedocs.io}}
}
```

### Peer-Reviewed Research

When citing the underlying methods and research, please reference the most relevant peer-reviewed publications from the list below:

[[1]](https://arxiv.org/pdf/2311.03454)
D. Schoenberger, S. Hillmich, M. Brandl, and R. Wille.
Using Boolean Satisfiability for Exact Shuttling in Trapped-Ion Quantum Computers.
_Asia and South Pacific Design Automation Conference_, 2024.

[[2]](https://arxiv.org/pdf/2402.14065)
D. Schoenberger, S. Hillmich, M. Brandl, and R. Wille.
Shuttling for Scalable Trapped-Ion Quantum Computers.
_IEEE Trans. on CAD of Integrated Circuits and Systems 44, 2144_, 2024.

[[3]](https://arxiv.org/abs/2505.07928)
D. Schoenberger and R. Wille
Orchestrating Multi-Zone Shuttling in Trapped-Ion Quantum Computers.

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
