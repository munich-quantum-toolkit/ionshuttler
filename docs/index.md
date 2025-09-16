# MQT IonShuttler

MQT IonShuttler provides a framework to utilize quantum computing as a technology for users with little to no quantum computing knowledge
It is developed as part of the _{doc}`Munich Quantum Toolkit (MQT) <mqt:index>`_.

We recommend you to start with the {doc}`installation instructions <installation>`.
Then proceed with the sections below.
If you are interested in the theory behind MQT IonShuttler, have a look at the publications in the {doc}`publication list <references>`.

We appreciate any feedback and contributions to the project.
If you want to contribute, you can find more information in the {doc}`contribution guide <contributing>`.
If you are having trouble with the installation or the usage of MQT QuSAT, please let us know on our {doc}`support page <support>`.

## Usage

### Exact Solution (single PZ)

```console
(.venv) $ python scripts/run_exact.py --help
(.venv) $ python scripts/run_heuristic.py inputs/algorithms_exact/qft_06.json
```

The script supports an additional `--plot` argument to visualise the result.
Architectures and algorithms are specified in JSON files.
For examples, see [`inputs/algorithms_exact`](https://github.com/munich-quantum-toolkit/ionshuttler/blob/main/inputs/algorithms_exact/).

### Heuristic Solution (single & multiple PZs)

```console
(.venv) $ python scripts/run_heuristic.py --help
(.venv) $ python scripts/run_heuristic.py inputs/algorithms_heuristic/qft_60_4pzs.json
```

Architectures and algorithms are specified in JSON files.
For examples, see [`inputs/algorithms_heuristic`](https://github.com/munich-quantum-toolkit/ionshuttler/blob/main/inputs/algorithms_heuristic/).

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
references
```

```{toctree}
:caption: Developers
:glob:
:hidden:
:maxdepth: 1

contributing
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

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<div style="margin-top: 0.5em">
<div class="only-light" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Banner">
</div>
<div class="only-dark" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%" alt="MQT Banner">
</div>
</div>

Thank you to all the contributors who have helped make MQT IonShuttler a reality!

<p align="center">
<a href="https://github.com/munich-quantum-toolkit/ionshuttler/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/ionshuttler" />
</a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: [https://github.com/munich-quantum-toolkit](https://github.com/munich-quantum-toolkit)
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see {doc}`References <references>`)
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: [https://github.com/sponsors/munich-quantum-toolkit](https://github.com/sponsors/munich-quantum-toolkit)

<p align="center">
<iframe src="https://github.com/sponsors/munich-quantum-toolkit/button" title="Sponsor munich-quantum-toolkit" height="32" width="114" style="border: 0; border-radius: 6px;"></iframe>
</p>
