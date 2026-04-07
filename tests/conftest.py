"""Shared fixtures for mqt-ionshuttler tests."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUTS_DIR = ROOT_DIR / "inputs"
EXACT_DIR = INPUTS_DIR / "algorithms_exact"
HEURISTIC_DIR = INPUTS_DIR / "algorithms_heuristic"
QASM_DIR = INPUTS_DIR / "qasm_files"


_pytest = cast("Any", importlib.import_module("pytest"))
_F = TypeVar("_F", bound=Callable[..., object])
ConfigDict = dict[str, object]


def fixture(func: _F) -> _F:
    """Typed wrapper around pytest.fixture."""
    return cast("_F", _pytest.fixture(func))


def _load_config(path: Path) -> ConfigDict:
    """Load and validate a JSON config object."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        msg = f"Expected JSON object in {path}, got {type(data).__name__}."
        raise TypeError(msg)
    return cast("ConfigDict", data)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@fixture
def exact_config_qft05() -> ConfigDict:
    """Load the small exact QFT-5 config."""
    return _load_config(EXACT_DIR / "qft_05.json")


@fixture
def exact_config_full_register() -> ConfigDict:
    """Load the full-register-access exact config."""
    return _load_config(EXACT_DIR / "full_register_access.json")


@fixture
def heuristic_config_1pz() -> ConfigDict:
    """Load the 1-PZ heuristic config."""
    return _load_config(HEURISTIC_DIR / "qft_06_1pz.json")


@fixture
def heuristic_config_2pzs() -> ConfigDict:
    """Load the 2-PZ heuristic config."""
    return _load_config(HEURISTIC_DIR / "qft_06_2pzs.json")


# ---------------------------------------------------------------------------
# Single-shuttler graph fixtures
# ---------------------------------------------------------------------------


@fixture
def small_grid_graph():
    """Create a small 2x2 grid graph via the single_shuttler create_graph."""
    from mqt.ionshuttler.single_shuttler.memory_sat import create_graph

    return create_graph(m=2, n=2, ion_chain_size_vertical=1, ion_chain_size_horizontal=1)


@fixture
def medium_grid_graph():
    """Create a 3x3 grid graph via the single_shuttler create_graph."""
    from mqt.ionshuttler.single_shuttler.memory_sat import create_graph

    return create_graph(m=3, n=3, ion_chain_size_vertical=1, ion_chain_size_horizontal=1)


# ---------------------------------------------------------------------------
# Multi-shuttler graph fixtures
# ---------------------------------------------------------------------------


@fixture
def multi_processing_zone_1pz():
    """Create a single ProcessingZone instance for multi_shuttler tests."""
    from mqt.ionshuttler.multi_shuttler.outside.processing_zone import ProcessingZone

    m, n, v, h = 3, 3, 1, 1
    return ProcessingZone(
        "pz1",
        [
            (float((m - 1) * v), float((n - 1) * h)),
            (float((m - 1) * v), float(0)),
            (float((m - 1) * v - (-4.5)), float((n - 1) * h / 2)),
        ],
    )


@fixture
def multi_graph_creator_1pz(multi_processing_zone_1pz):
    """Create a GraphCreator and PZCreator for multi_shuttler with 1 PZ."""
    from mqt.ionshuttler.multi_shuttler.outside.graph_creator import GraphCreator, PZCreator

    m, n, v, h = 3, 3, 1, 1
    pzs = [multi_processing_zone_1pz]
    basegraph = GraphCreator(m, n, v, h, 0, pzs, seed=0)
    pzgraph = PZCreator(m, n, v, h, 0, pzs, seed=0)
    return basegraph, pzgraph


# ---------------------------------------------------------------------------
# QASM file fixtures
# ---------------------------------------------------------------------------


@fixture
def qasm_file_qft6() -> Path:
    """Path to the QFT-6 QASM file."""
    return QASM_DIR / "qft_no_swaps_nativegates_quantinuum_tket" / "qft_no_swaps_nativegates_quantinuum_tket_6.qasm"


@fixture
def qasm_file_full_register_6() -> Path:
    """Path to the full-register-access 6-qubit QASM file."""
    return QASM_DIR / "full_register_access" / "full_register_access_6.qasm"
