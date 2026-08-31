# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the ``docs`` Nox session's invocation of ``uv run``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

nox = pytest.importorskip("nox")


def _load_noxfile() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "noxfile.py"
    spec = importlib.util.spec_from_file_location("noxfile", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeVirtualenv:
    location = "fake-venv"


class _FakeSession:
    """Minimal stand-in for ``nox.Session`` that records ``run`` invocations."""

    def __init__(self, *, posargs: list[str] | None = None, interactive: bool = False) -> None:
        self.posargs = posargs or []
        self.interactive = interactive
        self.virtualenv = _FakeVirtualenv()
        self.run_calls: list[tuple[str, ...]] = []
        self.installed: list[tuple[str, ...]] = []

    def install(self, *args: str) -> None:
        self.installed.append(args)

    def run(self, *args: str, **_kwargs: object) -> None:
        self.run_calls.append(args)


def test_docs_session_uv_run_includes_dd_extra_and_docs_group() -> None:
    """The docs session's ``uv run`` call must install the docs group plus the dd extra."""
    noxfile = _load_noxfile()
    session = _FakeSession(posargs=[], interactive=False)

    noxfile.docs(session)

    assert len(session.run_calls) == 1
    (call,) = session.run_calls
    assert call[0] == "uv"
    assert call[1] == "run"
    assert "--group" in call
    assert call[call.index("--group") + 1] == "docs"
    assert "--extra" in call
    assert call[call.index("--extra") + 1] == "dd"
    assert "sphinx-build" in call
