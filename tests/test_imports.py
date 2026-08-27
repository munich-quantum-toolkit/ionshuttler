# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Basic import and smoke tests for mqt-ionshuttler."""

from __future__ import annotations

import importlib
import pkgutil


def test_package_imports():
    """Test that the top-level package can be imported."""
    import mqt.ionshuttler  # ruff:ignore[unused-import]


def test_single_shuttler_importable():
    """Test that the single_shuttler subpackage can be imported."""
    import mqt.ionshuttler.single_shuttler  # ruff:ignore[unused-import]


def test_multi_shuttler_importable():
    """Test that the multi_shuttler subpackage can be imported."""
    import mqt.ionshuttler.multi_shuttler  # ruff:ignore[unused-import]


def test_all_submodules_importable():
    """Test that every submodule can be imported without error."""
    import mqt.ionshuttler

    # Legacy modules with known issues (deprecated code / broken imports)
    skip = {
        "mqt.ionshuttler.multi_shuttler.inside.run",  # DeprecationWarning on hashing seed
        "mqt.ionshuttler.single_shuttler.run",  # references legacy 'SAT' module
    }

    failed = []
    for _, name, _ in pkgutil.walk_packages(mqt.ionshuttler.__path__, prefix=mqt.ionshuttler.__name__ + "."):
        if name in skip:
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append((name, exc))

    assert not failed, "\n".join(f"{n}: {e}" for n, e in failed)


def test_package_has_version():
    """Test that the package exposes a __version__ string."""
    from importlib.metadata import version

    v = version("mqt-ionshuttler")
    assert isinstance(v, str)
    assert len(v) > 0
