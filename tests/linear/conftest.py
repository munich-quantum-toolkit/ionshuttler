# Copyright (c) 2026 Munich Quantum Software Company GmbH
"""Shared fixtures for the Linear compiler contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

_GOLDEN_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "production_default_golden.json"


@pytest.fixture
def production_default_golden() -> dict[str, object]:
    """Load the deterministic production-default compilation fixture.

    Returns:
        The source input, compiler configuration, and normalized expected result.
    """
    data = json.loads(_GOLDEN_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return cast("dict[str, object]", data)
