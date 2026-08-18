# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared fixtures for the Linear compiler contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

_GOLDEN_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "production_default_golden.json"
_SADD_FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "dd"


@pytest.fixture
def production_default_golden() -> dict[str, object]:
    """Load the deterministic production-default compilation fixture.

    Returns:
        The source input, compiler configuration, and normalized expected result.
    """
    data = json.loads(_GOLDEN_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return cast("dict[str, object]", data)


@pytest.fixture(params=sorted(_SADD_FIXTURE_DIRECTORY.glob("sadd_*.json")), ids=lambda path: path.stem)
def sadd_golden(request: pytest.FixtureRequest) -> dict[str, object]:
    """Load one deterministic source-derived SADD fixture.

    Returns:
        A base schedule, SADD configuration, and normalized expected observations.
    """
    path = cast("Path", request.param)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return cast("dict[str, object]", data)
