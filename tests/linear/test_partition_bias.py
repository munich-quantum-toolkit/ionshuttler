# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear processing-zone heuristic preferences."""

from __future__ import annotations

import pytest

from mqt.ionshuttler.linear import partition_bias
from mqt.ionshuttler.linear.actions import Rzz
from mqt.ionshuttler.linear.architecture import Architecture


def test_single_zone_assignment_does_not_call_partitioner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the no-op assignment before invoking tabu search."""

    def unexpected_partitioner(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the partitioner must not run for one processing zone")

    monkeypatch.setattr(partition_bias, "compute_fine_grained_gate_partition", unexpected_partitioner)
    architecture = Architecture(num_sites=4, processing_zones={"only": [1, 2]})
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=0.5)}

    assert partition_bias.compute_gate_zone_assignment([0], gates, architecture) == {}


def test_zone_site_pairs_are_scoped_to_each_processing_zone() -> None:
    """Exclude cross-zone pairs from a zone-specific heuristic target."""
    architecture = Architecture(
        num_sites=8,
        processing_zones={"left": [1, 2, 3], "right": [6, 7]},
    )

    assert partition_bias.zone_site_pairs(architecture) == {
        "left": ((1, 2), (1, 3), (2, 3)),
        "right": ((6, 7),),
    }
