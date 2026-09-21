# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fine-grained processing-zone preferences for Linear schedule search."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import SingleQubitGate, TwoQubitGate
from mqt.ionshuttler.partitioning import GateInfo, compute_fine_grained_gate_partition

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.linear.actions import GateAction
    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.partitioning import FineGrainedTabuConfig


def compute_gate_zone_assignment(
    gate_order: Sequence[int],
    gates: Mapping[int, GateAction],
    architecture: Architecture,
    *,
    config: FineGrainedTabuConfig | None = None,
) -> dict[int, str]:
    """Assign each gate to a preferred processing zone.

    The assignment only biases schedule search; normal action validation still
    determines where a gate may execute. Architectures with one processing zone
    need no preference and return immediately without running the partitioner.

    Args:
        gate_order: Stable gate ids in circuit order.
        gates: Linear gate actions keyed by gate id.
        architecture: Hardware layout whose zones receive the gates.
        config: Optional fine-grained tabu-search settings.

    Returns:
        Preferred processing-zone names keyed by gate id, or an empty mapping
        for a single-zone architecture.
    """
    processing_zones = architecture.processing_zones or {}
    if len(processing_zones) < 2:
        return {}

    gate_info = {
        gate_id: GateInfo(qubits=_gate_qubits(gates[gate_id]), qasm=type(gates[gate_id]).__name__)
        for gate_id in gate_order
    }
    zone_names = tuple(processing_zones)
    midpoints = [(sites[0] + sites[-1]) / 2 for sites in processing_zones.values()]
    distances = [[abs(source - target) for target in midpoints] for source in midpoints]
    result = compute_fine_grained_gate_partition(
        gate_order,
        gate_info,
        zone_names,
        distances,
        config=config,
    )
    return result.gate_assignment


def zone_site_pairs(architecture: Architecture) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return the valid two-ion site pairs belonging to each processing zone."""
    return {
        zone_name: tuple(combinations(sites, 2)) for zone_name, sites in (architecture.processing_zones or {}).items()
    }


def _gate_qubits(gate: GateAction) -> tuple[int, ...]:
    if isinstance(gate, TwoQubitGate):
        return (gate.ion_a, gate.ion_b)
    if isinstance(gate, SingleQubitGate):
        return (gate.ion,)
    return ()


__all__ = ["compute_gate_zone_assignment", "zone_site_pairs"]
