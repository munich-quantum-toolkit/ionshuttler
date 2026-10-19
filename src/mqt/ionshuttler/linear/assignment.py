# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Gate-to-processing-zone assignment for Linear compilation."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.gates import SingleQubitGate, TwoQubitGate
from mqt.ionshuttler.partitioning import GateInfo, compute_fine_grained_gate_partition

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.circuit import Circuit
    from mqt.ionshuttler.core.gates import GateAction
    from mqt.ionshuttler.linear.architecture import LinearArchitecture
    from mqt.ionshuttler.partitioning import FineGrainedTabuConfig


def compute_gate_zone_assignment(
    circuit: Circuit,
    architecture: LinearArchitecture,
    *,
    active_gate_ids: Sequence[int] | None = None,
    config: FineGrainedTabuConfig | None = None,
) -> dict[int, str]:
    """Assign each gate to a preferred processing zone.

    The assignment guides the selected compiler strategy. Normal action
    validation still determines where a gate may run. Architectures with one
    processing zone need no preference and return without running the
    partitioner.

    Args:
        circuit: Circuit whose gates receive preferences.
        architecture: Hardware layout whose zones receive the gates.
        active_gate_ids: Circuit gates to assign. Defaults to all gates.
        config: Optional fine-grained tabu-search settings.

    Returns:
        Preferred processing-zone names keyed by gate id, or an empty mapping
        for a single-zone architecture.
    """
    processing_zones = architecture.processing_zones or {}
    if len(processing_zones) < 2:
        return {}

    candidate_gate_ids = circuit.gate_ids if active_gate_ids is None else active_gate_ids
    gate_info_by_id = {
        gate_id: GateInfo(
            qubits=_physical_gate_qubits(circuit.gates[gate_id], architecture),
            qasm=type(circuit.gates[gate_id]).__name__,
        )
        for gate_id in candidate_gate_ids
    }
    partitioned_gate_ids = tuple(gate_id for gate_id in candidate_gate_ids if gate_info_by_id[gate_id].qubits)
    if not partitioned_gate_ids:
        return {}
    zone_names = tuple(processing_zones)
    zone_midpoints = [(sites[0] + sites[-1]) / 2 for sites in processing_zones.values()]
    zone_distances = [[abs(source - target) for target in zone_midpoints] for source in zone_midpoints]
    partition = compute_fine_grained_gate_partition(
        partitioned_gate_ids,
        gate_info_by_id,
        zone_names,
        zone_distances,
        processing_zone_capacities={name: len(sites) for name, sites in processing_zones.items()},
        config=config,
    )
    return partition.gate_assignment


def zone_site_pairs(architecture: LinearArchitecture) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return the valid two-ion site pairs belonging to each processing zone."""
    return {
        zone_name: tuple(combinations(sites, 2)) for zone_name, sites in (architecture.processing_zones or {}).items()
    }


def _physical_gate_qubits(
    gate: GateAction,
    architecture: LinearArchitecture,
) -> tuple[int, ...]:
    if isinstance(gate, TwoQubitGate):
        return (gate.ion_a, gate.ion_b)
    if isinstance(gate, SingleQubitGate) and not architecture.is_virtual_gate(gate):
        return (gate.ion,)
    return ()


__all__ = ["compute_gate_zone_assignment", "zone_site_pairs"]
