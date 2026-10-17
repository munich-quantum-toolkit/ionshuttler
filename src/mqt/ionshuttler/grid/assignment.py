# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Gate-to-processing-zone assignment for Grid compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.ionshuttler.partitioning import GateInfo, compute_fine_grained_gate_partition

from .routing import routing_distance

if TYPE_CHECKING:
    from mqt.ionshuttler.circuit import Circuit
    from mqt.ionshuttler.grid.architecture import GridArchitecture
    from mqt.ionshuttler.partitioning import FineGrainedTabuConfig


def compute_gate_zone_assignment(
    circuit: Circuit,
    architecture: GridArchitecture,
    allowed_crossings: frozenset[tuple[str, str]] | None = None,
    *,
    config: FineGrainedTabuConfig | None = None,
) -> dict[int, str]:
    """Assign each physical gate to a processing zone with the shared partitioner.

    Args:
        architecture: Grid hardware that supplies processing zones and distances.
        circuit: Parsed circuit to partition.
        allowed_crossings: Optional directed routing policy.
        config: Optional fine-grained tabu-search settings.

    Returns:
        Physical gate IDs mapped to processing-zone IDs.

    """
    physical_gates = tuple(gate for gate in circuit.gates if not architecture.is_virtual_gate(gate))
    if not physical_gates:
        return {}
    largest_gate_arity = max(len(gate.ions) for gate in physical_gates)
    zones = tuple(
        zone
        for zone in architecture.processing_zones
        if architecture.processing_zone_segment(zone.zone_id).capacity >= largest_gate_arity
        and all(zone.supports(gate) for gate in physical_gates)
    )
    if len(zones) < 2:
        return {}
    zone_ids = tuple(zone.zone_id for zone in zones)
    segment_ids = tuple(architecture.processing_zone_segment(zone_id).segment_id for zone_id in zone_ids)
    distance_matrix: list[list[float]] = []
    for source_segment_id in segment_ids:
        row: list[float] = []
        for target_segment_id in segment_ids:
            distance = routing_distance(architecture, source_segment_id, target_segment_id, allowed_crossings)
            if distance is None:
                return {}
            row.append(float(distance))
        distance_matrix.append(row)
    gate_info = {
        gate.gate_id: GateInfo(gate.ions, gate.circuit_name or type(gate).__name__)
        for gate in physical_gates
        if gate.gate_id is not None
    }
    result = compute_fine_grained_gate_partition(
        list(gate_info),
        gate_info,
        zone_ids,
        distance_matrix,
        config=config,
    )
    return result.gate_assignment


__all__ = ["compute_gate_zone_assignment"]
