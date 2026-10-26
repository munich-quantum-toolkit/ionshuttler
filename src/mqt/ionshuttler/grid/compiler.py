# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing entry point for Grid compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

from mqt.ionshuttler.circuit import parse_circuit, with_commutation_aware_dependencies
from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.grid.assignment import compute_gate_zone_assignment
from mqt.ionshuttler.grid.config import (
    GridCompilerConfig,
    GridCompilerStrategy,
    GridDependencyMode,
)
from mqt.ionshuttler.grid.routing import select_greedy_target
from mqt.ionshuttler.grid.scheduling import schedule_breadth_first, schedule_greedy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.circuit import Circuit, CircuitInput
    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.core.schedule import Schedule
    from mqt.ionshuttler.grid.architecture import GridArchitecture
    from mqt.ionshuttler.grid.result import GridCompilationResult, GridDiagnostics
    from mqt.ionshuttler.grid.state import GridMachineState


@dataclass(frozen=True)
class GridCompiler:
    """Compile supported circuits to a fixed Grid hardware model."""

    architecture: GridArchitecture
    config: GridCompilerConfig = field(default_factory=GridCompilerConfig)

    def compile(
        self,
        circuit: CircuitInput,
        *,
        initial_placement: Mapping[str, Sequence[int]] | None = None,
        target_placement: Mapping[str, Sequence[int]] | None = None,
        gate_to_processing_zone: Mapping[int, str] | None = None,
        pre_partition: bool = True,
    ) -> GridCompilationResult:
        """Compile a circuit with the configured scheduling strategy.

        Args:
            circuit: Circuit to compile.
            initial_placement: Optional segment-to-ion placement.
            target_placement: Optional final segment-to-ion placement after all
                gates finish. A placement that contains every initial ion is
                exact; omitted segments are empty. A placement that contains a
                subset constrains only those ions. Other ions and segments
                remain unconstrained.
            gate_to_processing_zone: Optional physical gate-to-processing-zone assignment.
                When set, this assignment replaces automatic pre-partitioning.
                Gates not in the mapping select a processing zone while scheduling.
            pre_partition: Assign gates to processing zones before scheduling.
                This is a no-op when fewer than two compatible, mutually
                reachable processing zones are available.

        Returns:
            A validated complete or partial compilation result.

        Raises:
            ValueError: If the target contains an ion outside the initial
                placement.
        """
        started = perf_counter()
        gate_types = tuple(
            action_type
            for action_type in self.architecture.supported_action_types
            if issubclass(action_type, GateAction)
        )
        parsed = parse_circuit(circuit, gate_types=gate_types)
        if self.config.dependency_mode is GridDependencyMode.COMMUTATION_AWARE:
            parsed = with_commutation_aware_dependencies(parsed)
        placement = (
            _default_placement(parsed.num_ions, self.architecture) if initial_placement is None else initial_placement
        )
        initial_state = self.architecture.initial_state(placement)
        _validate_circuit_ions(parsed, initial_state)
        requested_target = None if target_placement is None else self.architecture.initial_state(target_placement)
        if requested_target is not None and not set(requested_target.ions) <= set(initial_state.ions):
            msg = "target placement contains an ion outside the initial placement"
            raise ValueError(msg)
        target_state = (
            requested_target if requested_target is not None and requested_target.ions == initial_state.ions else None
        )
        partial_target_by_ion = (
            None
            if requested_target is None or target_state is not None
            else {ion: segment_id for segment_id, ions in requested_target.occupancy for ion in ions}
        )
        _validate_gate_capabilities(parsed, self.architecture)
        gate_to_segment_assignment = self._resolve_gate_assignment(
            parsed,
            gate_to_processing_zone,
            pre_partition=pre_partition,
        )
        if self.config.strategy is GridCompilerStrategy.GREEDY:
            _validate_initial_gate_reachability(
                parsed,
                self.architecture,
                initial_state,
                gate_to_segment_assignment,
                self.config.allowed_junction_crossings,
            )

        schedule, diagnostics = self._create_schedule(
            parsed,
            initial_state,
            gate_to_segment_assignment,
            target_state,
            partial_target_by_ion,
        )
        final_state = self.architecture.replay_schedule(schedule)
        target_reached = (
            final_state.occupancy == target_state.occupancy
            if target_state is not None
            else partial_target_by_ion is None
            or all(final_state.ion_segment(ion) == segment for ion, segment in partial_target_by_ion.items())
        )
        status = (
            CompilationStatus.SUCCESS
            if diagnostics.gates == len(parsed.gates) and target_reached
            else CompilationStatus.FAILED
        )
        result = CompilationResult(
            status=status,
            schedule=schedule,
            architecture=self.architecture,
            final_state=final_state,
            wall_clock_s=perf_counter() - started,
            diagnostics=diagnostics,
        )
        result.validate()
        return result

    def _create_schedule(
        self,
        circuit: Circuit,
        initial_state: GridMachineState,
        gate_to_segment_assignment: Mapping[int, str],
        target_state: GridMachineState | None,
        partial_target_by_ion: Mapping[int, str] | None,
    ) -> tuple[Schedule[Action, GridMachineState], GridDiagnostics]:
        """Create a schedule with the selected strategy.

        Returns:
            The schedule and its diagnostic counters.
        """
        if self.config.strategy is GridCompilerStrategy.GREEDY:
            return schedule_greedy(
                self.architecture,
                circuit,
                initial_state,
                self.config,
                gate_to_segment_assignment,
                target_state,
                partial_target_by_ion,
            )
        return schedule_breadth_first(
            self.architecture,
            circuit,
            initial_state,
            self.config,
            gate_to_segment_assignment,
            target_state,
            partial_target_by_ion,
        )

    def _resolve_gate_assignment(
        self,
        circuit: Circuit,
        explicit_assignment: Mapping[int, str] | None,
        *,
        pre_partition: bool,
    ) -> dict[int, str]:
        """Resolve physical gate assignments to processing-zone segments.

        Returns:
            Physical gate IDs mapped to segment IDs.

        Raises:
            TypeError: If an assignment key or value has the wrong type.
            ValueError: If an assignment is invalid.
        """
        if not isinstance(pre_partition, bool):
            msg = "pre_partition must be a boolean"
            raise TypeError(msg)
        if explicit_assignment is not None:
            assignment = dict(explicit_assignment)
        elif pre_partition:
            assignment = compute_gate_zone_assignment(
                circuit,
                self.architecture,
                self.config.allowed_junction_crossings,
                config=self.config.pre_partition_config,
            )
        else:
            assignment = {}
        if any(isinstance(gate_id, bool) or not isinstance(gate_id, int) for gate_id in assignment):
            msg = "gate assignment keys must be integer gate IDs"
            raise TypeError(msg)
        if any(not isinstance(zone_id, str) for zone_id in assignment.values()):
            msg = "gate assignment values must be processing-zone IDs"
            raise TypeError(msg)
        gates_by_id = {gate.gate_id: gate for gate in circuit.gates if gate.gate_id is not None}
        unknown_gate_ids = set(assignment) - set(gates_by_id)
        if unknown_gate_ids:
            msg = f"gate assignment refers to unknown gate IDs: {sorted(unknown_gate_ids)}"
            raise ValueError(msg)
        segment_assignment: dict[int, str] = {}
        for gate_id, zone_id in assignment.items():
            gate = gates_by_id[gate_id]
            if self.architecture.is_virtual_gate(gate):
                msg = f"virtual gate {gate_id} does not use a processing zone"
                raise ValueError(msg)
            try:
                zone = self.architecture.processing_zone(zone_id)
            except KeyError as error:
                msg = f"gate assignment refers to unknown processing zone {zone_id!r}"
                raise ValueError(msg) from error
            if not zone.supports(gate):
                msg = f"processing zone {zone_id!r} does not support gate {gate_id}"
                raise ValueError(msg)
            segment = self.architecture.processing_zone_segment(zone_id)
            if segment.capacity < len(gate.ions):
                msg = f"processing zone {zone_id!r} cannot accommodate gate {gate_id}"
                raise ValueError(msg)
            segment_assignment[gate_id] = segment.segment_id
        return segment_assignment


def _default_placement(num_ions: int, architecture: GridArchitecture) -> dict[str, tuple[int, ...]]:
    if num_ions > sum(segment.capacity for segment in architecture.segments):
        msg = "circuit has more ions than the Grid architecture capacity"
        raise ValueError(msg)
    remaining = iter(range(num_ions))
    placement: dict[str, tuple[int, ...]] = {}
    next_ion = next(remaining, None)
    for segment in architecture.segments:
        ions: list[int] = []
        while next_ion is not None and len(ions) < segment.capacity:
            ions.append(next_ion)
            next_ion = next(remaining, None)
        placement[segment.segment_id] = tuple(ions)
    return placement


def _validate_circuit_ions(circuit: Circuit, state: GridMachineState) -> None:
    if state.ions != tuple(range(circuit.num_ions)):
        msg = "initial placement must contain every circuit ion exactly once"
        raise ValueError(msg)


def _validate_gate_capabilities(circuit: Circuit, architecture: GridArchitecture) -> None:
    for gate in circuit.gates:
        if architecture.is_virtual_gate(gate):
            continue
        if not any(
            zone.supports(gate) and architecture.processing_zone_segment(zone.zone_id).capacity >= len(gate.ions)
            for zone in architecture.processing_zones
        ):
            msg = f"Grid architecture has no processing zone for {type(gate).__name__}"
            raise ValueError(msg)


def _validate_initial_gate_reachability(
    circuit: Circuit,
    architecture: GridArchitecture,
    state: GridMachineState,
    gate_to_segment_assignment: Mapping[int, str],
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> None:
    """Reject a Greedy input whose initial placement cannot reach a gate target."""
    for gate in circuit.gates:
        if architecture.is_virtual_gate(gate):
            continue
        target_segment_ids = (
            (gate_to_segment_assignment[gate.gate_id],)
            if gate.gate_id is not None and gate.gate_id in gate_to_segment_assignment
            else tuple(
                dict.fromkeys(
                    architecture.processing_zone_segment(zone.zone_id).segment_id
                    for zone in architecture.processing_zones
                    if zone.supports(gate)
                    and architecture.processing_zone_segment(zone.zone_id).capacity >= len(gate.ions)
                )
            )
        )
        select_greedy_target(architecture, state, gate, target_segment_ids, allowed_crossings)


__all__ = ["GridCompiler"]
