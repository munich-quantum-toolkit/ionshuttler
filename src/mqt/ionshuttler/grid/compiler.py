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
        gate_to_processing_zone: Mapping[int, str] | None = None,
        pre_partition: bool = True,
    ) -> GridCompilationResult:
        """Compile a circuit with the configured scheduling strategy.

        Args:
            circuit: Circuit to compile.
            initial_placement: Optional segment-to-ion placement.
            gate_to_processing_zone: Optional physical gate-to-processing-zone assignment.
                When set, this assignment replaces automatic pre-partitioning.
                Gates not in the mapping select a processing zone while scheduling.
            pre_partition: Assign gates to processing zones before scheduling.
                This is a no-op when fewer than two compatible, mutually
                reachable processing zones are available.

        Returns:
            A validated complete or partial compilation result.
        """
        started = perf_counter()
        gate_types = tuple(
            action_type
            for action_type in self.architecture.supported_action_types
            if issubclass(action_type, GateAction)
        )
        parsed = parse_circuit(circuit, gate_types=gate_types)
        if self.config.dependency_mode is GridDependencyMode.COMMUTATION_AWARE:
            parsed = with_commutation_aware_dependencies(parsed, circuit)
        placement = (
            _default_placement(parsed.num_ions, self.architecture) if initial_placement is None else initial_placement
        )
        initial_state = self.architecture.initial_state(placement)
        _validate_circuit_ions(parsed, initial_state)
        _validate_gate_capabilities(parsed, self.architecture)
        gate_to_segment_assignment = self._resolve_gate_assignment(
            parsed,
            gate_to_processing_zone,
            pre_partition=pre_partition,
        )

        schedule, diagnostics = self._create_schedule(parsed, initial_state, gate_to_segment_assignment)
        status = CompilationStatus.SUCCESS if diagnostics.gates == len(parsed.gates) else CompilationStatus.FAILED
        result = CompilationResult(
            status=status,
            schedule=schedule,
            architecture=self.architecture,
            final_state=self.architecture.replay_schedule(schedule),
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
            )
        return schedule_breadth_first(
            self.architecture,
            circuit,
            initial_state,
            self.config,
            gate_to_segment_assignment,
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
            segment_assignment[gate_id] = self.architecture.processing_zone_segment(zone_id).segment_id
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
        if not any(zone.supports(gate) for zone in architecture.processing_zones):
            msg = f"Grid architecture has no processing zone for {type(gate).__name__}"
            raise ValueError(msg)


__all__ = ["GridCompiler"]
