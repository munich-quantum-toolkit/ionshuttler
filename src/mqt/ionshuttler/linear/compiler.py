# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing entry point for Linear shuttling compilation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import TYPE_CHECKING

from mqt.ionshuttler.circuit import parse_circuit
from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.linear.config import LinearCompilerConfig
from mqt.ionshuttler.linear.partition_bias import compute_gate_zone_assignment, zone_site_pairs
from mqt.ionshuttler.linear.search import search
from mqt.ionshuttler.linear.state import create_initial_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.circuit import CircuitInput
    from mqt.ionshuttler.linear.architecture import LinearArchitecture
    from mqt.ionshuttler.linear.result import LinearCompilationResult


@dataclass(frozen=True)
class LinearCompiler:
    """Compile supported circuits to a fixed Linear hardware model.

    ``action_types`` selects the architecture-supported operations available to
    this compilation. It defaults to the architecture's full catalog. Time
    advancement remains an internal part of scheduling.
    """

    architecture: LinearArchitecture
    config: LinearCompilerConfig = field(default_factory=LinearCompilerConfig)
    action_types: tuple[type[Action], ...] | None = None

    def __post_init__(self) -> None:
        """Ensure the hardware action catalog contains action classes.

        Raises:
            TypeError: If an entry is not an ``Action`` subclass.
            ValueError: If an entry is repeated or unsupported by the architecture.
        """
        action_types = (
            self.architecture.supported_action_types if self.action_types is None else tuple(self.action_types)
        )
        object.__setattr__(self, "action_types", action_types)
        for action_type in action_types:
            if not isinstance(action_type, type) or not issubclass(action_type, Action):
                msg = "action_types must contain Action subclasses"
                raise TypeError(msg)
        if len(set(action_types)) != len(action_types):
            msg = "action_types must not contain duplicates"
            raise ValueError(msg)
        unsupported = [
            action_type.__name__ for action_type in action_types if not self.architecture.supports(action_type)
        ]
        if unsupported:
            msg = f"compiler action types are not supported by the architecture: {', '.join(unsupported)}"
            raise ValueError(msg)

    def compile(
        self,
        circuit: CircuitInput,
        *,
        initial_placement: Sequence[int] | None = None,
        pre_partition: bool = False,
    ) -> LinearCompilationResult:
        """Compile a circuit from QASM text, a QASM file, or Qiskit.

        Args:
            circuit: Circuit to compile.
            initial_placement: Optional starting site for each circuit ion.
            pre_partition: Bias multi-zone search toward a fine-grained gate partition.
                Partitioning consumes the compile time budget but is not interrupted
                when the budget expires.

        Returns:
            The resulting schedule and completion status.

        """
        action_types = self.action_types
        assert action_types is not None  # Normalized during initialization.
        parsed_circuit = parse_circuit(
            circuit,
            gate_types=tuple(action_type for action_type in action_types if issubclass(action_type, GateAction)),
        )
        initial_state = create_initial_state(
            parsed_circuit.num_ions,
            self.architecture,
            initial_positions=None if initial_placement is None else tuple(initial_placement),
        )
        gate_zone: dict[int, str] = {}
        preferred_zone_site_pairs: dict[str, tuple[tuple[int, int], ...]] = {}
        started = None
        if pre_partition and len(self.architecture.processing_zones or {}) >= 2:
            started = perf_counter()
            gate_zone = compute_gate_zone_assignment(
                parsed_circuit,
                self.architecture,
                config=self.config.search.pre_partition_config,
            )
            preferred_zone_site_pairs = zone_site_pairs(self.architecture)

        preparation_time = 0.0 if started is None else perf_counter() - started
        config = self.config
        if started is not None and config.search.max_compile_time is not None:
            config = replace(
                config,
                search=replace(
                    config.search, max_compile_time=max(0.0, config.search.max_compile_time - preparation_time)
                ),
            )
        result = search(
            initial_state,
            parsed_circuit,
            self.architecture,
            config,
            action_types=action_types,
            gate_zone=gate_zone,
            zone_site_pairs=preferred_zone_site_pairs,
        )
        if result.diagnostics is not None and gate_zone:
            result = replace(
                result,
                diagnostics=replace(
                    result.diagnostics,
                    preferred_gate_zones=tuple(gate_zone.items()),
                ),
            )
        return result if started is None else replace(result, wall_clock_s=perf_counter() - started)


__all__ = ["LinearCompiler"]
