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
from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.linear.assignment import compute_gate_zone_assignment, zone_site_pairs
from mqt.ionshuttler.linear.config import PermutationConfig, SearchConfig
from mqt.ionshuttler.linear.cost import makespan_lower_bound
from mqt.ionshuttler.linear.permutation import compile_with_permutations
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

    The configuration type selects the scheduling strategy. ``SearchConfig``
    is the default. Time advancement remains an internal part of scheduling.
    """

    architecture: LinearArchitecture
    config: SearchConfig | PermutationConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        """Ensure the configuration and hardware action catalog agree.

        Raises:
            TypeError: If the configuration has the wrong type.
            ValueError: If the configuration enables an unsupported action.
        """
        if not isinstance(self.config, SearchConfig | PermutationConfig):
            msg = "config must be SearchConfig or PermutationConfig"
            raise TypeError(msg)
        action_types = (
            self.architecture.supported_action_types if self.config.action_types is None else self.config.action_types
        )
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
    ) -> LinearCompilationResult:
        """Compile a circuit from QASM text, a QASM file, or Qiskit.

        Args:
            circuit: Circuit to compile.
            initial_placement: Optional starting site for each circuit ion.

        Returns:
            The resulting schedule and completion status.
        """
        compilation_started_at = perf_counter()
        config = self.config
        action_types = self.architecture.supported_action_types if config.action_types is None else config.action_types
        parsed_circuit = parse_circuit(
            circuit,
            gate_types=tuple(action_type for action_type in action_types if issubclass(action_type, GateAction)),
        )
        initial_state = create_initial_state(
            parsed_circuit.num_ions,
            self.architecture,
            initial_positions=None if initial_placement is None else tuple(initial_placement),
        )
        preferred_zone_by_gate = (
            compute_gate_zone_assignment(parsed_circuit, self.architecture, config=config.pre_partition_config)
            if config.pre_partition
            else {}
        )

        if config.max_compile_time is not None:
            remaining_compile_time = max(
                0.0,
                config.max_compile_time - (perf_counter() - compilation_started_at),
            )
            config = replace(config, max_compile_time=remaining_compile_time)
        # Config type determines the called compiler method
        if isinstance(config, PermutationConfig):
            result = compile_with_permutations(
                initial_state,
                parsed_circuit,
                self.architecture,
                config,
                preferred_zone_by_gate=preferred_zone_by_gate,
            )
        else:
            result = search(
                initial_state,
                parsed_circuit,
                self.architecture,
                config,
                gate_zone=preferred_zone_by_gate,
                zone_site_pairs=(zone_site_pairs(self.architecture) if preferred_zone_by_gate else {}),
            )
        assert result.diagnostics is not None  # Both strategies report diagnostics.
        diagnostics = replace(
            result.diagnostics,
            preferred_gate_zones=tuple(preferred_zone_by_gate.items()),
            makespan_lower_bound=makespan_lower_bound(parsed_circuit, self.architecture),
        )
        return replace(
            result,
            diagnostics=diagnostics,
            wall_clock_s=perf_counter() - compilation_started_at,
        )


__all__ = ["LinearCompiler"]
