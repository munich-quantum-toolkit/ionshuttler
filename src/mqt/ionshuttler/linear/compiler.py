# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing entry point for Linear shuttling compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.config import LinearCompilerConfig
from mqt.ionshuttler.linear.parser import parse_circuit
from mqt.ionshuttler.linear.search import search
from mqt.ionshuttler.linear.state import create_initial_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.linear.architecture import Architecture
    from mqt.ionshuttler.linear.parser import CircuitInput
    from mqt.ionshuttler.linear.result import CompilationResult


@dataclass(frozen=True)
class LinearCompiler:
    """Compile supported circuits to a fixed Linear hardware model."""

    architecture: Architecture
    config: LinearCompilerConfig = field(default_factory=LinearCompilerConfig)

    def compile(
        self,
        circuit: CircuitInput,
        *,
        initial_positions: Sequence[int] | None = None,
    ) -> CompilationResult:
        """Compile a circuit from QASM text, a QASM file, or Qiskit.

        Args:
            circuit: Circuit to compile.
            initial_positions: Optional starting site for each circuit qubit.

        Returns:
            The resulting schedule and completion status.
        """
        num_qubits, gate_list, predecessors, _ = parse_circuit(
            circuit,
            use_dependencies=self.config.search.use_dependencies,
            gate_timing=self.config.hardware_timing.gates,
        )
        initial_state = create_initial_state(
            num_qubits,
            self.architecture,
            initial_positions=None if initial_positions is None else tuple(initial_positions),
        )
        gate_order = list(range(len(gate_list)))
        gates = dict(zip(gate_order, gate_list, strict=True))

        return search(
            initial_state,
            gate_order,
            gates,
            self.architecture,
            predecessors,
            self.config,
        )


__all__ = ["LinearCompiler"]
