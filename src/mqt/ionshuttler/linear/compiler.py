# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing entry point for Linear shuttling compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES, Action, GateAction
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
    """Compile supported circuits to a fixed Linear hardware model.

    ``action_types`` lists every operation exposed by the hardware. Time
    advancement remains an internal part of scheduling.
    """

    architecture: Architecture
    config: LinearCompilerConfig = field(default_factory=LinearCompilerConfig)
    action_types: tuple[type[Action], ...] = DEFAULT_ACTION_TYPES

    def __post_init__(self) -> None:
        """Ensure the hardware action catalog contains action classes.

        Raises:
            TypeError: If an entry is not an ``Action`` subclass.
            ValueError: If two entries have the same serialized name.
        """
        for action_type in self.action_types:
            if not isinstance(action_type, type) or not issubclass(action_type, Action):
                msg = "action_types must contain Action subclasses"
                raise TypeError(msg)
        serialized_names = [action_type.__name__ for action_type in self.action_types]
        if len(set(serialized_names)) != len(serialized_names):
            msg = "action_types must have unique class names"
            raise ValueError(msg)

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
            gate_types=tuple(action_type for action_type in self.action_types if issubclass(action_type, GateAction)),
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
            action_types=self.action_types,
        )


__all__ = ["LinearCompiler"]
