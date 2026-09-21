# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.ionshuttler.partitioning import GateInfo

GateRef = int | tuple[int, ...]


@dataclass(slots=True)
class ParsedCircuit:
    """Circuit representation with stable gate ids and metadata."""

    sequence: list[int]
    gate_info: dict[int, GateInfo]

    @property
    def qubit_sequence(self) -> list[tuple[int, ...]]:
        """Return the legacy qubit-tuple view of the circuit."""

        return [self.gate_info[gate_id].qubits for gate_id in self.sequence]
