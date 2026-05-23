from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateInfo:
    """Immutable metadata for a parsed gate."""

    qubits: tuple[int, ...]
    qasm: str


@dataclass(slots=True)
class ParsedCircuit:
    """Circuit representation with stable gate ids and metadata."""

    sequence: list[int]
    gate_info: dict[int, GateInfo]

    @property
    def qubit_sequence(self) -> list[tuple[int, ...]]:
        """Return the legacy qubit-tuple view of the circuit."""

        return [self.gate_info[gate_id].qubits for gate_id in self.sequence]
