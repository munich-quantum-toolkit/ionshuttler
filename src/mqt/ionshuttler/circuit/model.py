# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Immutable circuit data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.ionshuttler.core.gates import GateAction


@dataclass(frozen=True)
class Circuit:
    """Store gate occurrences and their direct dependencies.

    This type is an internal boundary between circuit parsing and compiler
    methods. Public compiler entry points continue to accept QASM, paths, and
    Qiskit circuits.
    """

    num_ions: int
    gates: tuple[GateAction, ...]
    predecessors: tuple[frozenset[int], ...]
    barriers: tuple[tuple[frozenset[int], ...], ...] = ()

    def __post_init__(self) -> None:
        """Validate gate and dependency identities.

        Raises:
            TypeError: If the ion count has the wrong type.
            ValueError: If the ion count or dependency structure is invalid.
        """
        gates = tuple(self.gates)
        predecessors = tuple(frozenset(values) for values in self.predecessors)
        barriers = tuple(tuple(frozenset(targets) for targets in gate_barriers) for gate_barriers in self.barriers)
        if isinstance(self.num_ions, bool) or not isinstance(self.num_ions, int):
            msg = "num_ions must be an integer"
            raise TypeError(msg)
        if self.num_ions < 1:
            msg = "num_ions must be positive"
            raise ValueError(msg)
        if len(predecessors) != len(gates):
            msg = "predecessors must contain one entry for each gate"
            raise ValueError(msg)
        if barriers and len(barriers) != len(gates) + 1:
            msg = "barriers must contain one entry before each gate and one trailing entry"
            raise ValueError(msg)
        if not barriers:
            barriers = tuple(() for _gate_id in range(len(gates) + 1))
        barrier_ions = (ion for gate_barriers in barriers for targets in gate_barriers for ion in targets)
        if any(ion < 0 or ion >= self.num_ions for ion in barrier_ions):
            msg = "barriers must reference circuit ions"
            raise ValueError(msg)
        if tuple(gate.gate_id for gate in gates) != tuple(range(len(gates))):
            msg = "gates must have stable circuit-order gate_id values"
            raise ValueError(msg)
        for gate_id, gate_predecessors in enumerate(predecessors):
            if any(predecessor < 0 or predecessor >= gate_id for predecessor in gate_predecessors):
                msg = "each predecessor must identify an earlier gate"
                raise ValueError(msg)
        object.__setattr__(self, "gates", gates)
        object.__setattr__(self, "predecessors", predecessors)
        object.__setattr__(self, "barriers", barriers)

    @property
    def gate_ids(self) -> range:
        """Gate occurrence identifiers in circuit order."""
        return range(len(self.gates))


__all__ = ["Circuit"]
