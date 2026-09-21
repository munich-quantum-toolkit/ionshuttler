# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit gate metadata shared by partitioning clients."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateInfo:
    """Immutable metadata for a parsed gate."""

    qubits: tuple[int, ...]
    qasm: str


__all__ = ["GateInfo"]
