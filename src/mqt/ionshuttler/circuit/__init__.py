# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Internal circuit data used by compiler frontends."""

from mqt.ionshuttler.circuit.model import Circuit
from mqt.ionshuttler.circuit.parser import CircuitInput, parse_circuit

__all__ = ["Circuit", "CircuitInput", "parse_circuit"]
