# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared interface of the compilers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from mqt.ionshuttler.circuit import Circuit
    from mqt.ionshuttler.core.result import Architecture, CompilationResult

# Grid placements map segments to ions; Linear placements list the site of each ion.
PlacementT_contra = TypeVar("PlacementT_contra", contravariant=True)


class Compiler(Protocol[PlacementT_contra]):
    """Compile one straight-line circuit between placements of one architecture.

    ``GridCompiler`` and ``LinearCompiler`` have this interface.
    :func:`~mqt.ionshuttler.dynamic.compiler.compile_program` accepts every
    compiler with this interface. A compiler must meet these conditions:

    - ``architecture.replay_schedule`` returns the machine state at the end of
      each result schedule.
    - The schedule contains each gate of the circuit once, with its gate ID.
      Each gate starts after the gates that it depends on end. These
      dependencies include at least the ones that
      :func:`~mqt.ionshuttler.circuit.with_commutation_aware_dependencies`
      derives from the gates and barriers of the circuit.
    - ``final_state.placement`` has the form that ``compile`` accepts.
    - With ``target_placement``, the schedule ends in exactly this placement,
      or the status is not ``SUCCESS``.
    - A circuit without gates is valid. Its schedule only moves the ions to
      ``target_placement``, if given.

    :meth:`~mqt.ionshuttler.dynamic.compiler.ProgramResult.validate` checks
    the schedules of a compiled program.
    """

    @property
    def architecture(self) -> Architecture:
        """The architecture that the compiler targets."""
        ...

    def compile(
        self,
        circuit: Circuit,
        *,
        initial_placement: PlacementT_contra | None = None,
        target_placement: PlacementT_contra | None = None,
    ) -> CompilationResult[Any, Any, Any, Any]:
        """Compile a circuit that starts in ``initial_placement`` and ends in ``target_placement``."""
        ...


__all__ = ["Compiler"]
