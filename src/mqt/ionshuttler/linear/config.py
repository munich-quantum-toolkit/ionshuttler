# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Settings for Linear hardware timing and schedule search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SINGLE_QUBIT_GATE_NAMES = frozenset({"rx", "ry", "rz"})
TWO_QUBIT_GATE_NAMES = frozenset({"rxx", "ryy", "rzz"})
GATE_NAMES = SINGLE_QUBIT_GATE_NAMES | TWO_QUBIT_GATE_NAMES
HeuristicMode = Literal["quality", "zero"]


@dataclass(frozen=True)
class TransportTiming:
    """Configure how many timesteps ion transport operations take."""

    shuttle: int = 1
    swap: int = 3

    def __post_init__(self) -> None:
        """Ensure every transport operation takes at least one timestep."""
        _require_integer_at_least(self.shuttle, "shuttle duration", minimum=1)
        _require_integer_at_least(self.swap, "swap duration", minimum=1)


@dataclass(frozen=True)
class GateTiming:
    """Configure gate durations and virtual single-ion rotations.

    Rx and Ry use physical controls by default. Rz defaults to a zero-duration
    virtual frame update, as is common on trapped-ion hardware.
    """

    rx: int = 1
    ry: int = 1
    rz: int = 0
    rxx: int = 2
    ryy: int = 2
    rzz: int = 2
    virtual_single_qubit_gates: frozenset[str] = field(default_factory=lambda: frozenset({"rz"}))

    def __post_init__(self) -> None:
        """Ensure the configured timings describe usable gate implementations.

        Raises:
            ValueError: If a gate name or duration is invalid.
        """
        virtual_gates = _normalize_virtual_gates(self.virtual_single_qubit_gates)
        object.__setattr__(self, "virtual_single_qubit_gates", virtual_gates)

        for gate_name in SINGLE_QUBIT_GATE_NAMES:
            duration = self.duration_for(gate_name)
            _require_integer_at_least(duration, f"{gate_name} duration", minimum=0)
            if gate_name in virtual_gates and duration != 0:
                msg = f"duration for virtual gate {gate_name!r} must be 0"
                raise ValueError(msg)
        for gate_name in TWO_QUBIT_GATE_NAMES:
            _require_integer_at_least(
                self.duration_for(gate_name),
                f"{gate_name} duration",
                minimum=1,
            )

    def duration_for(self, gate_name: str) -> int:
        """Return how many timesteps a gate takes.

        Raises:
            TypeError: If the stored duration is not an integer.
            ValueError: If the hardware does not offer ``gate_name``.
        """
        normalized_name = gate_name.lower()
        if normalized_name not in GATE_NAMES:
            msg = f"unsupported gate name {gate_name!r}"
            raise ValueError(msg)
        value = getattr(self, normalized_name)
        if not isinstance(value, int):  # Defensive narrowing; construction validates the public fields.
            msg = f"duration for gate {gate_name!r} must be an integer"
            raise TypeError(msg)
        return value

    def is_virtual(self, gate_name: str) -> bool:
        """Return whether a single-ion rotation is implemented virtually.

        Raises:
            ValueError: If ``gate_name`` is not a supported single-ion rotation.
        """
        normalized_name = gate_name.lower()
        if normalized_name not in SINGLE_QUBIT_GATE_NAMES:
            msg = f"virtuality applies only to single-qubit gates, not {gate_name!r}"
            raise ValueError(msg)
        return normalized_name in self.virtual_single_qubit_gates

    @property
    def gate_durations(self) -> dict[str, int]:
        """Duration of each supported gate, keyed by its lowercase name."""
        return {gate_name: self.duration_for(gate_name) for gate_name in sorted(GATE_NAMES)}


@dataclass(frozen=True)
class HardwareTiming:
    """Describe the timing and gate implementations offered by the hardware."""

    transport: TransportTiming = field(default_factory=TransportTiming)
    gates: GateTiming = field(default_factory=GateTiming)


@dataclass(frozen=True)
class SearchConfig:
    """Configure how the compiler searches for a schedule.

    A finite ``horizon`` plans a few gates at a time and commits
    ``committed_gates`` before planning again. Set ``horizon`` to ``None`` to
    search the complete circuit at once. The default ``quality`` heuristic
    finds useful schedules faster but may overestimate their remaining time.
    The ``zero`` heuristic is admissible and supports exact search when all
    other limits and shortcuts are also disabled.
    """

    horizon: int | None = 3
    committed_gates: int = 2
    iterative_diving_search: bool = True
    informed_action_prioritization: bool = False
    num_solutions: int = 1
    max_frontier_size: int | None = 1000
    max_compile_time: float | None = 1800.0
    use_dependencies: bool = True
    heuristic_mode: HeuristicMode = "quality"

    def __post_init__(self) -> None:
        """Ensure all search limits are meaningful and mutually consistent.

        Raises:
            TypeError: If a Boolean option is not Boolean.
            ValueError: If a numeric bound or horizon relationship is invalid.
        """
        if self.horizon is not None:
            _require_integer_at_least(self.horizon, "horizon", minimum=1)
        _require_integer_at_least(self.committed_gates, "committed_gates", minimum=1)
        if self.horizon is not None and self.committed_gates > self.horizon:
            msg = "committed_gates must be <= horizon"
            raise ValueError(msg)
        _require_integer_at_least(self.num_solutions, "num_solutions", minimum=1)
        if self.max_frontier_size is not None:
            _require_integer_at_least(self.max_frontier_size, "max_frontier_size", minimum=1)
        if self.max_compile_time is not None and (
            isinstance(self.max_compile_time, bool)
            or not isinstance(self.max_compile_time, int | float)
            or self.max_compile_time < 0
        ):
            msg = "max_compile_time must be None or a nonnegative number"
            raise ValueError(msg)
        for name in (
            "iterative_diving_search",
            "informed_action_prioritization",
            "use_dependencies",
        ):
            if not isinstance(getattr(self, name), bool):
                msg = f"{name} must be a boolean"
                raise TypeError(msg)
        if not isinstance(self.heuristic_mode, str):
            msg = "heuristic_mode must be a string"
            raise TypeError(msg)
        if self.heuristic_mode not in {"quality", "zero"}:
            msg = "heuristic_mode must be 'quality' or 'zero'"
            raise ValueError(msg)


@dataclass(frozen=True)
class LinearCompilerConfig:
    """Group the hardware timing and schedule-search settings."""

    hardware_timing: HardwareTiming = field(default_factory=HardwareTiming)
    search: SearchConfig = field(default_factory=SearchConfig)


def _normalize_virtual_gates(gate_names: object) -> frozenset[str]:
    if not isinstance(gate_names, frozenset | set | tuple | list):
        msg = "virtual_single_qubit_gates must be a collection of gate names"
        raise TypeError(msg)
    normalized_names: set[str] = set()
    for gate_name in gate_names:
        if not isinstance(gate_name, str):
            msg = "virtual_single_qubit_gates must contain only strings"
            raise TypeError(msg)
        normalized_names.add(gate_name.lower())
    normalized = frozenset(normalized_names)
    unknown = normalized.difference(SINGLE_QUBIT_GATE_NAMES)
    if unknown:
        msg = f"unknown virtual single-qubit gates: {sorted(unknown)}"
        raise ValueError(msg)
    return normalized


def _require_integer_at_least(value: object, name: str, *, minimum: int) -> None:
    """Require a non-Boolean integer at or above a minimum.

    Raises:
        ValueError: If the value is Boolean, noninteger, or below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{name} must be an integer >= {minimum}"
        raise ValueError(msg)


__all__ = [
    "GATE_NAMES",
    "SINGLE_QUBIT_GATE_NAMES",
    "TWO_QUBIT_GATE_NAMES",
    "GateTiming",
    "HardwareTiming",
    "HeuristicMode",
    "LinearCompilerConfig",
    "SearchConfig",
    "TransportTiming",
]
