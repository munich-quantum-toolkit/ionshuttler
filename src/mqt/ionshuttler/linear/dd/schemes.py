# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reusable pulse sequences for dynamical decoupling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import isclose, pi

from mqt.ionshuttler.linear.actions import GateSpec

logger = logging.getLogger(__name__)
ARBITRARY_X_SCHEME_FAMILY = "arbitrary_x"
_XY4_AXES = ("X", "Y", "X", "Y")
_XY8_AXES = ("X", "Y", "X", "Y", "Y", "X", "Y", "X")
_XY16_AXES = (
    "X",
    "Y",
    "X",
    "Y",
    "Y",
    "X",
    "Y",
    "X",
    "-X",
    "-Y",
    "-X",
    "-Y",
    "-Y",
    "-X",
    "-Y",
    "-X",
)


@dataclass(frozen=True)
class DDScheme:
    """Define pulse axes and their relative positions in an idle window."""

    name: str
    relative_gate_times: tuple[float, ...]
    gate_specs: tuple[GateSpec, ...]
    allow_rounding: bool = False

    def __post_init__(self) -> None:
        """Validate that every pulse has exactly one relative time.

        Raises:
            ValueError: If pulse times and specifications are empty or inconsistent.
        """
        if len(self.relative_gate_times) != len(self.gate_specs):
            msg = "relative_gate_times and gate_specs must have the same length"
            raise ValueError(msg)
        if not self.gate_specs:
            msg = "gate_specs must not be empty"
            raise ValueError(msg)

    @property
    def num_pulses(self) -> int:
        """The number of pulses in this sequence."""
        return len(self.gate_specs)

    def _boundary_offsets(self, window: tuple[int, int]) -> tuple[float, ...]:
        """Convert relative pulse times to offsets within a schedule window.

        Returns:
            The unrounded boundary offset for every pulse.
        """
        t_start, t_end = window
        length = t_end - t_start
        return tuple(relative_time * length for relative_time in self.relative_gate_times)

    def gate_timesteps(self, window: tuple[int, int]) -> tuple[int, ...]:
        """Map relative pulse times to the nearest schedule boundaries.

        Offsets are rounded to nearest with ties-to-even before adding the window
        start. A pulse at boundary ``t`` acts before phase accumulates over
        ``[t, t + 1)``.

        Returns:
            The schedule boundary for every pulse.
        """
        t_start, _t_end = window
        timesteps = tuple(_quantized_boundary(t_start, offset) for offset in self._boundary_offsets(window))
        logger.debug(
            "computed raw DD gate timesteps scheme=%s window=%s timesteps=%s",
            self.name,
            window,
            timesteps,
        )
        return timesteps

    def resolved_gate_timesteps(self, window: tuple[int, int]) -> tuple[int, ...] | None:
        """Return distinct integral pulse boundaries, or ``None`` if infeasible.

        With ``allow_rounding`` unset, a pulse time must already coincide with a
        schedule boundary within tolerance. With ``allow_rounding`` set, a pulse
        time is rounded to the nearest boundary and rejected only if that falls
        outside the window.
        """
        t_start, t_end = window
        resolved: list[int] = []
        for offset in self._boundary_offsets(window):
            timestep = _quantized_boundary(t_start, offset)
            nearest_offset = timestep - t_start
            if self.allow_rounding:
                if not t_start <= timestep <= t_end:
                    return None
            elif not isclose(offset, nearest_offset, rel_tol=0.0, abs_tol=1e-9):
                return None
            resolved.append(timestep)
        if len(set(resolved)) != len(resolved):
            return None
        return tuple(resolved)

    def clamped_pulses(self, window: tuple[int, int]) -> tuple[tuple[int, GateSpec], ...]:
        """Return rounded, window-clamped, deduplicated pulse boundaries with their gate specs.

        Unlike :meth:`resolved_gate_timesteps`, a pulse time outside the window is
        clamped to the nearer boundary instead of making the sequence infeasible.

        Returns:
            The ordered, deduplicated ``(timestep, gate_spec)`` pairs to insert.
        """
        t_start, t_end = window
        seen: set[int] = set()
        pulses: list[tuple[int, GateSpec]] = []
        for offset, spec in zip(self._boundary_offsets(window), self.gate_specs, strict=True):
            timestep = min(max(_quantized_boundary(t_start, offset), t_start), t_end)
            if timestep in seen:
                continue
            seen.add(timestep)
            pulses.append((timestep, spec))
        return tuple(pulses)


def _quantized_boundary(window_start: int, relative_offset: float) -> int:
    """Quantize a relative offset with ties-to-even and add its window start.

    Returns:
        The translation-invariant nearest schedule boundary.
    """
    return window_start + round(relative_offset)


def make_cpmg_scheme(num_pulses: int) -> DDScheme:
    """Construct an evenly spaced Carr-Purcell-Meiboom-Gill sequence.

    Returns:
        The requested CPMG sequence.

    Raises:
        ValueError: If ``num_pulses`` is not an even integer of at least two.
    """
    if num_pulses < 2 or num_pulses % 2 != 0:
        msg = "num_pulses must be an even integer >= 2"
        raise ValueError(msg)
    return DDScheme(
        name=f"cpmg{num_pulses}",
        relative_gate_times=_even_relative_gate_times(num_pulses),
        gate_specs=tuple(GateSpec("Rx", theta=pi) for _ in range(num_pulses)),
    )


def make_arbitrary_x_scheme(num_pulses: int) -> DDScheme:
    """Construct an evenly spaced sequence of X pulses.

    Returns:
        The requested X-pulse sequence.

    Raises:
        ValueError: If ``num_pulses`` is less than one.
    """
    if num_pulses < 1:
        msg = "num_pulses must be >= 1"
        raise ValueError(msg)
    return DDScheme(
        name=f"{ARBITRARY_X_SCHEME_FAMILY}_{num_pulses}",
        relative_gate_times=_even_relative_gate_times(num_pulses),
        gate_specs=tuple(GateSpec("Rx", theta=pi) for _ in range(num_pulses)),
    )


def make_xy4_scheme(k: int = 1) -> DDScheme:
    """Construct an XY4 sequence repeated ``k`` times.

    Returns:
        The requested XY4 sequence.
    """
    return _make_xy_scheme("xy4", _XY4_AXES, k)


def make_xy8_scheme(k: int = 1) -> DDScheme:
    """Construct an XY8 sequence repeated ``k`` times.

    Returns:
        The requested XY8 sequence.
    """
    return _make_xy_scheme("xy8", _XY8_AXES, k)


def make_xy16_scheme(k: int = 1) -> DDScheme:
    """Construct an XY16 sequence repeated ``k`` times.

    Returns:
        The requested XY16 sequence.
    """
    return _make_xy_scheme("xy16", _XY16_AXES, k)


def make_cdd_scheme(level: int) -> DDScheme:
    """Construct a recursively concatenated sequence up to level three.

    Returns:
        The requested concatenated sequence.

    Raises:
        ValueError: If ``level`` lies outside the supported range.
    """
    if level < 1 or level > 3:
        msg = "level must be between 1 and 3"
        raise ValueError(msg)
    axes: tuple[str, ...] = ()
    for _ in range(level):
        axes = (*axes, "X", *axes, "Y", *axes, "X", *axes, "Y")
    return DDScheme(
        name=f"cdd{level}",
        relative_gate_times=_even_relative_gate_times(len(axes)),
        gate_specs=tuple(_gate_spec_for_axis(axis) for axis in axes),
    )


def _make_xy_scheme(name: str, axes: tuple[str, ...], k: int) -> DDScheme:
    if k < 1:
        msg = "k must be >= 1"
        raise ValueError(msg)
    repeated_axes = axes * k
    return DDScheme(
        name=name if k == 1 else f"{name}^{k}",
        relative_gate_times=_even_relative_gate_times(len(repeated_axes)),
        gate_specs=tuple(_gate_spec_for_axis(axis) for axis in repeated_axes),
    )


def _gate_spec_for_axis(axis: str) -> GateSpec:
    sign = -1 if axis.startswith("-") else 1
    gate_name = axis.removeprefix("-")
    if gate_name == "X":
        return GateSpec("Rx", theta=sign * pi)
    if gate_name == "Y":
        return GateSpec("Ry", theta=sign * pi)
    msg = f"unsupported DD axis: {axis}"
    raise ValueError(msg)


def _even_relative_gate_times(num_pulses: int) -> tuple[float, ...]:
    return tuple(index / num_pulses for index in range(1, num_pulses + 1))


HAHN_ECHO = DDScheme(
    "hahn",
    (0.5, 1.0),
    (GateSpec("Rx", theta=pi), GateSpec("Rx", theta=pi)),
)
MIDPOINT_ONLY_HAHN = DDScheme(
    "midpoint_only_hahn",
    (0.5,),
    (GateSpec("Rx", theta=pi),),
    allow_rounding=True,
)
CPMG_2 = make_cpmg_scheme(2)
CPMG_4 = make_cpmg_scheme(4)
CPMG_6 = make_cpmg_scheme(6)
CPMG_8 = make_cpmg_scheme(8)
XY4 = make_xy4_scheme()
XY8 = make_xy8_scheme()
XY16 = make_xy16_scheme()
CDD_1 = make_cdd_scheme(1)
CDD_2 = make_cdd_scheme(2)
CDD_3 = make_cdd_scheme(3)

_DD_SCHEME_REGISTRY = {
    scheme.name: scheme
    for scheme in (
        HAHN_ECHO,
        MIDPOINT_ONLY_HAHN,
        CPMG_2,
        CPMG_4,
        CPMG_6,
        CPMG_8,
        XY4,
        XY8,
        XY16,
        CDD_1,
        CDD_2,
        CDD_3,
    )
}


def available_dd_schemes() -> dict[str, DDScheme]:
    """Return a copy of the registered named schemes."""
    return dict(_DD_SCHEME_REGISTRY)


def get_dd_scheme(name: str) -> DDScheme:
    """Resolve a registered scheme by name.

    Returns:
        The registered scheme.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    try:
        return _DD_SCHEME_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(_DD_SCHEME_REGISTRY))
        msg = f"unknown DD scheme {name!r}, expected one of: {available}"
        raise ValueError(msg) from error


__all__ = [
    "ARBITRARY_X_SCHEME_FAMILY",
    "CDD_1",
    "CDD_2",
    "CDD_3",
    "CPMG_2",
    "CPMG_4",
    "CPMG_6",
    "CPMG_8",
    "HAHN_ECHO",
    "MIDPOINT_ONLY_HAHN",
    "XY4",
    "XY8",
    "XY16",
    "DDScheme",
    "available_dd_schemes",
    "get_dd_scheme",
    "make_arbitrary_x_scheme",
    "make_cdd_scheme",
    "make_cpmg_scheme",
    "make_xy4_scheme",
    "make_xy8_scheme",
    "make_xy16_scheme",
]
