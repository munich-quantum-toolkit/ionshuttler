# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Ion placement, hardware availability, and circuit progress."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture


@dataclass(frozen=True)
class State:
    """Describe the current state of a Linear compilation.

    Ion positions, availability times, and the current time describe the
    hardware. Completed and running gate identifiers describe progress through
    the circuit. Physical actions change only the hardware-related information;
    the compiler updates circuit progress separately.
    """

    positions: tuple[tuple[int, int], ...]
    completed_gates: frozenset[int]
    in_progress_gates: tuple[tuple[int, int], ...]
    ions_busy_until: tuple[tuple[int, int], ...]
    pzs_busy_until: tuple[tuple[str, int], ...]
    time: int

    def __post_init__(self) -> None:
        """Normalize tuple fields so equality and hashing are order-independent."""
        object.__setattr__(self, "positions", tuple(sorted(self.positions)))
        object.__setattr__(self, "in_progress_gates", tuple(sorted(self.in_progress_gates)))
        object.__setattr__(self, "ions_busy_until", tuple(sorted(self.ions_busy_until)))
        object.__setattr__(self, "pzs_busy_until", tuple(sorted(self.pzs_busy_until)))


def to_dict(state: State) -> dict[int, int]:
    """Return the ion-to-site position mapping."""
    return dict(state.positions)


def ions_busy_dict(state: State) -> dict[int, int]:
    """Return the ion-to-free-time mapping."""
    return dict(state.ions_busy_until)


def pzs_busy_dict(state: State) -> dict[str, int]:
    """Return the processing-zone-to-free-time mapping."""
    return dict(state.pzs_busy_until)


def in_progress_dict(state: State) -> dict[int, int]:
    """Return the gate-to-finish-time mapping."""
    return dict(state.in_progress_gates)


def has_pending_timed_work(state: State) -> bool:
    """Return whether advancing time could let scheduled work finish.

    The compiler uses this to avoid exploring pointless idle time. Waiting is
    still a valid schedule operation when no work is running.
    """
    return bool(
        state.in_progress_gates
        or any(free_time > state.time for _, free_time in state.ions_busy_until)
        or any(free_time > state.time for _, free_time in state.pzs_busy_until)
    )


def to_site_occupancy(state: State, num_sites: int) -> list[int | None]:
    """Convert ion positions to a site-indexed occupancy list.

    Args:
        state: State to convert.
        num_sites: Length of the architecture.

    Returns:
        Ion identifiers indexed by site, with ``None`` for empty sites.

    Raises:
        ValueError: If an ion position is outside the requested site range.
    """
    site_occupancy: list[int | None] = [None] * num_sites
    for ion, position in state.positions:
        if not 0 <= position < num_sites:
            msg = f"ion {ion} occupies invalid site {position}; expected a site within [0, {num_sites - 1}]"
            raise ValueError(msg)
        site_occupancy[position] = ion
    return site_occupancy


def to_metadata_dict(state: State, num_sites: int) -> dict[str, object]:
    """Return the JSON-compatible initial-state metadata."""
    return {"site_occupancy": to_site_occupancy(state, num_sites)}


def create_initial_state(
    num_ions: int,
    architecture: Architecture,
    initial_positions: list[int] | tuple[int, ...] | None = None,
) -> State:
    """Create a starting state with centered or explicitly placed ions.

    Args:
        num_ions: Number of logical ions to place.
        architecture: Architecture providing sites and processing zones.
        initial_positions: Optional site for each ion, ordered by ion identifier.
            Defaults to ``None``, which uses the centered-placement formula.

    Returns:
        The initial state at time zero, before any gates have started.

    Raises:
        ValueError: If the ion count or explicit positions are invalid.
    """
    if num_ions < 0:
        msg = "num_ions must be >= 0"
        raise ValueError(msg)

    num_sites = architecture.num_sites
    if num_sites < num_ions:
        msg = "num_sites must be >= num_ions"
        raise ValueError(msg)

    if initial_positions is not None:
        if len(initial_positions) != num_ions:
            msg = "initial_positions must have length num_ions"
            raise ValueError(msg)
        if len(set(initial_positions)) != len(initial_positions):
            msg = "initial_positions must not contain duplicate sites"
            raise ValueError(msg)
        if any(position < 0 or position >= num_sites for position in initial_positions):
            msg = "initial_positions must be within [0, num_sites - 1]"
            raise ValueError(msg)
        positions = tuple(enumerate(initial_positions))
    else:
        start_position = (num_sites - num_ions) // 2
        positions = tuple((ion, start_position + ion) for ion in range(num_ions))

    return State(
        positions=positions,
        completed_gates=frozenset(),
        in_progress_gates=(),
        ions_busy_until=tuple((ion, 0) for ion in range(num_ions)),
        pzs_busy_until=architecture.initial_pzs_busy_until(),
        time=0,
    )


def normalize_initial_state(state: State, architecture: Architecture) -> State:
    """Check a starting state against the hardware model and fill in missing zones.

    Args:
        state: Candidate initial compiler state.
        architecture: Hardware model against which to validate the state.

    Returns:
        A state containing an availability time for every processing zone.

    Raises:
        ValueError: If occupancy or processing-zone references are invalid.
    """
    occupied_sites: set[int] = set()
    for _, site in state.positions:
        if not 0 <= site < architecture.num_sites:
            msg = f"site must be within [0, {architecture.num_sites - 1}]"
            raise ValueError(msg)
        if site in occupied_sites:
            msg = f"initial state contains duplicate occupancy at site {site}"
            raise ValueError(msg)
        occupied_sites.add(site)

    known_pz_busy = dict(state.pzs_busy_until)
    architecture_zones = dict(architecture.processing_zones or {})
    unknown_zones = set(known_pz_busy).difference(architecture_zones)
    if unknown_zones:
        unknown_zone = min(unknown_zones)
        msg = f"initial state references unknown processing zone '{unknown_zone}'"
        raise ValueError(msg)

    for zone_name, free_time in architecture.initial_pzs_busy_until():
        known_pz_busy.setdefault(zone_name, free_time)

    return replace(state, pzs_busy_until=tuple(sorted(known_pz_busy.items())))


__all__ = [
    "State",
    "create_initial_state",
    "has_pending_timed_work",
    "in_progress_dict",
    "ions_busy_dict",
    "normalize_initial_state",
    "pzs_busy_dict",
    "to_dict",
    "to_metadata_dict",
    "to_site_occupancy",
]
