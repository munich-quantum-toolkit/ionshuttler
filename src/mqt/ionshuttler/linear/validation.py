# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Checks for actions and groups of transports in a Linear schedule."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import (
    Action,
    PhysicalSwap,
    Shuttle,
    TransportAction,
    is_adjacent,
)
from mqt.ionshuttler.linear.state import State, ions_busy_dict, to_dict

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.architecture import Architecture


def is_action_valid(state: State, action: Action, architecture: Architecture) -> bool:
    """Return whether an action can start in the current state.

    This asks the action to check its own requirements. Use
    :func:`is_transport_layer_valid` when several transports start together.
    """
    return action.is_valid(state, architecture)


def is_transport_layer_valid(
    state: State,
    actions: tuple[TransportAction, ...],
    architecture: Architecture,
) -> bool:
    """Return whether several transports can safely happen together.

    An ion may enter a site that another ion leaves in the same group, provided
    every ion ends at a different valid site.
    """
    if not actions:
        return True

    positions = to_dict(state)
    ions_busy = ions_busy_dict(state)
    current_time = state.time
    final_positions = dict(positions)
    acted_ions: set[int] = set()

    for action in actions:
        if isinstance(action, Shuttle):
            if action.ion in acted_ions:
                return False
            current_position = positions.get(action.ion)
            if (
                current_position != action.src
                or not 0 <= action.dst < architecture.num_sites
                or not is_adjacent(action.src, action.dst)
                or ions_busy.get(action.ion, current_time + 1) > current_time
            ):
                return False
            acted_ions.add(action.ion)
            final_positions[action.ion] = action.dst
        elif isinstance(action, PhysicalSwap):
            if action.ion_a == action.ion_b or action.ion_a in acted_ions or action.ion_b in acted_ions:
                return False
            pos_a = positions.get(action.ion_a)
            pos_b = positions.get(action.ion_b)
            if (
                pos_a != action.pos_a
                or pos_b != action.pos_b
                or ions_busy.get(action.ion_a, current_time + 1) > current_time
                or ions_busy.get(action.ion_b, current_time + 1) > current_time
                or not is_adjacent(action.pos_a, action.pos_b)
            ):
                return False
            acted_ions.update({action.ion_a, action.ion_b})
            final_positions[action.ion_a] = action.pos_b
            final_positions[action.ion_b] = action.pos_a
        else:
            return False

    return len(set(final_positions.values())) == len(final_positions)


__all__ = ["is_action_valid", "is_adjacent", "is_transport_layer_valid"]
