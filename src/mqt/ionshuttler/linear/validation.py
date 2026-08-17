# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Checks for actions and groups of transports in a Linear schedule."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import (
    Action,
    PhysicalSwap,
    Shuttle,
    TransportAction,
    is_adjacent,
)
from mqt.ionshuttler.linear.state import State, to_dict

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
    final_positions = dict(positions)
    action_positions: list[dict[int, int]] = []
    acted_ions: set[int] = set()
    shuttle_edges: set[tuple[int, int]] = set()

    for action in actions:
        if isinstance(action, Shuttle):
            if (action.dst, action.src) in shuttle_edges:
                return False
            shuttle_edges.add((action.src, action.dst))
            updated_positions = {action.ion: action.dst}
        elif isinstance(action, PhysicalSwap):
            if action.ion_a == action.ion_b:
                return False
            updated_positions = {action.ion_a: action.pos_b, action.ion_b: action.pos_a}
        else:
            if not action.is_valid(state, architecture):
                return False
            updated_positions = to_dict(action.apply(state, architecture))
            if set(updated_positions) != set(positions):
                return False
            updated_positions = {
                ion: updated_positions[ion] for ion, position in positions.items() if updated_positions[ion] != position
            }
        if set(updated_positions) & acted_ions:
            return False
        acted_ions.update(updated_positions)
        action_positions.append(updated_positions)

    for action in actions:
        validation_state = state
        if isinstance(action, Shuttle):
            layer_positions = tuple(
                (ion, position) for ion, position in state.positions if ion == action.ion or ion not in acted_ions
            )
            validation_state = replace(state, positions=layer_positions)
        if isinstance(action, Shuttle | PhysicalSwap) and not action.is_valid(validation_state, architecture):
            return False

    for updated_positions in action_positions:
        final_positions.update(updated_positions)

    return all(0 <= position < architecture.num_sites for position in final_positions.values()) and len(
        set(final_positions.values())
    ) == len(final_positions)


__all__ = ["is_action_valid", "is_adjacent", "is_transport_layer_valid"]
