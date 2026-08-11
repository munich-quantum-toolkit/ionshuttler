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
    acted_ions: set[int] = set()
    shuttle_edges: set[tuple[int, int]] = set()
    built_in_acted_ions = {
        ion
        for action in actions
        for ion in (
            (action.ion,)
            if isinstance(action, Shuttle)
            else (action.ion_a, action.ion_b)
            if isinstance(action, PhysicalSwap)
            else ()
        )
    }

    for action in actions:
        validation_state = state
        if isinstance(action, Shuttle):
            layer_positions = tuple(
                (ion, position)
                for ion, position in state.positions
                if ion == action.ion or ion not in built_in_acted_ions
            )
            validation_state = replace(state, positions=layer_positions)
        if not action.is_valid(validation_state, architecture):
            return False

        if isinstance(action, Shuttle):
            if action.ion in acted_ions:
                return False
            if (action.dst, action.src) in shuttle_edges:
                return False
            shuttle_edges.add((action.src, action.dst))
            acted_ions.add(action.ion)
            final_positions[action.ion] = action.dst
        elif isinstance(action, PhysicalSwap):
            if action.ion_a == action.ion_b or action.ion_a in acted_ions or action.ion_b in acted_ions:
                return False
            acted_ions.update({action.ion_a, action.ion_b})
            final_positions[action.ion_a] = action.pos_b
            final_positions[action.ion_b] = action.pos_a
        else:
            updated_positions = to_dict(action.apply(state, architecture))
            if set(updated_positions) != set(positions):
                return False
            changed_ions = {ion for ion, position in positions.items() if updated_positions[ion] != position}
            if changed_ions & acted_ions:
                return False
            acted_ions.update(changed_ions)
            final_positions.update({ion: updated_positions[ion] for ion in changed_ions})

    return all(0 <= position < architecture.num_sites for position in final_positions.values()) and len(
        set(final_positions.values())
    ) == len(final_positions)


__all__ = ["is_action_valid", "is_adjacent", "is_transport_layer_valid"]
