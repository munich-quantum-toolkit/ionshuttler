# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Checks for groups of transports that start together in a Linear schedule."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mqt.ionshuttler.linear.actions import PhysicalSwap, Shuttle, TransportAction
from mqt.ionshuttler.linear.state import State, to_dict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.linear.architecture import LinearArchitecture


def is_transport_layer_valid(
    state: State,
    actions: Sequence[TransportAction],
    architecture: LinearArchitecture,
) -> bool:
    """Return whether several transports can safely happen together.

    An ion may enter a site that another ion leaves in the same group, provided
    every ion ends at a different valid site. Two shuttles may not exchange
    their sites; a physical swap describes that operation.

    Raises:
        TypeError: If an action is not a Linear shuttle or physical swap.
    """
    if not actions:
        return True

    final_positions = to_dict(state)
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
            msg = _unsupported_transport_message(action)
            raise TypeError(msg)
        if set(updated_positions) & acted_ions:
            return False
        acted_ions.update(updated_positions)
        final_positions.update(updated_positions)

    if not all(_is_valid_in_layer(state, action, acted_ions, architecture) for action in actions):
        return False

    return all(0 <= position < architecture.num_sites for position in final_positions.values()) and len(
        set(final_positions.values())
    ) == len(final_positions)


def is_transport_valid_in_layer(
    state: State,
    action: TransportAction,
    actions: Sequence[TransportAction],
    architecture: LinearArchitecture,
) -> bool:
    """Return whether one transport of a layer can start in a state.

    The other ions that act in the layer do not block the transport, so it may
    enter a site that one of these ions leaves. Use
    :func:`is_transport_layer_valid` to check the layer as a whole.

    Args:
        state: State in which the transport starts.
        action: Transport to check.
        actions: Every transport of the layer, including ``action``.
        architecture: Hardware model that defines the transport rules.

    Returns:
        Whether the transport can start.
    """
    acted_ions = {ion for layer_action in actions for ion in _transport_ions(layer_action)}
    return _is_valid_in_layer(state, action, acted_ions, architecture)


def _is_valid_in_layer(
    state: State,
    action: TransportAction,
    acted_ions: set[int],
    architecture: LinearArchitecture,
) -> bool:
    own_ions = _transport_ions(action)
    layer_positions = tuple(
        (ion, position) for ion, position in state.positions if ion in own_ions or ion not in acted_ions
    )
    return architecture.is_action_valid(replace(state, positions=layer_positions), action)


def _transport_ions(action: TransportAction) -> frozenset[int]:
    if isinstance(action, Shuttle):
        return frozenset({action.ion})
    if isinstance(action, PhysicalSwap):
        return frozenset({action.ion_a, action.ion_b})
    msg = _unsupported_transport_message(action)
    raise TypeError(msg)


def _unsupported_transport_message(action: TransportAction) -> str:
    return f"Linear transport layers contain only shuttles and physical swaps, not {type(action).__name__}"


__all__ = ["is_transport_layer_valid", "is_transport_valid_in_layer"]
