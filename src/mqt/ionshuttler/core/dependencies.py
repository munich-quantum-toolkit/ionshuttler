# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared setting for the gate dependencies that constrain scheduling."""

from __future__ import annotations

from enum import StrEnum


class DependencyMode(StrEnum):
    """Select which circuit dependencies constrain scheduling.

    ``CONSERVATIVE`` keeps the dependencies of the parsed circuit: each gate
    follows the latest earlier gate on each of its ions and every barrier
    before it. ``COMMUTATION_AWARE`` derives the dependencies again from the
    gates and barriers, so that commuting gates can change order.
    """

    CONSERVATIVE = "conservative"
    COMMUTATION_AWARE = "commutation_aware"


__all__ = ["DependencyMode"]
