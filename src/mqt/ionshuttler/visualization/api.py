# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public visualization contracts and dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mqt.ionshuttler.core.result import CompilationResult


class Visualizer(Protocol):
    """Create a visual view of a compilation result."""

    def visualize(self, result: CompilationResult, /) -> object:
        """Return a visual view without showing or saving it.

        Args:
            result: Compilation result to visualize.

        Returns:
            A visualizer-owned figure or visualization object.
        """
        ...


def visualize(result: CompilationResult) -> object:
    """Return a visual view of a compilation result.

    The built-in Linear visualizer shows ion trajectories and gate applications.
    Matplotlib is loaded only when that view is drawn. Use a concrete visualizer
    directly to select it explicitly.

    Args:
        result: Compilation result to visualize.

    Returns:
        A visualizer-owned figure or visualization object.

    Raises:
        ValueError: If no built-in visualizer supports the result architecture.
    """
    selected_visualizer = _default_visualizer(result)
    if selected_visualizer is None:
        msg = f"no built-in visualizer supports {type(result.architecture).__name__}"
        raise ValueError(msg)
    return selected_visualizer.visualize(result)


def _default_visualizer(result: CompilationResult) -> Visualizer | None:
    """Select the built-in visualizer for a result architecture.

    Returns:
        The matching visualizer, or ``None`` if the architecture is unsupported.
    """
    from mqt.ionshuttler.linear.architecture import LinearArchitecture  # ruff: ignore[import-outside-top-level]
    from mqt.ionshuttler.visualization.linear import LinearVisualizer  # ruff: ignore[import-outside-top-level]

    if isinstance(result.architecture, LinearArchitecture):
        return LinearVisualizer()
    return None


__all__ = ["Visualizer", "visualize"]
