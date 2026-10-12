# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Visualize Linear compilation results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

    from mqt.ionshuttler.core.gates import GlobalGate
    from mqt.ionshuttler.core.result import CompilationResult


class _LocatedState(Protocol):
    """Expose ion positions to a Linear visualizer."""

    positions: tuple[tuple[int, int], ...]


class _LinearArchitectureView(Protocol):
    """Expose the architecture fields used by the Linear visualizer."""

    num_sites: int
    processing_zones: Mapping[str, Sequence[int]]


_FIGURE_WIDTH_INCHES = 10.0
_AXIS_WIDTH_INCHES = 8.0
_ROW_HEIGHT_INCHES = 3.4
_MIN_TIMESTEP_WIDTH_INCHES = 0.3
_MAJOR_TICK_INTERVAL = 5
_MAX_TIMESTEPS_PER_ROW = max(1, int(_AXIS_WIDTH_INCHES / _MIN_TIMESTEP_WIDTH_INCHES))
_TIMESTEPS_PER_ROW = max(
    _MAJOR_TICK_INTERVAL,
    (_MAX_TIMESTEPS_PER_ROW // _MAJOR_TICK_INTERVAL) * _MAJOR_TICK_INTERVAL,
)


class LinearVisualizer:
    """Plot ion trajectories and gates for a Linear compilation result."""

    @staticmethod
    def visualize(result: CompilationResult) -> object:
        """Return a Matplotlib figure for a Linear compilation result.

        Args:
            result: Linear compilation result to visualize.

        Returns:
            A Matplotlib figure with ion trajectories and gate applications.
        """
        import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]

        from mqt.ionshuttler.core.gates import (  # ruff: ignore[import-outside-top-level]
            GlobalGate,
            SingleQubitGate,
            TwoQubitGate,
        )
        from mqt.ionshuttler.linear.actions import (  # ruff: ignore[import-outside-top-level]
            PhysicalSwap,
            Shuttle,
        )
        from mqt.ionshuttler.linear.timeline import build_timeline  # ruff: ignore[import-outside-top-level]

        architecture = cast("_LinearArchitectureView", result.architecture)
        initial_positions = dict(cast("_LocatedState", result.initial_state).positions)
        initial_time = result.initial_state.time
        final_time = result.end_time
        display_final_time = max(final_time, initial_time + 1)
        time_windows = _time_windows(initial_time, display_final_time)
        figure = plt.figure(
            figsize=(_FIGURE_WIDTH_INCHES, _ROW_HEIGHT_INCHES * len(time_windows)),
            layout="constrained",
        )
        if len(time_windows) == 1:
            axes = (figure.add_subplot(),)
        else:
            grid = figure.add_gridspec(len(time_windows), _TIMESTEPS_PER_ROW)
            axes = tuple(
                figure.add_subplot(grid[row, : window_end - window_start])
                for row, (window_start, window_end) in enumerate(time_windows)
            )

        trajectory_times = {ion: [initial_time] for ion in initial_positions}
        trajectory_sites = {ion: [site] for ion, site in initial_positions.items()}
        for item in result.schedule.scheduled_actions:
            action = item.action
            if isinstance(action, Shuttle):
                _append_linear_movement(
                    trajectory_times,
                    trajectory_sites,
                    action.ion,
                    item.start_time,
                    item.end_time,
                    action.dst,
                )
            elif isinstance(action, PhysicalSwap):
                _append_linear_movement(
                    trajectory_times,
                    trajectory_sites,
                    action.ion_a,
                    item.start_time,
                    item.end_time,
                    action.pos_b,
                )
                _append_linear_movement(
                    trajectory_times,
                    trajectory_sites,
                    action.ion_b,
                    item.start_time,
                    item.end_time,
                    action.pos_a,
                )

        for ion in initial_positions:
            _append_hold(trajectory_times[ion], trajectory_sites[ion], display_final_time)
        timeline = build_timeline(result.schedule, result.architecture)
        ion_colors: dict[int, object] = {}
        for row, (axis, (window_start, window_end)) in enumerate(zip(axes, time_windows, strict=True)):
            show_legend = row == 0
            for zone_index, zone_sites in enumerate(architecture.processing_zones.values()):
                axis.axhspan(
                    min(zone_sites) - 0.45,
                    max(zone_sites) + 0.45,
                    alpha=0.1,
                    label="PZ" if show_legend and zone_index == 0 else "_nolegend_",
                    zorder=0,
                )
            for ion in sorted(initial_positions):
                color = ion_colors.get(ion)
                (line,) = axis.plot(
                    trajectory_times[ion],
                    trajectory_sites[ion],
                    color=color,
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    label=f"ion {ion}" if show_legend else "_nolegend_",
                    zorder=2,
                )
                ion_colors.setdefault(ion, line.get_color())

            for item in result.schedule.scheduled_actions:
                action = item.action
                starts_in_window = window_start <= item.start_time < window_end or (
                    row == len(time_windows) - 1 and item.start_time == window_end
                )
                overlaps_window = item.end_time > window_start and item.start_time < window_end
                if isinstance(action, SingleQubitGate):
                    site = timeline.ion_position(action.ion, item.start_time)
                    color = ion_colors[action.ion]
                    if overlaps_window:
                        _draw_gate_interval(axis, item.start_time, item.end_time, site, color)
                    if starts_in_window:
                        axis.scatter(
                            item.start_time,
                            site,
                            marker="D",
                            s=50,
                            facecolor="white",
                            edgecolor=color,
                            linewidth=1.5,
                            zorder=4,
                        )
                        _label_gate(axis, item.start_time, site, type(action).__name__)
                elif isinstance(action, TwoQubitGate):
                    sites = tuple(timeline.ion_position(ion, item.start_time) for ion in action.ions)
                    if overlaps_window:
                        for ion, site in zip(action.ions, sites, strict=True):
                            _draw_gate_interval(axis, item.start_time, item.end_time, site, ion_colors[ion])
                    if starts_in_window:
                        axis.plot(
                            [item.start_time, item.start_time],
                            [min(sites), max(sites)],
                            color="0.2",
                            linewidth=1.5,
                            zorder=3,
                        )
                        axis.scatter(
                            [item.start_time] * len(sites),
                            sites,
                            marker="D",
                            s=50,
                            facecolor="white",
                            edgecolor="0.2",
                            linewidth=1.5,
                            zorder=4,
                        )
                        _label_two_qubit_gate(
                            axis,
                            item.start_time,
                            sum(sites) / len(sites),
                            type(action).__name__,
                        )
                elif isinstance(action, GlobalGate):
                    if overlaps_window:
                        axis.axvspan(item.start_time, item.end_time, color="0.2", alpha=0.08, zorder=1)
                    if starts_in_window:
                        axis.axvline(item.start_time, color="0.2", linestyle="--", linewidth=1.2, zorder=1)
                        _label_gate(
                            axis,
                            item.start_time,
                            architecture.num_sites - 0.5,
                            _global_gate_label(action, tuple(sorted(initial_positions))),
                        )

            axis.set(
                xlim=(window_start, window_end),
                ylim=(-0.5, architecture.num_sites - 0.5),
                yticks=range(architecture.num_sites),
                ylabel="site",
            )
            first_major_tick = (
                (window_start + _MAJOR_TICK_INTERVAL - 1) // _MAJOR_TICK_INTERVAL
            ) * _MAJOR_TICK_INTERVAL
            axis.set_xticks(range(first_major_tick, window_end + 1, _MAJOR_TICK_INTERVAL))
            axis.grid(axis="both", color="0.9", linewidth=0.7, zorder=-1)
        axes[0].set_title("Linear ion trajectories")
        axes[-1].set_xlabel("timestep")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        return figure


def _global_gate_label(gate: GlobalGate, all_ions: tuple[int, ...]) -> str:
    """Name a global rotation and, unless it targets every ion, its ions.

    Returns:
        The label for the plotted gate.
    """
    name = gate.gate_name.capitalize()
    if gate.ions == all_ions:
        return name
    return f"{name} {list(gate.ions)}"


def _time_windows(initial_time: int, final_time: int) -> tuple[tuple[int, int], ...]:
    """Split a long schedule into readable consecutive plot rows.

    Returns:
        Ordered inclusive-start, exclusive-end plot windows.
    """
    return tuple(
        (window_start, min(window_start + _TIMESTEPS_PER_ROW, final_time))
        for window_start in range(initial_time, final_time, _TIMESTEPS_PER_ROW)
    )


def _append_linear_movement(
    trajectory_times: dict[int, list[int]],
    trajectory_sites: dict[int, list[int]],
    ion: int,
    start_time: int,
    end_time: int,
    destination: int,
) -> None:
    """Append one hold and movement segment to an ion trajectory."""
    _append_hold(trajectory_times[ion], trajectory_sites[ion], start_time)
    trajectory_times[ion].append(end_time)
    trajectory_sites[ion].append(destination)


def _append_hold(times: list[int], sites: list[int], until: int) -> None:
    """Extend a trajectory horizontally to one time boundary."""
    if times[-1] < until:
        times.append(until)
        sites.append(sites[-1])


def _draw_gate_interval(axis: Axes, start_time: int, end_time: int, site: int, color: object) -> None:
    """Emphasize the interval during which a gate occupies an ion."""
    if end_time > start_time:
        axis.plot(
            [start_time, end_time],
            [site, site],
            color=color,
            linewidth=6,
            alpha=0.3,
            solid_capstyle="butt",
            zorder=3,
        )


def _label_gate(axis: Axes, timestep: int, site: float, label: str) -> None:
    """Place a compact gate label above its trajectory marker."""
    axis.annotate(
        label,
        (timestep, site),
        xytext=(3, 6),
        textcoords="offset points",
        fontsize="small",
        rotation=35,
        ha="left",
        va="bottom",
        zorder=5,
    )


def _label_two_qubit_gate(axis: Axes, timestep: int, site: float, label: str) -> None:
    """Center a two-ion gate label on its connector."""
    axis.annotate(
        label,
        (timestep, site),
        fontsize="small",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        zorder=5,
    )


__all__ = ["LinearVisualizer"]
