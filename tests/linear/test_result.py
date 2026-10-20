# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Linear schedules, compilation results, and their persistence."""

from __future__ import annotations

import itertools
from dataclasses import replace
from inspect import isabstract
from math import isnan, pi
from typing import TYPE_CHECKING, Any, cast

import pytest

import mqt.ionshuttler.visualization as visualization_module
from mqt.ionshuttler import visualize
from mqt.ionshuttler.core.gates import GlobalGate, Rx, Rz, Rzz
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.linear import (
    DEFAULT_ACTION_TYPES,
    load_result,
    load_schedule,
    result_from_dict,
    result_from_json,
    schedule_from_dict,
    schedule_from_json,
)
from mqt.ionshuttler.linear.actions import PhysicalSwap, Shuttle
from mqt.ionshuttler.linear.architecture import LinearArchitecture
from mqt.ionshuttler.linear.replay import replay_schedule
from mqt.ionshuttler.linear.result import GateGroup, LinearDiagnostics, PermutationDiagnostics, SearchDiagnostics
from mqt.ionshuttler.linear.schedule import LinearMachineState, schedule_from_path
from mqt.ionshuttler.linear.state import AdvanceTime, create_initial_state
from mqt.ionshuttler.visualization import LinearVisualizer, Visualizer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from matplotlib.figure import Figure
    from matplotlib.text import Annotation

    from mqt.ionshuttler.linear.state import SearchTransition


def _program(path: Sequence[SearchTransition] | None = None) -> Schedule:
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    return schedule_from_path(
        path or [AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )


@pytest.mark.parametrize(
    ("build", "error", "message"),
    [
        pytest.param(
            lambda: SearchDiagnostics(score=cast("int", object()), explored_nodes=0),
            TypeError,
            "score must be an integer",
            id="score-type",
        ),
        pytest.param(
            lambda: SearchDiagnostics(score=-1, explored_nodes=0),
            ValueError,
            "score must be non-negative",
            id="score-negative",
        ),
        pytest.param(
            lambda: SearchDiagnostics(
                score=0,
                explored_nodes=0,
                preferred_gate_zones=((cast("int", object()), "pz"),),
            ),
            TypeError,
            "preferred gate IDs must be integers",
            id="gate-id-type",
        ),
        pytest.param(
            lambda: SearchDiagnostics(score=0, explored_nodes=0, preferred_gate_zones=((-1, "pz"),)),
            ValueError,
            "preferred gate IDs must be non-negative",
            id="gate-id-negative",
        ),
        pytest.param(
            lambda: SearchDiagnostics(
                score=0,
                explored_nodes=0,
                preferred_gate_zones=((0, cast("str", 1)),),
            ),
            TypeError,
            "processing-zone IDs must be strings",
            id="zone-type",
        ),
        pytest.param(
            lambda: SearchDiagnostics(score=0, explored_nodes=0, preferred_gate_zones=((0, ""),)),
            ValueError,
            "processing-zone IDs must be non-empty",
            id="zone-empty",
        ),
        pytest.param(
            lambda: SearchDiagnostics(
                score=0,
                explored_nodes=0,
                preferred_gate_zones=((0, "left"), (0, "right")),
            ),
            ValueError,
            "preferred gate IDs must be unique",
            id="duplicate-gate-id",
        ),
    ],
)
def test_linear_diagnostics_reject_invalid_values(
    build: Callable[[], object],
    error: type[Exception],
    message: str,
) -> None:
    """Protect the public validation contract for compiler diagnostics."""
    with pytest.raises(error, match=message):
        build()


def test_linear_diagnostics_define_strategy_specific_types() -> None:
    """Keep shared diagnostics abstract and expose one type per strategy."""
    assert isabstract(LinearDiagnostics)
    assert issubclass(SearchDiagnostics, LinearDiagnostics)
    assert issubclass(PermutationDiagnostics, LinearDiagnostics)


@pytest.mark.parametrize(
    ("assignments", "message"),
    [
        ([0, "pz"], "entries must be"),
        ([[True, "pz"]], "preferred gate IDs must be integers"),
        ([[0, 3]], "processing-zone IDs must be strings"),
    ],
)
def test_linear_diagnostics_reject_malformed_json(assignments: object, message: str) -> None:
    """Reject malformed gate-zone assignments at the JSON boundary."""
    with pytest.raises(ValueError, match=message):
        LinearDiagnostics.from_dict({
            "score": 0,
            "explored_nodes": 0,
            "preferred_gate_zones": assignments,
            "makespan_lower_bound": 0,
            "permutation": None,
        })


def _gate_group(**changes: object) -> GateGroup:
    values: dict[str, object] = {
        "gate_ids": (0, 1),
        "zone_by_ion": ((0, "pz"),),
        "target_site_of_ion": (1, 0),
        "swaps": 1,
        "shuttles": 2,
        "start_time": 0,
        "end_time": 5,
        **changes,
    }
    return GateGroup(**cast("dict[str, Any]", values))


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param({"gate_ids": ()}, ValueError, "at least one gate", id="no-gates"),
        pytest.param({"gate_ids": (0, 0)}, ValueError, "gate IDs must be unique", id="repeated-gate"),
        pytest.param(
            {"zone_by_ion": ((0, "pz"), (0, "pz"))},
            ValueError,
            "assigned ions must be unique",
            id="repeated-ion",
        ),
        pytest.param(
            {"zone_by_ion": ((0, 1),)},
            TypeError,
            "processing-zone IDs must be strings",
            id="numeric-zone",
        ),
        pytest.param(
            {"zone_by_ion": ((0, ""),)},
            ValueError,
            "processing-zone IDs must be non-empty",
            id="empty-zone",
        ),
        pytest.param({"target_site_of_ion": ()}, ValueError, "at least one ion", id="no-target-sites"),
        pytest.param(
            {"target_site_of_ion": (1, 1)},
            ValueError,
            "target sites must be unique",
            id="repeated-site",
        ),
        pytest.param({"swaps": -1}, ValueError, "swaps must be non-negative", id="negative-swaps"),
        pytest.param({"shuttles": True}, TypeError, "shuttles must be an integer", id="boolean-shuttles"),
        pytest.param({"start_time": 6}, ValueError, "must not end before it starts", id="reversed-times"),
    ],
)
def test_gate_group_rejects_invalid_values(changes: dict[str, object], error: type[Exception], message: str) -> None:
    """Reject a group record that cannot describe one group of scheduled gates."""
    with pytest.raises(error, match=message):
        _gate_group(**changes)


def test_permutation_diagnostics_reject_invalid_values() -> None:
    """Reject negative counters and invalid trace entries."""
    with pytest.raises(ValueError, match="groups must be non-negative"):
        PermutationDiagnostics(score=0, groups=-1, groups_without_moves=0, gates_in_preferred_zone=0)
    with pytest.raises(TypeError, match="GateGroup values"):
        PermutationDiagnostics(
            score=0,
            groups=1,
            groups_without_moves=0,
            gates_in_preferred_zone=0,
            planning_trace=(cast("GateGroup", object()),),
        )
    with pytest.raises(ValueError, match="groups_without_moves must be <= groups"):
        PermutationDiagnostics(score=0, groups=0, groups_without_moves=1, gates_in_preferred_zone=0)
    with pytest.raises(ValueError, match="one entry per group"):
        PermutationDiagnostics(
            score=0,
            groups=2,
            groups_without_moves=0,
            gates_in_preferred_zone=0,
            planning_trace=(_gate_group(),),
        )
    with pytest.raises(ValueError, match="makespan_lower_bound must be non-negative"):
        PermutationDiagnostics(
            score=0,
            groups=0,
            groups_without_moves=0,
            gates_in_preferred_zone=0,
            makespan_lower_bound=-1,
        )


def test_gate_group_rejects_malformed_zone_assignments() -> None:
    """Reject an ion-zone entry that is not an ion and a zone name."""
    data = _gate_group().to_dict()
    data["zone_by_ion"] = [[0, 1]]

    with pytest.raises(ValueError, match="zone_by_ion entries"):
        GateGroup.from_dict(data)


def test_permutation_diagnostics_round_trip_with_and_without_trace() -> None:
    """Restore permutation statistics and the optional planning trace from JSON."""
    traced = PermutationDiagnostics(
        score=5,
        makespan_lower_bound=3,
        groups=1,
        groups_without_moves=0,
        gates_in_preferred_zone=2,
        planning_trace=(_gate_group(),),
    )
    untraced = replace(traced, planning_trace=None)

    assert LinearDiagnostics.from_dict(traced.to_dict()) == traced
    assert LinearDiagnostics.from_dict(untraced.to_dict()) == untraced
    assert "planning_trace" not in cast("dict[str, object]", untraced.to_dict()["permutation"])


def test_linear_diagnostics_loads_version_one_data_without_lower_bound() -> None:
    """Use zero when version 1 diagnostics omit the newer lower-bound field."""
    data = SearchDiagnostics(score=5, explored_nodes=3).to_dict()
    del data["makespan_lower_bound"]

    assert LinearDiagnostics.from_dict(data).makespan_lower_bound == 0


def test_linear_diagnostics_validates_present_lower_bound() -> None:
    """Reject a malformed lower bound when the serialized field is present."""
    data = SearchDiagnostics(score=5, explored_nodes=3).to_dict()
    data["makespan_lower_bound"] = "0"

    with pytest.raises(ValueError, match="makespan_lower_bound must be an integer"):
        LinearDiagnostics.from_dict(data)


def test_schedule_round_trips_identity_and_machine_metadata() -> None:
    """Preserve the complete action-level execution boundary."""
    architecture = LinearArchitecture(
        num_sites=2,
        processing_zones={"pz": [0, 1]},
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    program = schedule_from_path(
        [
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Rx(ion=0, theta=pi),
            GlobalGate(gate_name="rx", theta=pi, ions=(0,)),
            AdvanceTime(),
            Rz(ion=0, theta=0.2),
        ],
        create_initial_state(1, architecture),
        architecture,
    )

    restored = schedule_from_json(program.to_json())

    assert restored == program
    replay_schedule(restored, architecture)
    assert [item.action_id for item in restored.scheduled_actions] == list(range(4))
    assert [item.start_time for item in restored.scheduled_actions] == [0, 1, 1, 2]
    serialized = restored.to_dict()
    actions = serialized["actions"]
    assert isinstance(actions, list)
    assert [cast("dict[str, dict[str, object]]", item)["action"]["type"] for item in actions] == [
        "linear.shuttle",
        "gate.rx",
        "gate.global",
        "gate.rz",
    ]
    assert "architecture" not in serialized
    assert "action_types" not in serialized
    assert serialized["schema"] == "mqt.ionshuttler.schedule"
    assert serialized["version"] == 1


def test_machine_state_strips_compiler_progress_and_canonicalizes_availability() -> None:
    """Expose replay state without leaking circuit-search progress."""
    state = replace(
        create_initial_state(1, LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})),
        completed_gates=frozenset({3}),
        in_progress_gates=((4, 7),),
        ions_busy_until=((0, 2),),
        pzs_busy_until=(("pz", 1),),
        time=5,
    )

    machine = LinearMachineState.from_compiler_state(state)
    replay = machine.to_replay_state()

    assert machine.ions_busy_until == ((0, 5),)
    assert machine.pzs_busy_until == (("pz", 5),)
    assert replay.completed_gates == frozenset()
    assert replay.in_progress_gates == ()


def test_compilation_result_round_trips_only_compiler_diagnostics() -> None:
    """Keep compiler outcome fields outside the nested scheduled program."""
    architecture = LinearArchitecture(num_sites=4, processing_zones={"pz": [3, 0]}, topology="ring")
    program = schedule_from_path(
        [Shuttle(ion=0, src=0, dst=3), AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=architecture,
        wall_clock_s=0.5,
        final_state=architecture.replay_schedule(program),
        diagnostics=SearchDiagnostics(
            score=1,
            explored_nodes=7,
            preferred_gate_zones=((2, "pz"),),
        ),
    )

    restored = result_from_json(result.to_json())

    assert restored == result
    assert restored.wall_clock_s == result.wall_clock_s
    assert restored.architecture.topology == "ring"
    serialized = restored.to_dict()
    assert serialized["architecture"] == architecture.to_dict()
    assert serialized["diagnostics"] == {
        "score": 1,
        "explored_nodes": 7,
        "preferred_gate_zones": [[2, "pz"]],
        "makespan_lower_bound": 0,
        "permutation": None,
    }
    assert serialized["schema"] == "mqt.ionshuttler.compilation_result"
    assert serialized["version"] == 1
    assert "dd_insertions" not in serialized


def test_linear_results_are_saved_and_loaded_through_the_linear_package(tmp_path: Path) -> None:
    """Load a saved Linear artifact with the Linear loader and validate it."""
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    program = schedule_from_path([Rx(ion=0, theta=0.2)], create_initial_state(1, architecture), architecture)
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=architecture,
        final_state=architecture.replay_schedule(program),
    )
    path = result.save(tmp_path / "result")

    restored = load_result(path)

    assert path.suffix == ".json"
    assert restored == result
    restored.validate()
    assert not hasattr(CompilationResult, "load")


def test_loaders_restore_only_actions_implemented_by_linear_architectures(tmp_path: Path) -> None:
    """Refuse a saved action or catalog entry that Linear architectures do not implement."""
    architecture = LinearArchitecture(num_sites=1)
    program = schedule_from_path([Rx(ion=0, theta=0.2)], create_initial_state(1, architecture), architecture)
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=architecture,
        final_state=architecture.replay_schedule(program),
    )
    assert load_schedule(program.save(tmp_path / "program")) == program

    schedule_data = program.to_dict()
    actions = cast("list[dict[str, dict[str, object]]]", schedule_data["actions"])
    actions[0]["action"]["type"] = "test.marker"
    result_data = result.to_dict()
    cast("dict[str, object]", result_data["architecture"])["supported_action_types"] = ["test.marker"]

    with pytest.raises(ValueError, match=r"unknown action type: test\.marker"):
        schedule_from_dict(schedule_data)
    with pytest.raises(ValueError, match=r"unknown architecture action type: 'test\.marker'"):
        result_from_dict(result_data)


def test_visualizer_protocol_supports_explicit_visualizers() -> None:
    """Use an explicit visualizer directly through the shared contract."""
    program = _program()
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]}),
        final_state=program.initial_state,
    )

    class StubVisualizer:
        def visualize(self, received: CompilationResult) -> tuple[str, CompilationResult]:
            return "view", received

    visualizer: Visualizer = StubVisualizer()

    assert visualizer.visualize(result) == ("view", result)


def test_result_visualization_builds_linear_trajectory_plot() -> None:
    """Plot ion motion, processing zones, and gates on one time-site axis."""
    pyplot = pytest.importorskip("matplotlib.pyplot")
    architecture = LinearArchitecture(
        num_sites=4,
        processing_zones={"pz": [1, 2]},
        supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalGate),
    )
    program = schedule_from_path(
        [
            Shuttle(ion=0, src=0, dst=1),
            AdvanceTime(),
            Rzz(ion_a=0, ion_b=1, theta=0.2),
            GlobalGate(gate_name="rx", theta=pi, ions=(1,)),
            AdvanceTime(),
            AdvanceTime(),
        ],
        create_initial_state(2, architecture, initial_positions=[0, 2]),
        architecture,
    )
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=architecture,
        final_state=architecture.replay_schedule(program),
    )

    figure = cast("Figure", visualize(result))

    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert axis.get_xlabel() == "timestep"
    assert axis.get_ylabel() == "site"
    ion_lines = {
        label: line for line in axis.lines if isinstance(label := line.get_label(), str) and label.startswith("ion ")
    }
    assert tuple(cast("Sequence[int]", ion_lines["ion 0"].get_xdata())) == (0, 1, 3)
    assert tuple(cast("Sequence[int]", ion_lines["ion 0"].get_ydata())) == (0, 1, 1)
    assert tuple(cast("Sequence[int]", ion_lines["ion 1"].get_xdata())) == (0, 3)
    assert tuple(cast("Sequence[int]", ion_lines["ion 1"].get_ydata())) == (2, 2)
    assert any(text.get_text() == "Rzz" for text in axis.texts)
    assert any(text.get_text() == "Rx [1]" for text in axis.texts)
    assert [patch.get_label() for patch in axis.patches].count("PZ") == 1
    pyplot.close(figure)


def test_ring_visualization_splits_wrapping_zones_and_wrap_shuttles() -> None:
    """Show both sides of a wrapping zone without a full-array transport line."""
    pyplot = pytest.importorskip("matplotlib.pyplot")
    architecture = LinearArchitecture(
        num_sites=5,
        processing_zones={"wrap": [4, 0]},
        topology="ring",
    )
    program = schedule_from_path(
        [Shuttle(ion=0, src=0, dst=4), AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[0]),
        architecture,
    )
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=architecture,
        final_state=architecture.replay_schedule(program),
    )

    figure = cast("Figure", visualize(result))

    axis = figure.axes[0]
    # Matplotlib versions represent axhspan regions with different patch classes.
    zone_spans = tuple(
        coordinate
        for patch in axis.patches
        for coordinate in patch.get_extents().transformed(axis.transData.inverted()).intervaly
    )
    assert zone_spans == pytest.approx((-0.45, 0.45, 3.55, 4.45))
    seam_lines = [line for line in axis.lines if line.get_linestyle() == ":"]
    assert [tuple(cast("Sequence[float]", line.get_ydata())) for line in seam_lines] == [
        (-0.5, -0.5),
        (4.5, 4.5),
    ]
    assert axis.get_ylim() == (-0.75, 4.75)
    ion_line = next(line for line in axis.lines if line.get_label() == "ion 0")
    ion_sites = tuple(cast("Sequence[float]", ion_line.get_ydata()))
    assert ion_sites[0] == 0
    assert isnan(ion_sites[1])
    assert ion_sites[2] == 4
    wrap_lines = [line for line in axis.lines if line.get_gid() == "ring transport"]
    assert [tuple(cast("Sequence[float]", line.get_xdata())) for line in wrap_lines] == [(0, 0.25), (0.75, 1)]
    assert [tuple(cast("Sequence[float]", line.get_ydata())) for line in wrap_lines] == [
        (0, -0.5),
        (4.5, 4),
    ]
    assert len(axis.collections) == 0
    pyplot.close(figure)


def test_ring_visualization_breaks_both_wrap_swap_trajectories() -> None:
    """Break both ion trajectories when a physical swap crosses the ring seam."""
    pyplot = pytest.importorskip("matplotlib.pyplot")
    architecture = LinearArchitecture(num_sites=5, processing_zones={"pz": [2]}, topology="ring")
    program = schedule_from_path(
        [PhysicalSwap(ion_a=0, ion_b=1, pos_a=0, pos_b=4)],
        create_initial_state(2, architecture, initial_positions=[0, 4]),
        architecture,
    )
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=architecture,
        final_state=architecture.replay_schedule(program),
    )

    figure = cast("Figure", visualize(result))

    axis = figure.axes[0]
    ion_lines = {line.get_label(): tuple(cast("Sequence[float]", line.get_ydata())) for line in axis.lines}
    assert isnan(ion_lines["ion 0"][1])
    assert isnan(ion_lines["ion 1"][1])
    swap_end = architecture.transport_timing.swap
    wrap_lines = [line for line in axis.lines if line.get_gid() == "ring transport"]
    assert [tuple(cast("Sequence[float]", line.get_xdata())) for line in wrap_lines] == [
        (0, 0.25),
        (swap_end - 0.25, swap_end),
        (0, 0.25),
        (swap_end - 0.25, swap_end),
    ]
    assert [tuple(cast("Sequence[float]", line.get_ydata())) for line in wrap_lines] == [
        (0, -0.5),
        (4.5, 4),
        (4, 4.5),
        (-0.5, 0),
    ]
    assert len(axis.collections) == 0
    pyplot.close(figure)


def test_ring_visualization_connects_wrap_gates_through_the_seam() -> None:
    """Draw a two-ion gate in a wrapping zone through the seam, not across the array."""
    pyplot = pytest.importorskip("matplotlib.pyplot")
    architecture = LinearArchitecture(num_sites=5, processing_zones={"wrap": [3, 4, 0]}, topology="ring")
    program = schedule_from_path(
        [Rzz(ion_a=0, ion_b=1, theta=0.2), AdvanceTime(), AdvanceTime(), Rzz(ion_a=1, ion_b=2, theta=0.2)],
        create_initial_state(3, architecture, initial_positions=[0, 4, 3]),
        architecture,
    )
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=architecture,
        final_state=architecture.replay_schedule(program),
    )

    figure = cast("Figure", visualize(result))

    axis = figure.axes[0]
    wrap_gate, inner_gate = program.scheduled_actions
    seam_connectors = [line for line in axis.lines if line.get_gid() == "ring gate connector"]
    assert [tuple(cast("Sequence[float]", line.get_xdata())) for line in seam_connectors] == [
        (wrap_gate.start_time, wrap_gate.start_time),
    ] * 2
    assert [tuple(cast("Sequence[float]", line.get_ydata())) for line in seam_connectors] == [(0, -0.5), (4, 4.5)]
    connectors = [line for line in axis.lines if line.get_color() == "0.2" and line.get_gid() is None]
    assert [tuple(cast("Sequence[float]", line.get_ydata())) for line in connectors] == [(3, 4)]
    labels = {cast("Annotation", text).xy: text.get_text() for text in axis.texts}
    assert labels == {(wrap_gate.start_time, 0): "Rzz", (inner_gate.start_time, 3.5): "Rzz"}
    pyplot.close(figure)


def test_linear_visualizer_is_available_as_a_concrete_adapter() -> None:
    """Expose the built-in visualizer for direct use and extension examples."""
    pyplot = pytest.importorskip("matplotlib.pyplot")
    program = _program()
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]}),
        final_state=program.initial_state,
    )

    figure = cast("Figure", LinearVisualizer().visualize(result))

    assert len(figure.axes) == 1
    pyplot.close(figure)


def test_visualization_package_exports_its_supported_api() -> None:
    """Keep the visualization extension surface explicit."""
    assert visualization_module.__all__ == ["LinearVisualizer", "Visualizer", "visualize"]


def test_visualization_rejects_an_unsupported_architecture() -> None:
    """Report when no built-in visualizer supports the result architecture."""
    program = _program()
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=cast("LinearArchitecture", object()),
        final_state=program.initial_state,
    )

    with pytest.raises(ValueError, match="no built-in visualizer supports object"):
        visualize(result)


def test_result_visualization_accepts_architecture_subclasses() -> None:
    """Route a derived Linear architecture to the built-in schematic."""
    pyplot = pytest.importorskip("matplotlib.pyplot")

    class DerivedArchitecture(LinearArchitecture):
        """Linear architecture used to check subclass dispatch."""

    program = _program([Rx(ion=0, theta=0.2)])
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=DerivedArchitecture(num_sites=2, processing_zones={"pz": [0, 1]}),
        final_state=program.initial_state,
    )

    figure = cast("Figure", visualize(result))

    assert len(figure.axes) == 1
    pyplot.close(figure)


def test_result_visualization_wraps_long_timelines() -> None:
    """Fold on five-tick boundaries and preserve one physical timestep scale."""
    pyplot = pytest.importorskip("matplotlib.pyplot")
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    initial_state = create_initial_state(1, architecture)
    program = Schedule((), 92, LinearMachineState.from_compiler_state(initial_state))
    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=program,
        architecture=architecture,
        final_state=architecture.replay_schedule(program),
    )

    figure = cast("Figure", visualize(result))

    assert len(figure.axes) > 1
    windows = tuple(axis.get_xlim() for axis in figure.axes)
    assert windows == ((0, 25), (25, 50), (50, 75), (75, 92))
    assert all(left[1] == right[0] for left, right in itertools.pairwise(windows))
    assert all(window_end % 5 == 0 for _, window_end in windows[:-1])
    assert all(set(axis.get_xticks()[1:] - axis.get_xticks()[:-1]) == {5} for axis in figure.axes)
    first_spec = figure.axes[0].get_subplotspec()
    last_spec = figure.axes[-1].get_subplotspec()
    assert first_spec is not None
    assert last_spec is not None
    first_span = first_spec.colspan
    last_span = last_spec.colspan
    assert (first_span.stop - first_span.start, last_span.stop - last_span.start) == (25, 17)
    pyplot.close(figure)


def test_schedule_rejects_actions_before_the_initial_state() -> None:
    """Reject a schedule that starts before its own initial state time."""
    architecture = LinearArchitecture(num_sites=2, processing_zones={"pz": [0, 1]})
    initial_state = replace(create_initial_state(1, architecture, initial_positions=[0]), time=3)

    with pytest.raises(ValueError, match="must not start before the initial state time"):
        Schedule(
            scheduled_actions=(ScheduledAction(0, Rx(ion=0, theta=0.2), start_time=1, duration=1),),
            end_time=5,
            initial_state=LinearMachineState.from_compiler_state(initial_state),
        )
