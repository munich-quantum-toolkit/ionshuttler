# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for generating and applying Linear schedule actions."""

from __future__ import annotations

import pytest

from mqt.ionshuttler.linear.actions import (
    DEFAULT_ACTION_TYPES,
    AdvanceTime,
    GateSpec,
    GlobalPulse,
    PhysicalSwap,
    Rx,
    Ry,
    Rz,
    Rzz,
    Shuttle,
)
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.config import TransportTiming
from mqt.ionshuttler.linear.expand import (
    ExpansionOptions,
    GenerationMode,
    apply,
    expand,
    generate_actions,
    generate_actions_by_mode,
    ready_gate_ids,
    replay_path,
)
from mqt.ionshuttler.linear.state import State


def make_state(
    positions: tuple[tuple[int, int], ...],
    *,
    completed: frozenset[int] = frozenset(),
    in_progress: tuple[tuple[int, int], ...] = (),
    ions_busy: tuple[tuple[int, int], ...] | None = None,
    pzs_busy: tuple[tuple[str, int], ...] = (("all_sites", 0),),
    time: int = 0,
) -> State:
    """Build a compact state for expansion tests."""
    return State(
        positions=positions,
        completed_gates=completed,
        in_progress_gates=in_progress,
        ions_busy_until=(tuple((ion, 0) for ion, _ in positions) if ions_busy is None else ions_busy),
        pzs_busy_until=pzs_busy,
        time=time,
    )


def test_ready_gate_ids_returns_all_dependency_ready_gates() -> None:
    """Offer every unfinished gate whose direct predecessors are complete."""
    predecessors = {4: frozenset(), 7: frozenset({4}), 9: frozenset()}

    assert ready_gate_ids([4, 7, 9], predecessors, frozenset(), ()) == [4, 9]
    assert ready_gate_ids([4, 7, 9], predecessors, frozenset({4}), ()) == [7, 9]
    assert ready_gate_ids([4, 7, 9], predecessors, frozenset({4}), ((7, 3),)) == [9]


def test_full_generation_includes_gates_and_configured_transports() -> None:
    """Offer ready gates, swaps, and shuttles with the configured durations."""
    state = make_state(((0, 0), (1, 1)))
    gate = Rx(ion=0, theta=1.0)

    actions = generate_actions(
        state,
        Architecture(num_sites=3),
        [0],
        {0: gate},
        transport_timing=TransportTiming(shuttle=2, swap=3),
    )

    assert gate in actions
    assert PhysicalSwap(ion_a=0, ion_b=1, pos_a=0, pos_b=1, duration=3) in actions
    assert Shuttle(ion=1, src=1, dst=2, duration=2) in actions


@pytest.mark.parametrize("mode", [GenerationMode.FULL, GenerationMode.INFORMED])
def test_generation_excludes_gate_types_missing_from_hardware_catalog(mode: GenerationMode) -> None:
    """Do not propose circuit gates that the hardware does not expose."""
    state = make_state(((0, 0),), pzs_busy=(("pz", 0),))
    gate = Rx(ion=0, theta=1.0)

    actions = generate_actions_by_mode(
        state,
        Architecture(num_sites=2, processing_zones={"pz": [0]}),
        [0],
        {0: gate},
        options=ExpansionOptions(mode=mode, action_types=(Shuttle,)),
    )

    assert gate not in actions


def test_generation_respects_dependencies_and_busy_resources() -> None:
    """Wait for predecessors and occupied hardware before offering a gate."""
    first = Rx(ion=0, theta=1.0)
    second = Ry(ion=0, theta=0.5)
    state = make_state(
        ((0, 0),),
        in_progress=((0, 2),),
        ions_busy=((0, 2),),
    )

    actions = generate_actions(
        state,
        Architecture(num_sites=1),
        [0, 1],
        {0: first, 1: second},
        predecessors={0: frozenset(), 1: frozenset({0})},
    )

    assert first not in actions
    assert second not in actions
    assert actions == [AdvanceTime()]


def test_advance_time_is_generated_only_while_timed_work_is_pending() -> None:
    """Avoid creating idle branches while retaining deliberate waiting as an action."""
    architecture = Architecture(num_sites=2)
    idle = make_state(((0, 0),))
    busy = make_state(((0, 0),), ions_busy=((0, 2),), time=1)

    assert AdvanceTime() not in generate_actions(idle, architecture, [], {})
    assert AdvanceTime() in generate_actions(busy, architecture, [], {})
    assert AdvanceTime().is_valid(idle, architecture)


def test_virtual_and_zero_duration_physical_gates_complete_immediately() -> None:
    """Complete instantaneous gates without placing them in the running set."""
    architecture = Architecture(num_sites=1)
    state = make_state(((0, 0),))
    virtual_gate = Rz(ion=0, theta=0.25)
    with pytest.warns(UserWarning, match="zero duration"):
        physical_gate = Rx(ion=0, theta=0.5, duration=0)

    after_virtual = apply(state, architecture, virtual_gate, gate_id=3)
    after_physical = apply(after_virtual, architecture, physical_gate, gate_id=7)

    assert after_physical.completed_gates == frozenset({3, 7})
    assert after_physical.in_progress_gates == ()
    assert after_physical.time == 0


def test_timed_gate_reserves_hardware_and_finishes_after_time_advances() -> None:
    """Track a running gate until its duration has elapsed."""
    architecture = Architecture(num_sites=1)
    state = make_state(((0, 0),))
    gate = Rx(ion=0, theta=1.0, duration=2)

    started = apply(state, architecture, gate, gate_id=4)
    after_one_tick = apply(started, architecture, AdvanceTime())
    finished = apply(after_one_tick, architecture, AdvanceTime())

    assert started.in_progress_gates == ((4, 2),)
    assert started.ions_busy_until == ((0, 2),)
    assert after_one_tick.completed_gates == frozenset()
    assert finished.completed_gates == frozenset({4})
    assert finished.in_progress_gates == ()


def test_expand_carries_the_exact_gate_id_for_equal_actions() -> None:
    """Keep equal circuit gates distinct through scheduler bookkeeping."""
    state = make_state(((0, 0), (1, 1)))
    first = Rx(ion=0, theta=1.0)
    second = Rx(ion=0, theta=1.0)
    independent = Ry(ion=1, theta=0.5)

    children = expand(
        state,
        Architecture(num_sites=2),
        [4, 7, 9],
        {4: first, 7: second, 9: independent},
        predecessors={4: frozenset(), 7: frozenset({4}), 9: frozenset()},
    )

    gate_children = [(action, gate_id, child) for action, gate_id, child in children if gate_id is not None]
    assert [(action, gate_id) for action, gate_id, _ in gate_children] == [
        (first, 4),
        (independent, 9),
    ]
    assert gate_children[0][2].in_progress_gates == ((4, 1),)


def test_informed_generation_prefers_ready_gates() -> None:
    """Choose executable circuit work before considering ion movement."""
    state = make_state(((0, 0), (1, 2)))
    gate = Rx(ion=0, theta=1.0)

    actions = generate_actions_by_mode(
        state,
        Architecture(num_sites=3),
        [0],
        {0: gate},
        options=ExpansionOptions(mode=GenerationMode.INFORMED),
    )

    assert actions == [gate]


def test_informed_generation_moves_relevant_ions_toward_a_zone() -> None:
    """Move only upcoming gate ions closer when no gate can run yet."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [2, 3]})
    state = make_state(
        ((0, 0), (1, 4)),
        pzs_busy=(("pz", 0),),
    )
    gate = Rzz(ion_a=0, ion_b=1, theta=1.0)

    actions = generate_actions_by_mode(
        state,
        architecture,
        [0],
        {0: gate},
        options=ExpansionOptions(mode=GenerationMode.INFORMED),
    )

    assert actions == [
        Shuttle(ion=0, src=0, dst=1),
        Shuttle(ion=1, src=4, dst=3),
    ]


def test_informed_generation_swaps_a_relevant_ion_toward_a_zone() -> None:
    """Use a neighboring ion when it is the only path toward a processing zone."""
    architecture = Architecture(num_sites=3, processing_zones={"pz": [0]})
    state = make_state(((0, 2), (1, 1)), pzs_busy=(("pz", 0),))
    gate = Rx(ion=0, theta=1.0)

    actions = generate_actions_by_mode(
        state,
        architecture,
        [0],
        {0: gate},
        options=ExpansionOptions(mode=GenerationMode.INFORMED),
    )

    assert actions == [PhysicalSwap(ion_a=0, ion_b=1, pos_a=2, pos_b=1, duration=3)]


def test_uninformed_generation_is_the_remaining_full_action_set() -> None:
    """Use the broader candidates as a fallback after informed choices."""
    architecture = Architecture(num_sites=5, processing_zones={"pz": [2, 3]})
    state = make_state(((0, 0), (1, 4)), pzs_busy=(("pz", 0),))
    gates = {0: Rzz(ion_a=0, ion_b=1, theta=1.0)}
    uninformed = generate_actions_by_mode(
        state,
        architecture,
        [0],
        gates,
        options=ExpansionOptions(mode=GenerationMode.UNINFORMED),
    )

    assert uninformed == []


def test_generation_accepts_an_additional_gate_action_type() -> None:
    """Schedule a hardware gate without teaching generation about its concrete class."""
    architecture = Architecture(num_sites=1)
    state = make_state(((0, 0),))
    gate = GlobalPulse(gate=GateSpec("rx", 0.5), duration=2)

    children = expand(
        state,
        architecture,
        [6],
        {6: gate},
        options=ExpansionOptions(action_types=(*DEFAULT_ACTION_TYPES, GlobalPulse)),
    )

    assert children[0][0:2] == (gate, 6)
    assert children[0][2].in_progress_gates == ((6, 2),)


def test_replay_path_validates_actions_and_preserves_the_initial_state() -> None:
    """Reproduce a schedule without mutating its starting state."""
    architecture = Architecture(num_sites=1)
    initial = make_state(((0, 0),))
    gate = Rx(ion=0, theta=1.0)
    path = [gate, AdvanceTime()]

    final = replay_path(initial, architecture, path, [0], {0: gate})

    assert final.completed_gates == frozenset({0})
    assert initial.completed_gates == frozenset()
    assert initial.time == 0

    with pytest.raises(ValueError, match="not valid"):
        replay_path(initial, architecture, [Shuttle(ion=0, src=1, dst=0)], [], {})


def test_replay_matches_an_equivalent_reconstructed_gate() -> None:
    """Recognize an equivalent gate after a schedule has been reconstructed."""
    architecture = Architecture(num_sites=1)
    initial = make_state(((0, 0),))
    circuit_gate = Rx(ion=0, theta=1.0)
    scheduled_gate = Rx(ion=0, theta=1.0)

    final = replay_path(
        initial,
        architecture,
        [scheduled_gate, AdvanceTime()],
        [4],
        {4: circuit_gate},
    )

    assert final.completed_gates == frozenset({4})


def test_replay_rejects_an_action_matching_multiple_ready_gates() -> None:
    """Reject equality matching when it cannot identify one circuit gate."""
    architecture = Architecture(num_sites=1)
    initial = make_state(((0, 0),))
    gates = {
        4: Rx(ion=0, theta=1.0),
        5: Rx(ion=0, theta=1.0),
    }

    with pytest.raises(ValueError, match="does not identify exactly one ready gate"):
        replay_path(
            initial,
            architecture,
            [Rx(ion=0, theta=1.0)],
            [4, 5],
            gates,
            predecessors={4: frozenset(), 5: frozenset()},
        )


def test_gate_action_requires_an_explicit_circuit_id() -> None:
    """Keep circuit progress separate from a gate's hardware transition."""
    with pytest.raises(ValueError, match="gate_id is required"):
        apply(
            make_state(((0, 0),)),
            Architecture(num_sites=1),
            Rx(ion=0, theta=1.0),
        )
