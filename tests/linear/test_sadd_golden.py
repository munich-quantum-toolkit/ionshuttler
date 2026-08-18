# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact source-derived golden tests for the SADD backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, cast

import pytest

from mqt.ionshuttler.linear.dd import OperationDurations, SADDConfig, SADDMethod, run_sadd
from mqt.ionshuttler.linear.result import CompilationResult

if TYPE_CHECKING:
    from mqt.ionshuttler.linear.dd.sadd import SADDOpportunityRecord
    from mqt.ionshuttler.linear.state import State

pytest.importorskip("ortools.sat.python.cp_model")


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _config(raw: object) -> SADDConfig:
    values = _mapping(raw)
    durations = _mapping(values["operation_durations"])
    return SADDConfig(
        min_window_length=cast("int", values["min_window_length"]),
        max_window_length=cast("int", values["max_window_length"]),
        max_participating_ions=cast("int", values["max_participating_ions"]),
        timeout_s=cast("float", values["timeout_s"]),
        ion_preselection=cast('Literal["phase", "distance"]', values["ion_preselection"]),
        opportunity_order=cast(
            'Literal["chronological", "reverse_chronological"]',
            values["opportunity_order"],
        ),
        max_accepted_windows=cast("int | None", values["max_accepted_windows"]),
        improvement_tolerance=cast("float", values["improvement_tolerance"]),
        allow_pulses=cast("bool", values["allow_pulses"]),
        scale=cast("int", values["scale"]),
        operation_durations=OperationDurations(
            shuttle=cast("int", durations["shuttle"]),
            swap=cast("int", durations["swap"]),
            one_qubit_gate=cast("int", durations["one_qubit_gate"]),
        ),
    )


def _record_observation(record: SADDOpportunityRecord) -> dict[str, object]:
    return {
        "target_pz": record.target_pz,
        "window": record.window,
        "participating_ions": record.participating_ions,
        "status": record.status,
        "validation_status": record.validation_status,
        "objective_before": record.objective_before,
        "objective_after": record.objective_after,
        "accepted": record.accepted,
        "pulse_count": record.pulse_count,
        "transport_action_count": record.transport_action_count,
        "message": record.message,
        "eligible_ions": record.eligible_ions,
        "rejected_busy_ions": record.rejected_busy_ions,
        "rejected_no_active_segment_ions": record.rejected_no_active_segment_ions,
        "rejected_unreachable_ions": record.rejected_unreachable_ions,
        "selection_scores": record.selection_scores,
        "phase_before_by_ion": None if record.phase_before_by_ion is None else dict(record.phase_before_by_ion),
        "phase_after_by_ion": None if record.phase_after_by_ion is None else dict(record.phase_after_by_ion),
        "pulse_timesteps": None if record.pulse_timesteps is None else dict(record.pulse_timesteps),
        "transport_actions": record.transport_actions,
        "trajectories": None if record.trajectories is None else dict(record.trajectories),
        "model_num_variables": record.model_num_variables,
        "model_num_constraints": record.model_num_constraints,
    }


def _state_observation(state: State | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "positions": state.positions,
        "completed_gates": sorted(state.completed_gates),
        "in_progress_gates": state.in_progress_gates,
        "ions_busy_until": state.ions_busy_until,
        "pzs_busy_until": state.pzs_busy_until,
        "time": state.time,
    }


def _json_values(value: object) -> object:
    return json.loads(json.dumps(value))


@pytest.mark.parametrize(
    ("method_name", "method"),
    [("pulse_only_sadd", SADDMethod.PULSE_ONLY), ("full_sadd", SADDMethod.FULL)],
)
def test_sadd_matches_frozen_source_observations_exactly(
    sadd_golden: dict[str, object],
    method_name: str,
    method: SADDMethod,
) -> None:
    """Match schedule, objective, selection, and model observations exactly."""
    expected_by_method = _mapping(sadd_golden["expected"])
    if method_name not in expected_by_method:
        pytest.skip(f"fixture does not contain {method_name}")
    base = CompilationResult.from_dict(sadd_golden["base_result"])
    output = run_sadd(base, method, config=_config(sadd_golden["config"]))
    expected = _mapping(expected_by_method[method_name])

    result_observation = output.program.to_dict()
    result_observation.pop("wall_clock_s", None)
    result_observation.pop("action_types", None)
    actual = {
        "allow_transport": method.allow_transport,
        "unavailable_reason": output.unavailable_reason,
        "final_state": _state_observation(output.program.final_state if output.program is not base else None),
        "result": result_observation,
        "records": [_record_observation(record) for record in output.report.opportunities],
    }

    assert _json_values(actual) == expected
