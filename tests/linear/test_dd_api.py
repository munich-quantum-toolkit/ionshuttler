# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the public Linear dynamical-decoupling contract."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from mqt.ionshuttler.linear import dd
from mqt.ionshuttler.linear.dd import (
    DDPassResult,
    GlobalDDConfig,
    GlobalDDReport,
    IdealizedHahnConfig,
    IdealizedHahnReport,
    OperationDurations,
    SADDConfig,
    SADDMethod,
    SADDOpportunityRecord,
    SADDReport,
    sadd_solver,
)


def _opportunity(**overrides: object) -> SADDOpportunityRecord:
    values: dict[str, object] = {
        "target_pz": "pz",
        "window": (0, 3),
        "participating_ions": (0,),
        "status": "OPTIMAL",
        "validation_status": "valid",
        "objective_before": 2.0,
        "objective_after": 0.5,
        "accepted": True,
        "pulse_count": 1,
        "transport_action_count": 0,
        "runtime_s": 0.1,
        "eligible_ions": (0,),
        "selection_scores": ((0, 2.0, 0),),
        "phase_before_by_ion": {0: 2.0},
        "phase_after_by_ion": {0: 0.5},
        "pulse_timesteps": {0: (2,)},
        "trajectories": {0: (1, 1, 1)},
        "model_num_variables": 38,
        "model_num_constraints": 94,
    }
    values.update(overrides)
    return SADDOpportunityRecord(**values)  # ty: ignore[invalid-argument-type]


def test_dd_package_exports_only_the_supported_public_surface() -> None:
    """Keep the DD public surface narrow and intentional."""
    assert dd.__all__ == [
        "CriticalSegment",
        "CriticalSegmentResult",
        "DDPassResult",
        "GlobalDDConfig",
        "GlobalDDReport",
        "IdealizedHahnConfig",
        "IdealizedHahnReport",
        "OperationDurations",
        "SADDConfig",
        "SADDMethod",
        "SADDOpportunityRecord",
        "SADDReport",
        "apply_idealized_hahn",
        "apply_periodic_global_dd",
        "compute_critical_segments",
        "decoupling_ratio",
        "gate_z_effect",
        "max_absolute_residual_phase",
        "normalized_sensitivity_values",
        "residual_phase_by_ion",
        "run_sadd",
        "sum_absolute_residual_phase",
        "sum_absolute_residual_phases_at_gates",
        "sum_squared_residual_phase",
    ]


def test_comparator_report_types_are_available_without_optional_dependencies() -> None:
    """Expose solver-free immutable result types for both comparator methods."""
    idealized = IdealizedHahnReport()
    global_report = GlobalDDReport(
        scheme_name="periodic_x",
        pulse_timesteps=(1, 3),
        spacing=2,
        sum_absolute_residual_phase=0.0,
        sum_squared_residual_phase=0.0,
        max_absolute_residual_phase=0.0,
    )

    assert IdealizedHahnConfig().label == "IdealizedHahn"
    assert GlobalDDConfig(spacing=5).half_first_window
    assert idealized.insertions == ()
    assert global_report.pulse_timesteps == (1, 3)


def test_sadd_defaults_freeze_the_paper_configuration() -> None:
    """Expose the current paper parameters as the library defaults."""
    config = SADDConfig()

    assert config == SADDConfig(
        min_window_length=2,
        max_window_length=16,
        max_participating_ions=5,
        timeout_s=10.0,
        ion_preselection="phase",
        opportunity_order="chronological",
        max_accepted_windows=None,
        improvement_tolerance=1e-12,
        allow_pulses=True,
        scale=1000,
        num_search_workers=8,
        operation_durations=OperationDurations(shuttle=1, swap=3, one_qubit_gate=1),
    )
    assert SADDMethod.PULSE_ONLY.allow_transport is False
    assert SADDMethod.FULL.allow_transport is True


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"min_window_length": 0}, ValueError, "min_window_length"),
        ({"min_window_length": True}, TypeError, "min_window_length"),
        ({"max_window_length": 1}, ValueError, "max_window_length"),
        ({"max_participating_ions": 0}, ValueError, "max_participating_ions"),
        ({"timeout_s": float("inf")}, ValueError, "timeout_s"),
        ({"timeout_s": 0.0}, ValueError, "timeout_s"),
        ({"ion_preselection": "unknown"}, ValueError, "ion_preselection"),
        ({"opportunity_order": "unknown"}, ValueError, "opportunity_order"),
        ({"max_accepted_windows": -1}, ValueError, "max_accepted_windows"),
        ({"improvement_tolerance": -1.0}, ValueError, "improvement_tolerance"),
        ({"allow_pulses": 1}, TypeError, "allow_pulses"),
        ({"scale": 0}, ValueError, "scale"),
        ({"num_search_workers": 0}, ValueError, "num_search_workers"),
        ({"operation_durations": object()}, TypeError, "operation_durations"),
    ],
)
def test_sadd_config_rejects_invalid_values(
    overrides: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    """Reject invalid SADD parameters at configuration construction."""
    with pytest.raises(exception, match=message):
        SADDConfig(**overrides)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"window": (2, 2)}, ValueError, "window end"),
        ({"accepted": True, "objective_after": None}, ValueError, "objective_after"),
        ({"objective_before": float("nan")}, ValueError, "objective_before"),
        ({"pulse_count": 2}, ValueError, "pulse_count"),
        ({"transport_action_count": 1}, ValueError, "transport_action_count"),
        ({"participating_ions": (0, 0)}, ValueError, "duplicate"),
    ],
)
def test_sadd_opportunity_rejects_inconsistent_values(
    overrides: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    """Reject malformed or internally inconsistent solver observations."""
    with pytest.raises(exception, match=message):
        _opportunity(**overrides)


def test_sadd_values_are_immutable_and_copy_mutable_inputs() -> None:
    """Prevent reports from mutating either inputs or frozen observations."""
    phase_before = {0: 2.0}
    pulse_timesteps = {0: (2,)}
    opportunity = _opportunity(
        phase_before_by_ion=phase_before,
        pulse_timesteps=pulse_timesteps,
    )
    report = SADDReport(method=SADDMethod.PULSE_ONLY, opportunities=(opportunity,))
    result = DDPassResult(program="unchanged", report=report)

    phase_before[0] = 99.0
    pulse_timesteps[0] = (1, 2)

    assert opportunity.phase_before_by_ion == {0: 2.0}
    assert opportunity.pulse_timesteps == {0: (2,)}
    assert result.program == "unchanged"
    with pytest.raises(TypeError):
        cast("dict[int, float]", opportunity.phase_before_by_ion)[0] = 4.0
    program_attribute = "program"
    with pytest.raises(FrozenInstanceError):
        setattr(result, program_attribute, "mutated")


def test_dd_pass_result_rejects_empty_unavailability_reason() -> None:
    """Require actionable optional-dependency failure information."""
    report = SADDReport(method=SADDMethod.FULL)

    with pytest.raises(ValueError, match="unavailable_reason"):
        DDPassResult(program="unchanged", report=report, unavailable_reason="")


def test_missing_ortools_has_installation_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate only a missing OR-Tools package into the narrow dependency error."""

    def missing_ortools(_name: str) -> None:
        msg = "No module named 'ortools'"
        raise ModuleNotFoundError(msg, name="ortools")

    monkeypatch.setattr(sadd_solver, "import_module", missing_ortools)

    with pytest.raises(ImportError, match=r"'dd' extra"):
        sadd_solver._load_cp_model()


def test_solver_loader_does_not_mask_unrelated_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve failures raised by an installed solver's unrelated dependencies."""

    def missing_dependency(_name: str) -> None:
        msg = "No module named 'unrelated'"
        raise ModuleNotFoundError(msg, name="unrelated")

    monkeypatch.setattr(sadd_solver, "import_module", missing_dependency)

    with pytest.raises(ModuleNotFoundError, match="unrelated"):
        sadd_solver._load_cp_model()


def test_linear_and_dd_imports_do_not_load_ortools() -> None:
    """Keep ordinary package imports independent of the optional solver."""
    command = (
        "import sys; "
        "import mqt.ionshuttler.linear; "
        "import mqt.ionshuttler.linear.dd; "
        "assert not any(name == 'ortools' or name.startswith('ortools.') for name in sys.modules)"
    )

    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - The executable and arguments are fixed by the test.
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
