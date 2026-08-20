# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dynamical-decoupling methods for Linear action schedules."""

from mqt.ionshuttler.linear.dd.critical_segments import (
    CriticalSegment,
    CriticalSegmentResult,
    compute_critical_segments,
    gate_z_effect,
    normalized_sensitivity_values,
)
from mqt.ionshuttler.linear.dd.global_dd import (
    GlobalDDConfig,
    GlobalDDReport,
    apply_periodic_global_dd,
)
from mqt.ionshuttler.linear.dd.idealized_hahn import (
    IdealizedHahnConfig,
    IdealizedHahnReport,
    apply_idealized_hahn,
)
from mqt.ionshuttler.linear.dd.metrics import (
    decoupling_ratio,
    max_absolute_residual_phase,
    residual_phase_by_ion,
    sum_absolute_residual_phase,
    sum_absolute_residual_phases_at_gates,
    sum_squared_residual_phase,
)
from mqt.ionshuttler.linear.dd.result import DDPassResult, LocalDDSequence
from mqt.ionshuttler.linear.dd.sadd import (
    OperationDurations,
    SADDConfig,
    SADDMethod,
    SADDOpportunityRecord,
    SADDReport,
    run_sadd,
)

__all__ = [
    "CriticalSegment",
    "CriticalSegmentResult",
    "DDPassResult",
    "GlobalDDConfig",
    "GlobalDDReport",
    "IdealizedHahnConfig",
    "IdealizedHahnReport",
    "LocalDDSequence",
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
