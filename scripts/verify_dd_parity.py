# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Manually compare the SADD port with a local IonSwapper checkout.

This developer tool is intentionally outside pytest and CI. It creates its
inputs in memory and does not write fixtures. Run it from an IonShuttler
development environment with the optional DD dependencies installed.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import mqt
from mqt.ionshuttler.linear.actions import AdvanceTime
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd import SADDConfig, SADDMethod, run_sadd
from mqt.ionshuttler.linear.field_profile import FieldProfile
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state

EXPECTED_IONSWAPPER_COMMIT = "5446fddbfa0bb161da9919c45bacdc88d38ae5f8"
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ionswapper",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "ionswapper",
        help="path to the read-only IonSwapper checkout",
    )
    parser.add_argument(
        "--strict-actions",
        action="store_true",
        help="also require one particular optimal action schedule when several optima tie",
    )
    return parser.parse_args()


def _load_source_modules(checkout: Path) -> dict[str, Any]:
    git = shutil.which("git")
    if git is None:
        msg = "git must be available to verify the IonSwapper revision"
        raise RuntimeError(msg)
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - No shell; arguments remain distinct.
        [git, "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != EXPECTED_IONSWAPPER_COMMIT:
        msg = f"expected IonSwapper commit {EXPECTED_IONSWAPPER_COMMIT}, found {commit}"
        raise RuntimeError(msg)
    source_root = checkout.resolve() / "src"
    if not source_root.is_dir():
        msg = f"IonSwapper source directory does not exist: {source_root}"
        raise FileNotFoundError(msg)
    namespace_root = str(source_root / "mqt")
    sys.path.insert(0, str(source_root))
    mqt.__path__.append(namespace_root)
    try:
        actions = importlib.import_module("mqt.ionswapper.actions")
        architecture = importlib.import_module("mqt.ionswapper.architecture")
        control_centric = importlib.import_module("mqt.ionswapper.dd.control_centric")
        field_profile = importlib.import_module("mqt.ionswapper.field_profile")
        result = importlib.import_module("mqt.ionswapper.result")
        state = importlib.import_module("mqt.ionswapper.state")
    finally:
        sys.path.pop(0)
    return {
        "AdvanceTime": actions.AdvanceTime,
        "Architecture": architecture.Architecture,
        "ControlCentricDDConfig": control_centric.ControlCentricDDConfig,
        "FieldProfile": field_profile.FieldProfile,
        "CompilationResult": result.CompilationResult,
        "CompilationStatus": result.CompilationStatus,
        "create_initial_state": state.create_initial_state,
        "run": control_centric.run_control_centric_dd,
    }


def _case(case_name: str, source: dict[str, Any]) -> tuple[ActionSchedule, Architecture, object, object]:
    initial_position = 1 if case_name == "local_pulse" else 0
    target_architecture = Architecture(
        num_sites=3,
        processing_zones={"pz": [1]},
        field_profile=FieldProfile(3, ((0, 3.0), (1, 1.0), (2, 0.5))),
    )
    target_schedule = ActionSchedule.from_actions(
        [AdvanceTime() for _ in range(3)],
        create_initial_state(1, target_architecture, initial_positions=[initial_position]),
    )

    source_architecture = source["Architecture"](
        num_sites=3,
        processing_zones={"pz": (1,)},
        field_profile=source["FieldProfile"](3, ((0, 3.0), (1, 1.0), (2, 0.5))),
    )
    source_result = source["CompilationResult"](
        status=source["CompilationStatus"].SUCCESS,
        path=[source["AdvanceTime"]() for _ in range(3)],
        num_timesteps=3,
        architecture=source_architecture,
        initial_state=source["create_initial_state"](
            1,
            source_architecture,
            initial_positions=[initial_position],
        ),
    )
    return target_schedule, target_architecture, source_architecture, source_result


def _actions(path: object) -> list[str]:
    return [repr(action) for action in path]  # ty: ignore[not-iterable]


def _opportunities(records: object) -> list[dict[str, object]]:
    # ty: ignore[not-iterable]
    normalized: list[dict[str, object]] = [
        {
            "window": list(record.window),
            "participating_ions": list(record.participating_ions),
            "status": record.status,
            "validation_status": record.validation_status,
            "phase_cost_before": (
                record.phase_cost_before if hasattr(record, "phase_cost_before") else record.objective_before
            ),
            "phase_cost_after": (
                record.phase_cost_after if hasattr(record, "phase_cost_after") else record.objective_after
            ),
            "accepted": record.accepted,
            "pulse_count": record.pulse_count,
            "transport_action_count": getattr(record, "transport_action_count", len(record.transport_actions)),
        }
        for record in records
    ]
    return normalized


def _observation(
    actions: list[str],
    opportunities: list[dict[str, object]],
    *,
    strict_actions: bool,
) -> dict[str, object]:
    observation: dict[str, object] = {"opportunities": opportunities}
    if strict_actions:
        observation["actions"] = actions
    return observation


def _diff(source: object, target: object) -> str:
    source_text = json.dumps(source, indent=2, sort_keys=True).splitlines()
    target_text = json.dumps(target, indent=2, sort_keys=True).splitlines()
    return "\n".join(difflib.unified_diff(source_text, target_text, fromfile="IonSwapper", tofile="IonShuttler"))


def main() -> int:
    """Run the in-memory parity cases and return a process exit status.

    Returns:
        Zero when all observations match; one otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    source = _load_source_modules(args.ionswapper)
    differences: list[str] = []
    for case_name in ("local_pulse", "transport_required"):
        target_schedule, target_architecture, source_architecture, source_result = _case(case_name, source)
        for method in (SADDMethod.PULSE_ONLY, SADDMethod.FULL):
            source_output = source["run"](
                source_result,
                source_architecture,
                source["ControlCentricDDConfig"](
                    allow_transport=method.allow_transport,
                    max_accepted_windows=1,
                ),
            )
            target_output = run_sadd(
                target_schedule,
                target_architecture,
                method,
                SADDConfig(max_accepted_windows=1),
            )
            source_observation = _observation(
                _actions(source_output.result.path),
                _opportunities(source_output.records),
                strict_actions=args.strict_actions,
            )
            target_observation = _observation(
                _actions(target_output.schedule.path),
                _opportunities(target_output.report.opportunities),
                strict_actions=args.strict_actions,
            )
            if source_observation != target_observation:
                differences.append(f"{case_name}/{method.value}\n{_diff(source_observation, target_observation)}")

    if differences:
        logger.error("\n\n%s", "\n\n".join(differences))
        return 1
    logger.info("SADD observations match IonSwapper baseline %s.", EXPECTED_IONSWAPPER_COMMIT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
