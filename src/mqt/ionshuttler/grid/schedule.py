# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Grid schedule persistence through the shared schedule format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.schedule import Schedule
from mqt.ionshuttler.grid.actions import decode_grid_action
from mqt.ionshuttler.grid.state import GridMachineState

if TYPE_CHECKING:
    from mqt.ionshuttler.core.actions import Action


def schedule_from_dict(data: object) -> Schedule[Action, GridMachineState]:
    """Restore a Grid schedule from its versioned representation.

    Returns:
        The restored schedule.
    """
    return Schedule.from_dict(data, decode_action=decode_grid_action, decode_state=GridMachineState.from_dict)


def schedule_from_json(raw: str) -> Schedule[Action, GridMachineState]:
    """Restore a Grid schedule from JSON text.

    Returns:
        The restored schedule.
    """
    return schedule_from_dict(json.loads(raw))


def load_schedule(filename: str | Path) -> Schedule[Action, GridMachineState]:
    """Load a Grid schedule from a UTF-8 JSON file.

    Returns:
        The restored schedule.
    """
    return schedule_from_json(Path(filename).read_text(encoding="utf-8"))


__all__ = ["load_schedule", "schedule_from_dict", "schedule_from_json"]
