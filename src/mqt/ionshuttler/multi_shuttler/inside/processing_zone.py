from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Edge


@dataclass
class ProcessingZone:
    name: str
    edge_idc: Edge
