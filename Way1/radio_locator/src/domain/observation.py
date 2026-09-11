"""
单次观测记录。所有 set-membership 约束都从 Observation 派生。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .types import ObservationType, Vec2


@dataclass(frozen=True)
class Observation:
    position: Vec2
    channel: int
    result: ObservationType
    bearing_deg: Optional[float] = None   # 仅 BEARING 有值（含 ±1° 误差）
    virtual_time: float = 0.0
    step: int = 0

    @property
    def bearing_rad(self) -> Optional[float]:
        if self.bearing_deg is None:
            return None
        return math.radians(self.bearing_deg)

    def is_positive(self) -> bool:
        """正观测：收到该频道信号（BEARING 或 TOO_STRONG）。"""
        return self.result in (ObservationType.BEARING, ObservationType.TOO_STRONG)

    def is_negative(self) -> bool:
        """负观测：该频道无信号。"""
        return self.result == ObservationType.NO_SIGNAL
