"""
模拟器协议：统一的响应结构与解析。client 与 mock 都产出/消费同一 Response，
使运行时对“真实模拟器”与“本地 mock”无感切换。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..domain.types import ObservationType


@dataclass
class EnterResponse:
    accepted: bool
    virtual_time_s: float = 0.0
    max_virtual_duration_s: float = 360000.0
    max_real_duration_s: float = 1200.0
    remaining_real_duration_s: int = 1200


@dataclass
class MeasureResponse:
    accepted: bool
    virtual_time_s: float = 0.0
    measure_result: Optional[str] = None    # "no_signal" | "near" | "direction"
    svd_deg: Optional[float] = None

    def to_observation_type(self) -> ObservationType:
        return {
            "direction": ObservationType.BEARING,
            "near": ObservationType.TOO_STRONG,
            "no_signal": ObservationType.NO_SIGNAL,
        }[self.measure_result]


@dataclass
class ClearResponse:
    accepted: bool
    virtual_time_s: float = 0.0
    clear_result: Optional[str] = None      # "success" | "no_target_in_range"

    def is_success(self) -> bool:
        return self.clear_result == "success"


@dataclass
class ExitResponse:
    accepted: bool
    virtual_time_s: float = 0.0
    exit_reason: Optional[str] = None
