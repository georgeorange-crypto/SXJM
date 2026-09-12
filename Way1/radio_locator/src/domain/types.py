"""
公共类型与题目常量的集中定义。

ProblemConstants / RobotConstants 从 config 载入，全程只读传递，
避免把 1800、1000、20m 等硬编码散落各处。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

Vec2 = Tuple[float, float]


class ObservationType(Enum):
    """一次动作的结果类型（对齐模拟器返回码）。"""
    BEARING = "direction"          # measure_result="direction"，带 svd_deg
    NO_SIGNAL = "no_signal"        # measure_result="no_signal"
    TOO_STRONG = "near"            # measure_result="near"，≤5m 在覆盖角内
    CLEAR_SUCCESS = "success"      # clear_result="success"
    CLEAR_FAILURE = "no_target_in_range"  # clear_result="no_target_in_range"


class ChannelStatus(Enum):
    UNKNOWN = "unknown"            # 尚未在任一覆盖点检测过
    DETECTED = "detected"         # 检测到信号，进入定位
    LOCALIZING = "localizing"     # 定位中（MEC 仍 > 阈值）
    READY_TO_CLEAR = "ready"      # MEC <= 阈值，可清除
    CLEARED = "cleared"           # /clear success（唯一“已清”判据）
    ABSENT = "absent"             # 已证明无源


class SourceType(Enum):
    OMNI = "omni"
    DIRECTIONAL = "directional"


@dataclass(frozen=True)
class ProblemConstants:
    area_radius: float = 1800.0
    channels: int = 20
    source_count_min: int = 10
    source_count_max: int = 16
    bearing_error_deg: float = 1.0
    # 示向度报告量化 + 数值安全裕量：楔形半张角在物理误差界之上再放宽 margin，
    # 保证真源【绝不】被裁出可行域（模拟器把 svd 量化到 0.01°，实机精度未知，
    # 故留 0.1° 裕量；代价仅使可行带略宽，属【效率】范畴，不损正确性）。
    bearing_margin_deg: float = 0.1
    receive_radius_min: float = 1000.0
    receive_radius_max: float = 1500.0
    directional_half_angle_deg: float = 90.0

    @property
    def wedge_half_deg(self) -> float:
        """集员估计/裁剪应使用的楔形半张角 = 物理误差界 + 报告量化裕量。"""
        return self.bearing_error_deg + self.bearing_margin_deg


@dataclass(frozen=True)
class RobotConstants:
    speed: float = 5.0
    switch_time: float = 1.0
    detection_time: float = 5.0
    clear_hit_time: float = 5.0
    clear_miss_time: float = 3.0
    optical_radius: float = 20.0
    too_strong_radius: float = 5.0
    clear_mec_radius: float = 19.0
    radius_delete_margin: float = 1e-3
