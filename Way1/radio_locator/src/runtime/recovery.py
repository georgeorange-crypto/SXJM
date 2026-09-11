"""
异常恢复：把运行中可能出现的“异常但合法”情形转成安全动作，保证系统不卡死、不误判。

覆盖的情形：
  R1 可行集意外为空（正观测下）：多因数值/分辨率过粗。→ 以更细分辨率重建；仍空则
     放宽 delete_margin 重建（更保守，宁多勿漏），并降级该频道回 DETECTED 重测。
  R2 CLEAR miss（证书成立却未命中）：把“源不在该点 20m 内”并入约束，细化后重试；
     连续 miss 超阈值 → 换一个外包络采样点（防止圆心恰在盲区边界）。
  R3 选点停滞（候选全部不降残差）：扩大候选环 / 加入垂直基线点（recovery 提供新候选）。
  R4 时间预算紧张：切换到“只清高置信频道 + 跳过低价值探索”的保守收尾。

recovery 只产出建议，不直接操作模拟器——由 controller 决定采纳，便于测试。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

from ..domain.types import Vec2
from ..geometry.mec import min_enclosing_circle


class RecoveryKind(Enum):
    REFINE = "refine"                  # 更细分辨率重建
    LOOSEN_MARGIN = "loosen_margin"    # 放宽删除裕量重建
    REMEASURE = "remeasure"            # 降级重测
    ALT_CLEAR_POINT = "alt_clear_point"  # 换清除点
    EXPAND_CANDIDATES = "expand_candidates"
    CONSERVATIVE_FINISH = "conservative_finish"


@dataclass
class RecoveryAdvice:
    kind: RecoveryKind
    location: Optional[Vec2] = None
    param: Optional[float] = None
    reason: str = ""


def on_empty_feasible(current_min_size: float, current_margin: float) -> RecoveryAdvice:
    """R1：正观测下可行集为空。先细化；仍空由 controller 再调用 loosen。"""
    if current_min_size > 0.5:
        return RecoveryAdvice(RecoveryKind.REFINE, param=max(0.5, current_min_size / 4.0),
                              reason="empty set: refine resolution")
    return RecoveryAdvice(RecoveryKind.LOOSEN_MARGIN, param=current_margin * 10.0,
                          reason="empty set at fine resolution: loosen delete margin")


def alt_clear_point(envelope_points: Sequence[Vec2], tried: Sequence[Vec2],
                    tol: float = 1.0) -> RecoveryAdvice:
    """R2：换一个清除点。优先取外包络 MEC 圆心；若已试过，取包络点的重心方向偏移点。"""
    if not envelope_points:
        return RecoveryAdvice(RecoveryKind.REMEASURE, reason="no envelope to clear")
    circle = min_enclosing_circle(list(envelope_points))
    center = circle.center
    if all(math.hypot(center[0] - t[0], center[1] - t[1]) > tol for t in tried):
        return RecoveryAdvice(RecoveryKind.ALT_CLEAR_POINT, location=center,
                              reason="use envelope MEC center")
    # 取离已试点最远的包络代表点作为备选
    best = max(envelope_points,
               key=lambda p: min(math.hypot(p[0] - t[0], p[1] - t[1]) for t in tried))
    return RecoveryAdvice(RecoveryKind.ALT_CLEAR_POINT, location=best,
                          reason="alternate envelope point")


def should_conservative_finish(remaining_real_s: float, threshold_s: float = 60.0) -> bool:
    """R4：现实时间预算不足 → 进入保守收尾。"""
    return remaining_real_s <= threshold_s
