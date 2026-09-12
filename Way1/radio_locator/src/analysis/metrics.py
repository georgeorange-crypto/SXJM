"""
指标计算：从 controller 的 StepLog / WorldState 提炼评审关注的量化指标。

时间分解严格对齐题目目标：
  T = Σ move(d/5) + Σ switch(1·[换频]) + 5·N_measure + Σ_clear(5 命中 / 3 未命中)。
本模块把 log 复算一遍，交叉验证 world.virtual_time（防止账本漂移）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..domain.types import RobotConstants
from ..domain.world_state import WorldState


@dataclass
class TimeBreakdown:
    move: float = 0.0
    switch: float = 0.0
    detection: float = 0.0
    clear_hit: float = 0.0
    clear_miss: float = 0.0

    @property
    def total(self) -> float:
        return self.move + self.switch + self.detection + self.clear_hit + self.clear_miss


@dataclass
class Metrics:
    total_time: float
    breakdown: TimeBreakdown
    n_measure: int
    n_clear_success: int
    n_clear_miss: int
    cleared: int
    absent: int
    present: int
    steps: int
    ledger_consistent: bool


def compute_metrics(log: Sequence, world: WorldState, robot: RobotConstants) -> Metrics:
    """
    从 StepLog 序列复算时间分解。log 元素需有 .kind/.channel/.location/.note 字段。
    move/switch 需要相邻位姿；这里用 log 的 location 顺序重建轨迹。
    """
    bd = TimeBreakdown()
    n_measure = n_succ = n_miss = 0
    pos = (0.0, 0.0)
    channel = world.channels and 1
    cur_ch = 1

    for entry in log:
        loc = getattr(entry, "location", None)
        kind = getattr(entry, "kind", "")
        note = getattr(entry, "note", "")
        ch = getattr(entry, "channel", None)
        if loc is None:
            continue
        move = math.hypot(loc[0] - pos[0], loc[1] - pos[1]) / robot.speed
        if kind == "measure":
            if note == "rejected":
                continue
            bd.move += move
            if ch is not None and ch != cur_ch:
                bd.switch += robot.switch_time
                cur_ch = ch
            bd.detection += robot.detection_time
            n_measure += 1
            pos = loc
        elif kind == "clear":
            bd.move += move
            if note == "success":
                bd.clear_hit += robot.clear_hit_time
                n_succ += 1
            else:
                bd.clear_miss += robot.clear_miss_time
                n_miss += 1
            pos = loc
        # clear-blocked：无动作，不计时

    ledger_ok = abs(bd.total - world.virtual_time) <= 1e-3 + 1e-6 * max(1.0, world.virtual_time)
    return Metrics(total_time=bd.total, breakdown=bd, n_measure=n_measure,
                   n_clear_success=n_succ, n_clear_miss=n_miss,
                   cleared=world.cleared_count(), absent=world.confirmed_absent(),
                   present=world.confirmed_present(), steps=world.step,
                   ledger_consistent=ledger_ok)
