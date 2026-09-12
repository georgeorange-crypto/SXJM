"""
主动定位循环：把一个频道的可行集从“检测到”驱动到“可清除”（MEC<=19m）。

给定当前可行集（Q3 或 Q4 组合集）与机器人位姿，产出下一步【测点建议】：
  1. 若可行集已 ready_to_clear → 返回 CLEAR 动作（前往外包络圆心）。
  2. 否则用 Minimax 选下一测点（second_viewpoint），返回 MEASURE 动作。
  3. 内置停滞检测：同一点重复测量无新信息（误差按地点固定），故绝不在原地重复；
     若候选全部不降残差，扩大候选环（recovery 由上层 controller 处理）。

本模块不直接调用模拟器，只做“看着当前信息决定下一步”的纯函数，便于测试与回放。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

from ..domain.types import ProblemConstants, RobotConstants, Vec2
from .second_viewpoint import ViewpointDecision, choose_second_viewpoint


class ActionKind(Enum):
    MEASURE = "measure"
    CLEAR = "clear"
    DONE = "done"


@dataclass
class LocalizationAction:
    kind: ActionKind
    location: Optional[Vec2]
    channel: int
    reason: str = ""
    decision: Optional[ViewpointDecision] = None


class ActiveLocalizer:
    def __init__(self, problem: ProblemConstants, robot: RobotConstants,
                 half_deg: float = 1.0, lam: float = 1.0, max_candidates: int = 48):
        self.problem = problem
        self.robot = robot
        self.half_deg = half_deg
        self.lam = lam
        self.max_candidates = max_candidates

    def next_action(self, feasible, channel: int, robot_pos: Vec2,
                    first_viewpoint: Optional[Vec2] = None,
                    measured_points: Optional[Sequence[Vec2]] = None) -> LocalizationAction:
        """
        feasible：具备 is_empty/ready_to_clear/clear_point/representative_points 接口的可行集。
        measured_points：已测过的点（避免重复选到同点）。
        """
        if feasible.is_empty():
            return LocalizationAction(ActionKind.DONE, None, channel,
                                      reason="feasible set empty (hypothesis refuted)")
        if feasible.ready_to_clear():
            return LocalizationAction(ActionKind.CLEAR, feasible.clear_point(), channel,
                                      reason=f"MEC {feasible.mec_radius:.2f}m <= threshold")

        points = list(feasible.representative_points())
        decision = choose_second_viewpoint(
            points, self.problem, self.robot, robot_pos,
            first_viewpoint=first_viewpoint, half_deg=self.half_deg,
            lam=self.lam, max_candidates=self.max_candidates)

        target = decision.best
        # 停滞防护：若最优点与某已测点几乎重合，选次优的“新”点
        if measured_points:
            target = self._avoid_repeat(decision, measured_points)

        return LocalizationAction(ActionKind.MEASURE, target, channel,
                                  reason=f"minimax worst residual {decision.evaluation.worst_residual:.1f}m",
                                  decision=decision)

    @staticmethod
    def _avoid_repeat(decision: ViewpointDecision, measured: Sequence[Vec2],
                      tol: float = 2.0) -> Vec2:
        def is_new(p: Vec2) -> bool:
            return all(math.hypot(p[0] - m[0], p[1] - m[1]) > tol for m in measured)
        if is_new(decision.best):
            return decision.best
        for e in sorted(decision.all_evals, key=lambda e: e.score):
            if is_new(e.location):
                return e.location
        return decision.best  # 全部重合（极端）——交由上层 recovery
