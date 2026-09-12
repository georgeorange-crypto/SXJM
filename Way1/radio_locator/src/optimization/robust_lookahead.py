"""
鲁棒前瞻（receding-horizon）：把“下一步测量”的收益/代价做一步/多步展望评估。

与 second_viewpoint 的 Minimax 互补——second_viewpoint 只看单频道单步最坏残差；
本模块把【时间代价】纳入统一目标，评估“继续在当前频道多测一次”与“切换到下一任务”
的相对优先级，服务于 task_scheduler 的调度决策。

目标（cost-to-go 近似）：
  J(action) = 立即时间代价 + λ · 预期剩余不确定性（用最坏残差 MEC 近似）。
无概率分布，故“预期”一律取【最坏情形】上界（robust），不引入人为分布。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..domain.types import ProblemConstants, RobotConstants, Vec2
from ..localization.second_viewpoint import (ViewpointEval, choose_second_viewpoint,
                                             evaluate_viewpoint)


@dataclass
class LookaheadResult:
    measure_here_cost: float        # 在建议点测一次的时间代价
    residual_after: float           # 最坏残差（MEC 半径）
    expected_measurements_left: int # 达到清除阈值还需的粗略测量数（下界估计）
    objective: float                # measure_cost + λ·residual


def estimate_measurements_left(residual: float, threshold: float,
                               shrink_ratio: float = 0.5) -> int:
    """
    粗估还需多少次测量把残差降到 threshold（每次最坏按 shrink_ratio 收缩）。
    仅用于调度排序，不影响正确性。
    """
    if residual <= threshold:
        return 0
    if shrink_ratio <= 0.0 or shrink_ratio >= 1.0:
        return 1
    k = math.log(threshold / residual) / math.log(shrink_ratio)
    return max(1, int(math.ceil(k)))


def lookahead_channel(points: Sequence[Vec2], problem: ProblemConstants,
                      robot: RobotConstants, robot_pos: Vec2,
                      first_viewpoint: Optional[Vec2] = None,
                      lam: float = 1.0, half_deg: float = 1.0,
                      max_candidates: int = 48,
                      exclude_points: Optional[Sequence[Vec2]] = None) -> Optional[LookaheadResult]:
    """对单频道做一步前瞻，返回其调度评分。points 为该频道可行集代表点。
    exclude_points：该频道已测 apex——避免把“重测零增益点”当作可选动作。"""
    if not points:
        return None
    decision = choose_second_viewpoint(points, problem, robot, robot_pos,
                                        first_viewpoint=first_viewpoint,
                                        half_deg=half_deg, lam=lam,
                                        max_candidates=max_candidates,
                                        exclude_points=exclude_points)
    ev = decision.evaluation
    threshold = robot.clear_mec_radius
    left = estimate_measurements_left(ev.worst_residual, threshold)
    # 测量本身固定 detection_time + 可能切频；此处含移动
    measure_cost = ev.move_cost + robot.detection_time
    objective = measure_cost + lam * ev.worst_residual
    return LookaheadResult(measure_here_cost=measure_cost,
                           residual_after=ev.worst_residual,
                           expected_measurements_left=left,
                           objective=objective)
