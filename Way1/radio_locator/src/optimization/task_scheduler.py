"""
任务调度器：把“发现 / 定位 / 清除”三类任务在线编排成低总时间的动作序列。

在线组合优化（无未来信息，事件驱动）：
  状态 = WorldState（各频道 status + 可行集）。每步在候选动作集中选【单位时间收益最高】者：
    - COVER   ：去未探测覆盖点 measure（发现新源 / 排除频道）。
    - LOCALIZE：对 DETECTED/LOCALIZING 频道执行 Minimax 测点（缩小残差）。
    - CLEAR   ：对 READY_TO_CLEAR 频道前往外包络圆心清除。
  切频成本、移动成本进入评分；同点连测被禁止（信息零增益）。

设计取向：正确性由 set_estimation 保证；调度只影响【效率】。故此处启发式可自由迭代，
不承担“绝不误清/误删”的责任——那由可行集与清除证书独立保证。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from ..domain.types import ChannelStatus, ProblemConstants, RobotConstants, Vec2
from ..domain.world_state import WorldState
from .robust_lookahead import lookahead_channel
from .two_opt import solve_route


class TaskKind(Enum):
    COVER = "cover"
    LOCALIZE = "localize"
    CLEAR = "clear"


@dataclass
class Task:
    kind: TaskKind
    location: Vec2
    channel: Optional[int]
    score: float
    reason: str = ""


@dataclass
class Scheduler:
    problem: ProblemConstants
    robot: RobotConstants
    lam: float = 1.0
    half_deg: float = 1.0
    max_candidates: int = 48

    def _switch_cost(self, world: WorldState, channel: Optional[int]) -> float:
        if channel is None or channel == world.current_channel:
            return 0.0
        return self.robot.switch_time

    # ---------- CLEAR 优先（已就绪的直接清）----------
    def ready_clears(self, world: WorldState) -> List[Task]:
        tasks: List[Task] = []
        for c, cs in world.channels.items():
            if cs.status == ChannelStatus.READY_TO_CLEAR and cs.feasible_set is not None:
                fp = cs.feasible_set
                pt = fp.clear_point()
                if pt is not None:
                    move = math.hypot(pt[0] - world.position[0], pt[1] - world.position[1]) / self.robot.speed
                    tasks.append(Task(TaskKind.CLEAR, pt, c,
                                      score=move + self.robot.clear_hit_time,
                                      reason=f"ready MEC {fp.mec_radius:.2f}m"))
        return tasks

    # ---------- LOCALIZE（对已检测未就绪频道）----------
    def localize_candidates(self, world: WorldState) -> List[Task]:
        tasks: List[Task] = []
        for c, cs in world.channels.items():
            if cs.status in (ChannelStatus.DETECTED, ChannelStatus.LOCALIZING) and cs.feasible_set is not None:
                fp = cs.feasible_set
                if fp.is_empty():
                    continue
                pts = list(fp.representative_points())
                # 已测 apex：重测同点对方位约束零信息增益（示向度确定性复现）。
                # 从候选中剔除，强制换点取视差——收缩定向源狭长带的唯一途径。
                visited_apex = [o.position for o in cs.observations]
                res = lookahead_channel(pts, self.problem, self.robot, world.position,
                                        lam=self.lam, half_deg=self.half_deg,
                                        max_candidates=self.max_candidates,
                                        exclude_points=visited_apex)
                if res is None:
                    continue
                dec = self._best_measure_point(pts, world, exclude_points=visited_apex)
                if dec is None:
                    continue
                loc, add = dec
                score = res.objective + self._switch_cost(world, c)
                tasks.append(Task(TaskKind.LOCALIZE, loc, c, score=score,
                                  reason=res and f"residual→{res.residual_after:.1f}m, ~{res.expected_measurements_left} left"))
        return tasks

    def _best_measure_point(self, pts: Sequence[Vec2], world: WorldState,
                            exclude_points: Optional[Sequence[Vec2]] = None) -> Optional[Tuple[Vec2, float]]:
        from ..localization.second_viewpoint import choose_second_viewpoint
        if not pts:
            return None
        dec = choose_second_viewpoint(pts, self.problem, self.robot, world.position,
                                      half_deg=self.half_deg, lam=self.lam,
                                      max_candidates=self.max_candidates,
                                      exclude_points=exclude_points)
        return dec.best, dec.evaluation.worst_residual

    # ---------- COVER（发现新源）----------
    def cover_candidates(self, world: WorldState, cover_points: Sequence[Vec2],
                         visited: Sequence[Vec2]) -> List[Task]:
        tasks: List[Task] = []
        vis = set((round(p[0], 3), round(p[1], 3)) for p in visited)
        for p in cover_points:
            key = (round(p[0], 3), round(p[1], 3))
            if key in vis:
                continue
            move = math.hypot(p[0] - world.position[0], p[1] - world.position[1]) / self.robot.speed
            tasks.append(Task(TaskKind.COVER, p, None,
                              score=move + self.robot.detection_time,
                              reason="explore uncovered point"))
        return tasks

    # ---------- 主选择 ----------
    def next_task(self, world: WorldState, cover_points: Sequence[Vec2],
                  visited: Sequence[Vec2]) -> Optional[Task]:
        """
        优先级：先清（消灭确定任务，减少后续状态）→ 再定位（把 DETECTED 收敛）→ 最后探索。
        每类内部按 score（时间代价 / 目标值）取最小。
        """
        clears = self.ready_clears(world)
        if clears:
            return min(clears, key=lambda t: t.score)
        locs = self.localize_candidates(world)
        covers = self.cover_candidates(world, cover_points, visited)
        pool = locs + covers
        if not pool:
            return None
        return min(pool, key=lambda t: t.score)

    def order_cover_route(self, start: Vec2, cover_points: Sequence[Vec2],
                          held_karp_max: int = 16) -> List[Vec2]:
        """对覆盖点求一条低成本访问顺序（start 在首）。"""
        nodes = [start] + list(cover_points)
        _, order = solve_route(nodes, start_index=0, held_karp_max=held_karp_max)
        return [nodes[i] for i in order[1:]]
