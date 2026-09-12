"""
Q2 主动感知：Minimax 第二观测点选择（鲁棒最坏情形）。

问题：已在 S₁ 得到一次示向度（楔形），可行集 Ω 仍是狭长带。下一次从哪测？
目标（Minimax）：
  S* = argmin_{S∈候选} max_{o∈可能结果} [ 代价(S) + R_MEC( Ω ∩ C(S,o) ) ]
即在【最坏结果】下仍把剩余不确定性（外包络 MEC 半径）压到最小；代价项计入移动/测量
时间，实现 cost-to-go 权衡（lookahead_lambda 调权）。

结果分支（一等公民，不只 bearing）：
  - BEARING：源在从 S 出发、宽 2δ 的楔形内（对每个假设真位置取【对抗误差】方位）。
  - NO_SIGNAL：源在 S 的射程外（保守用 Rmin：剔除 ‖x−S‖<=Rmin 的点）。
  - TOO_STRONG(near)：源在 S 的 5m 内。

正确性说明：本模块只决定“去哪测”，属【效率】范畴。真正的 Ω 更新由 set_estimation
的保守四叉树完成，与此处选点无关——故有限场景近似绝不损害正确性（§Q2 审核结论）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from ..domain.types import ProblemConstants, RobotConstants, Vec2
from ..geometry.angle import deg2rad, wrap_pi
from ..geometry.mec import enclosing_radius
from ..geometry.wedge import Wedge

# 选点评分用点云上限：_worst_bearing_outcome 是 O(n²)，全云可达数百点。
# 选点属【效率层】（见模块 docstring：有限场景近似绝不损害正确性），故对【评分】
# 用的点云下采样到此上限——真源过滤仍用全分辨率四叉树，不受影响。
_SCORING_CAP = 96


@dataclass
class ViewpointEval:
    location: Vec2
    worst_residual: float          # 最坏结果下的 MEC 残差半径
    worst_outcome: str
    move_cost: float
    score: float                   # worst_residual + λ·move_cost


@dataclass
class ViewpointDecision:
    best: Vec2
    evaluation: ViewpointEval
    all_evals: List[ViewpointEval]


def _subsample(points: Sequence[Vec2], cap: int) -> List[Vec2]:
    """等距下采样到至多 cap 个点（仅用于选点评分的启发式，不影响正确性）。"""
    n = len(points)
    if n <= cap:
        return list(points)
    step = n / float(cap)
    return [points[int(i * step)] for i in range(cap)]


# ---------- 结果分支的点云过滤（快速近似，仅用于选点）----------
def _bearing_survivors(points: Sequence[Vec2], S: Vec2, bearing_deg: float,
                       half_deg: float) -> List[Vec2]:
    w = Wedge(S, bearing_deg, half_deg)
    return [p for p in points if w.contains_point(p)]


def _nosignal_survivors(points: Sequence[Vec2], S: Vec2, rmin: float) -> List[Vec2]:
    return [p for p in points if math.hypot(p[0] - S[0], p[1] - S[1]) > rmin]


def _near_survivors(points: Sequence[Vec2], S: Vec2, near_r: float) -> List[Vec2]:
    return [p for p in points if math.hypot(p[0] - S[0], p[1] - S[1]) <= near_r]


def _worst_bearing_outcome(points: Sequence[Vec2], S: Vec2, half_deg: float) -> float:
    """
    对 bearing 结果的最坏情形：真源可能是 Ω 中任一点 x_h，其真方位 ∠(x_h−S)。
    对抗误差把楔形中心推到 ±δ 最不利处——但楔形宽度固定，最坏残差由“哪个 x_h 的楔形
    切出的存留点最分散”决定。这里对每个 x_h 评估其楔形存留点 MEC，取 max。

    实现：把每个点相对 S 的方位角【预计算一次】（原 O(n²) 次 atan2 降到 O(n)），
    然后成对比较角差。语义与 Wedge.contains_point 完全一致（含顶点恒真的特例）。
    """
    n = len(points)
    if n == 0:
        return 0.0
    sx, sy = S
    ang: List[float] = []
    anchor: List[bool] = []
    for p in points:
        dx = p[0] - sx
        dy = p[1] - sy
        if dx == 0.0 and dy == 0.0:
            ang.append(0.0)
            anchor.append(True)
        else:
            ang.append(math.atan2(dy, dx))
            anchor.append(False)
    half_rad = deg2rad(half_deg) + 1e-9
    worst = 0.0
    for h in range(n):
        center = ang[h]
        surv = [points[i] for i in range(n)
                if anchor[i] or abs(wrap_pi(ang[i] - center)) <= half_rad]
        r = enclosing_radius(surv) if surv else 0.0
        if r > worst:
            worst = r
    return worst


def evaluate_viewpoint(points: Sequence[Vec2], S: Vec2, problem: ProblemConstants,
                       robot: RobotConstants, from_pos: Vec2,
                       half_deg: float = 1.0, lam: float = 1.0,
                       consider_negative: bool = True) -> ViewpointEval:
    """
    评估在 S 处测量的最坏残差（Minimax 内层 max）。points 为 Ω 的代表点云。
    """
    branches: List[Tuple[str, float]] = []

    # BEARING 分支（最坏方位）
    r_bear = _worst_bearing_outcome(points, S, half_deg)
    branches.append(("BEARING", r_bear))

    # NO_SIGNAL 分支：仅当存在“可能收不到”的假设（有点在 Rmin 外，或 R_eff 可小）
    if consider_negative:
        surv = _nosignal_survivors(points, S, problem.receive_radius_min)
        if surv:
            branches.append(("NO_SIGNAL", enclosing_radius(surv)))

    # TOO_STRONG 分支：仅当有点在 near 半径内才可能触发
    near_surv = _near_survivors(points, S, robot.too_strong_radius)
    if near_surv:
        branches.append(("TOO_STRONG", enclosing_radius(near_surv)))

    worst_outcome, worst_residual = max(branches, key=lambda t: t[1])
    move_cost = math.hypot(S[0] - from_pos[0], S[1] - from_pos[1]) / robot.speed
    score = worst_residual + lam * move_cost
    return ViewpointEval(location=S, worst_residual=worst_residual,
                         worst_outcome=worst_outcome, move_cost=move_cost, score=score)


def candidate_locations(points: Sequence[Vec2], problem: ProblemConstants,
                        n_angles: int = 12, radii: Optional[Sequence[float]] = None,
                        include_perpendicular_from: Optional[Vec2] = None,
                        max_candidates: int = 48,
                        exclude_points: Optional[Sequence[Vec2]] = None,
                        exclude_tol: float = 20.0) -> List[Vec2]:
    """
    生成候选测点：以 Ω 质心为中心的多环多角布点（限制在竞技场内）。
    可选注入“相对首测点的垂直基线方向”候选（三角定位最优基线的先验）。
    exclude_points：已测过的测点（apex）——重复同一 apex 测方位【零信息增益】
      （示向度确定性复现，楔形不变），故在 exclude_tol 内的候选一律剔除，
      强制换点取得视差（parallax），这才是收缩定向源狭长带的唯一途径。
    """
    if not points:
        return [(0.0, 0.0)]
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    spread = max((math.hypot(p[0] - cx, p[1] - cy) for p in points), default=100.0)
    if radii is None:
        # 环半径随 spread 自适应：区域大时远距三角定位，区域小时贴近以触发 near/紧交会。
        # 不再对内环设 50m 地板，否则区域收紧后永远测不到近处，无法收敛到 clear 证书。
        base = max(spread, 1.0)
        radii = [base * 0.25, base * 0.5, base, base * 2.0]
    arena = problem.area_radius
    cand: List[Vec2] = []
    # 利用候选：质心即当前最优估计。落在真源 5m 内可直接触发 near 塌缩到证书内。
    if cx * cx + cy * cy <= arena * arena:
        cand.append((cx, cy))
    for rr in radii:
        for k in range(n_angles):
            ang = 2.0 * math.pi * k / n_angles
            x, y = cx + rr * math.cos(ang), cy + rr * math.sin(ang)
            if x * x + y * y <= arena * arena:
                cand.append((x, y))
    if include_perpendicular_from is not None:
        sx, sy = include_perpendicular_from
        vx, vy = cx - sx, cy - sy
        norm = math.hypot(vx, vy)
        if norm > 1e-9:
            px, py = -vy / norm, vx / norm
            for rr in radii:
                for sgn in (+1.0, -1.0):
                    x, y = cx + sgn * px * rr, cy + sgn * py * rr
                    if x * x + y * y <= arena * arena:
                        cand.append((x, y))

    def _too_close(c: Vec2) -> bool:
        if not exclude_points:
            return False
        return any(math.hypot(c[0] - e[0], c[1] - e[1]) <= exclude_tol for e in exclude_points)

    # 去重 + 剔除已测 apex + 截断
    seen = set()
    out: List[Vec2] = []
    for c in cand:
        key = (round(c[0], 3), round(c[1], 3))
        if key in seen or _too_close(c):
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_candidates:
            break
    # 全被剔除的兜底：区域仍大但候选恰好都贴着旧 apex——放宽约束，宁可重测也不空手。
    if not out:
        for c in cand:
            key = (round(c[0], 3), round(c[1], 3))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= max_candidates:
                break
    return out


def choose_second_viewpoint(points: Sequence[Vec2], problem: ProblemConstants,
                            robot: RobotConstants, from_pos: Vec2,
                            first_viewpoint: Optional[Vec2] = None,
                            half_deg: float = 1.0, lam: float = 1.0,
                            max_candidates: int = 48,
                            exclude_points: Optional[Sequence[Vec2]] = None) -> ViewpointDecision:
    """
    Minimax 选点主入口：返回最优测点及全部候选评估。
    exclude_points：已测 apex——从候选中剔除（重测同点零信息增益），强制取得视差基线。
    """
    cands = candidate_locations(points, problem,
                                include_perpendicular_from=first_viewpoint,
                                max_candidates=max_candidates,
                                exclude_points=exclude_points)
    # 评分点云下采样：_worst_bearing_outcome 是 O(n²)，对每个候选都跑一遍。
    # 候选生成用全云（质心/展布更准），评分用下采样云（仅启发式，不影响正确性）。
    scoring_pts = _subsample(points, _SCORING_CAP)
    evals = [evaluate_viewpoint(scoring_pts, S, problem, robot, from_pos, half_deg, lam)
             for S in cands]
    best_eval = min(evals, key=lambda e: e.score)
    return ViewpointDecision(best=best_eval.location, evaluation=best_eval, all_evals=evals)
