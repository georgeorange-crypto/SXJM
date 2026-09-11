"""
Q3 覆盖布点：用最少的探测圆盘覆盖整个竞技场，保证任何全向源被至少一点探测到。

覆盖判据：源在半径 R_eff>=Rmin=1000 的圆内才可能被收到。为保证【必被探测】，
用保守接收半径 r_safe（默认 980，留裕量）。若一组探测点 {pk} 满足
竞技场内任意点到最近 pk 的距离 <= r_safe，则任何源都落在某 pk 的 r_safe 内，
在该点 measure 必收到信号。

几何构造（中心 + 单层正 m 边形环）：
  竞技场半径 R_area=1800。中心点 1 个 + 外环 m 个（默认 m=6，正六边形）。
  环半径 ρ 的选取：使“最难覆盖点”（竞技场边界、两相邻环点角平分线方向）到
  最近探测点距离 <= r_safe。对 m=6，ρ* 可解析求得（见 ring_radius_for）。
  当单环不足以覆盖 1800（r_safe 偏小）时，退化为规则三角网格布点（fallback）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from ..domain.types import Vec2


@dataclass(frozen=True)
class CoverPlan:
    points: List[Vec2]
    safe_radius: float
    covers_arena: bool


def _farthest_uncovered_distance(points: List[Vec2], arena_r: float, samples: int = 720) -> float:
    """
    数值估计：竞技场内到最近探测点的最大距离（覆盖半径需 >= 此值）。
    采样边界 + 若干内部环，取上确界近似（边界通常是最难覆盖处）。
    """
    worst = 0.0
    rings = [arena_r, arena_r * 0.98, arena_r * 0.85, arena_r * 0.6, arena_r * 0.3, 0.0]
    for rr in rings:
        n = max(1, int(samples * (rr / arena_r))) if rr > 0 else 1
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            x, y = rr * math.cos(ang), rr * math.sin(ang)
            dmin = min(math.hypot(x - px, y - py) for px, py in points)
            if dmin > worst:
                worst = dmin
    return worst


def ring_radius_for(arena_r: float, r_safe: float, m: int) -> float:
    """
    单层 m 边形环 + 中心 的最优环半径（解析近似）。
    两个最难点：
      (a) 边界上相邻两环点角平分线方向的点 B=(arena_r,0) 经旋转 → 距最近环点。
      (b) 中心到环点中途。
    取使 max(dist(B, 最近环点), 覆盖中心盘) 最小的 ρ。这里给出对 m=6 的稳健解析式，
    其余 m 用数值搜索兜底。
    """
    if m <= 0:
        return 0.0
    half = math.pi / m
    # 令边界最难点被环点覆盖：环点到边界最难点距离 == r_safe
    # 最难边界点在角平分线上，环点在 0 角：dist^2 = arena_r^2 + ρ^2 - 2 arena_r ρ cos(half)
    # 解 ρ：ρ = arena_r cos(half) - sqrt(r_safe^2 - arena_r^2 sin^2(half))
    disc = r_safe * r_safe - (arena_r * math.sin(half)) ** 2
    if disc < 0:
        # r_safe 太小，单环无法覆盖到边界 → 取能覆盖中心盘的最大 ρ
        return max(0.0, r_safe)
    rho = arena_r * math.cos(half) - math.sqrt(disc)
    return max(0.0, rho)


def hex_cover(arena_r: float, r_safe: float, m: int = 6,
              ring_radius: float = None) -> CoverPlan:
    """
    中心 + 单层 m 边形环 的覆盖布点。若单环不足覆盖，则回退三角网格。
    """
    rho = ring_radius if ring_radius is not None else ring_radius_for(arena_r, r_safe, m)
    points: List[Vec2] = [(0.0, 0.0)]
    for k in range(m):
        ang = 2.0 * math.pi * k / m
        points.append((rho * math.cos(ang), rho * math.sin(ang)))

    worst = _farthest_uncovered_distance(points, arena_r)
    if worst <= r_safe + 1e-6:
        return CoverPlan(points=points, safe_radius=r_safe, covers_arena=True)

    # 回退：正三角网格布点（间距 s 使覆盖半径 s/√3 <= r_safe）
    grid = triangular_grid_cover(arena_r, r_safe)
    worst2 = _farthest_uncovered_distance(grid, arena_r)
    return CoverPlan(points=grid, safe_radius=r_safe, covers_arena=worst2 <= r_safe + 1e-6)


def triangular_grid_cover(arena_r: float, r_safe: float) -> List[Vec2]:
    """
    正三角（hex packing）网格布点：覆盖半径 = s/√3，取 s = r_safe*√3*0.98 留裕量。
    仅保留落在竞技场（含边界外 r_safe 环带内、以覆盖边界）的点。
    """
    s = r_safe * math.sqrt(3.0) * 0.98
    dy = s * math.sqrt(3.0) / 2.0
    pts: List[Vec2] = []
    j = 0
    y = -arena_r - s
    while y <= arena_r + s:
        offset = 0.0 if (j % 2 == 0) else s / 2.0
        x = -arena_r - s
        while x <= arena_r + s:
            px = x + offset
            if px * px + y * y <= (arena_r + r_safe) ** 2:
                pts.append((px, y))
            x += s
        y += dy
        j += 1
    return pts
