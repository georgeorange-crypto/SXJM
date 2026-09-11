"""
点到轴对齐 box 的距离上下界，以及 box 相对某点的方位角区间。

set-membership 的 box 层判定全靠这些【保守】界：只要真实点在 box 内，
其真实距离/方位一定落在这里给出的 [min,max] 内。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from .angle import wrap_2pi
from .vector import Vec2


@dataclass(frozen=True)
class Box:
    xl: float
    yl: float
    xu: float
    yu: float

    @property
    def width(self) -> float:
        return self.xu - self.xl

    @property
    def height(self) -> float:
        return self.yu - self.yl

    @property
    def size(self) -> float:
        return max(self.width, self.height)

    @property
    def center(self) -> Vec2:
        return ((self.xl + self.xu) / 2.0, (self.yl + self.yu) / 2.0)

    def corners(self) -> Tuple[Vec2, Vec2, Vec2, Vec2]:
        return (
            (self.xl, self.yl),
            (self.xu, self.yl),
            (self.xu, self.yu),
            (self.xl, self.yu),
        )

    def contains_point(self, p: Vec2) -> bool:
        return self.xl <= p[0] <= self.xu and self.yl <= p[1] <= self.yu

    def split4(self) -> Tuple["Box", "Box", "Box", "Box"]:
        cx, cy = self.center
        return (
            Box(self.xl, self.yl, cx, cy),
            Box(cx, self.yl, self.xu, cy),
            Box(cx, cy, self.xu, self.yu),
            Box(self.xl, cy, cx, self.yu),
        )


def distance_min(s: Vec2, b: Box) -> float:
    """点 s 到 box 的最小距离（点在 box 内为 0）。"""
    dx = max(b.xl - s[0], 0.0, s[0] - b.xu)
    dy = max(b.yl - s[1], 0.0, s[1] - b.yu)
    return math.hypot(dx, dy)


def distance_max(s: Vec2, b: Box) -> float:
    """点 s 到 box 的最大距离（必为某个角点）。"""
    dx = max(abs(s[0] - b.xl), abs(s[0] - b.xu))
    dy = max(abs(s[1] - b.yl), abs(s[1] - b.yu))
    return math.hypot(dx, dy)


def bearing_interval(s: Vec2, b: Box) -> Tuple[float, float, bool]:
    """
    从点 s 观察 box 的方位角区间 arg(P - s), P∈box（弧度）。
    返回 (lo, hi, contains_s)：
      - contains_s=True 时 box 含 s，方位角覆盖整圆，(lo,hi) 无意义。
      - 否则区间为从 lo 逆时针到 hi 的圆弧（弧长 = wrap_2pi(hi-lo)）。

    做法：取四角方位角，找覆盖它们的最小弧。box 不含 s 时角跨度必 < π（凸集从外部看）。
    """
    if b.contains_point(s):
        return 0.0, 0.0, True
    angs = [wrap_2pi(math.atan2(c[1] - s[1], c[0] - s[0])) for c in b.corners()]
    # 找最小覆盖弧：排序后取最大间隙，弧的补即为覆盖区间
    angs_sorted = sorted(angs)
    n = len(angs_sorted)
    max_gap = -1.0
    gap_idx = 0
    for i in range(n):
        a = angs_sorted[i]
        nxt = angs_sorted[(i + 1) % n] + (2.0 * math.pi if i + 1 == n else 0.0)
        gap = nxt - a
        if gap > max_gap:
            max_gap = gap
            gap_idx = i
    # 覆盖区间从 gap 后的点开始，到 gap 前的点结束
    lo = angs_sorted[(gap_idx + 1) % n]
    hi = angs_sorted[gap_idx]
    return lo, hi, False
