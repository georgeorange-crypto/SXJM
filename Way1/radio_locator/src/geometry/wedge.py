"""
楔形（bearing wedge）：一次示向度观测确定的 ±δ 角域。

提供点判定与 box 判定（保守），供 Q1 半平面裁剪与 set-membership quadtree 共用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .angle import deg2rad, wrap_pi
from .box_distance import Box, bearing_interval
from .halfplane import HalfPlane, wedge_halfplanes
from .vector import Vec2


@dataclass(frozen=True)
class Wedge:
    """以 anchor 为顶点、中线方位 bearing_deg、半张角 half_deg 的楔形。"""
    anchor: Vec2
    bearing_deg: float
    half_deg: float = 1.0

    def halfplanes(self) -> tuple[HalfPlane, HalfPlane]:
        return wedge_halfplanes(self.anchor, self.bearing_deg, self.half_deg)

    def contains_point(self, p: Vec2, eps: float = 1e-9) -> bool:
        dx = p[0] - self.anchor[0]
        dy = p[1] - self.anchor[1]
        if dx == 0.0 and dy == 0.0:
            return True
        ang = math.atan2(dy, dx)
        diff = abs(wrap_pi(ang - deg2rad(self.bearing_deg)))
        return diff <= deg2rad(self.half_deg) + eps


def box_wedge_relation(w: Wedge, b: Box) -> str:
    """
    box 与楔形的关系（保守）：
      "OUTSIDE"  —— box 整体在楔形外（可安全删除）
      "INSIDE"   —— box 整体在楔形内
      "UNKNOWN"  —— 相交/不确定（需细分）
    判据：比较 box 相对 anchor 的方位角区间与楔形角区间的最小夹角。
    """
    lo, hi, contains = bearing_interval(w.anchor, b)
    if contains:
        # box 含顶点，任意方向都可能 → 不确定
        return "UNKNOWN"
    half = deg2rad(w.half_deg)
    center = deg2rad(w.bearing_deg)
    # box 方位角区间 [lo,hi]（逆时针弧）。计算该弧上任一角到 center 的最小/最大夹角。
    # 采样端点足够判定（弧 < π，单峰）：
    d_lo = abs(wrap_pi(lo - center))
    d_hi = abs(wrap_pi(hi - center))
    # 弧是否跨过 center（即 center 落在 [lo,hi] 弧内）
    from .angle import angle_in_arc
    center_in = angle_in_arc(center, lo, hi)
    min_diff = 0.0 if center_in else min(d_lo, d_hi)
    max_diff = max(d_lo, d_hi)
    if min_diff > half + 1e-12:
        return "OUTSIDE"
    if max_diff <= half + 1e-12:
        return "INSIDE"
    return "UNKNOWN"
