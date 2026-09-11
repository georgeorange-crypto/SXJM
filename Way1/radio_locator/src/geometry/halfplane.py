"""
半平面表示与楔形（wedge）构造。

半平面用 { P : cross(dir, P - anchor) >= 0 } 表示（dir 左侧，含边界）。
采用“过点 + 方向”的形式而非 ax+by+c<=0，便于直接从 bearing 构造，
并天然规避 359.5°±1° 跨零问题（角度只用于生成方向向量）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .angle import deg2rad, unit
from .vector import Vec2, cross, sub


@dataclass(frozen=True)
class HalfPlane:
    """
    半平面 { P : cross(direction, P - anchor) >= -eps }。
    direction 为边界线方向向量，内侧在其左手侧。
    """
    anchor: Vec2
    direction: Vec2

    def signed(self, p: Vec2) -> float:
        """有向值；>=0 表示在内侧（含边界）。"""
        return cross(self.direction, sub(p, self.anchor))

    def contains(self, p: Vec2, eps: float = 1e-9) -> bool:
        return self.signed(p) >= -eps


def wedge_halfplanes(anchor: Vec2, bearing_deg: float, half_angle_deg: float) -> tuple[HalfPlane, HalfPlane]:
    """
    构造以 anchor 为顶点、中线方位角 bearing_deg、半张角 half_angle_deg 的楔形，
    返回两个半平面 (H_low, H_high)，其交集即楔形（张角 < 180°）。

    真实方位角满足 bearing_deg - δ <= arg(P-anchor) <= bearing_deg + δ。
    - 低边界方向 dir_lo = unit(bearing - δ)：楔形在其【左侧】 → cross(dir_lo, P-anchor) >= 0
    - 高边界方向 dir_hi = unit(bearing + δ)：楔形在其【右侧】 → cross(dir_hi, P-anchor) <= 0
      等价改写为 cross(-dir_hi, P-anchor) >= 0，统一为“左侧”约定。
    """
    b = deg2rad(bearing_deg)
    d = deg2rad(half_angle_deg)
    dir_lo = unit(b - d)
    dir_hi = unit(b + d)
    h_low = HalfPlane(anchor, dir_lo)                     # 内侧 = dir_lo 左侧
    h_high = HalfPlane(anchor, (-dir_hi[0], -dir_hi[1]))  # 内侧 = dir_hi 右侧
    return h_low, h_high
