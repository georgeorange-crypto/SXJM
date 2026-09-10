"""
问题 1：多测向定位区域 D、其直径、以及"以直径为直径的圆"能否覆盖 D。

场景（思路.md §3）：对同一干扰源，机器狗在 n 个不同检测点分别测得示向度 svd_i，
每次误差 ∈[-1°,+1°]。第 i 个检测点给出一个 ±1° 的示向度楔形 W_i（两个半平面之交）。
真源必落在所有楔形之交 D = ∩ W_i（凸多边形）内。

问题要点：
  (1) 求定位区域 D（凸多边形顶点）。
  (2) 求 D 的直径 diam(D)（最远点对距离）。
  (3) 判定"以 diam 对应两端点连线为直径的圆"能否覆盖整个 D。
      —— 这不是恒真命题：等边三角形形状的 D 会给出反例（Jung 定理，
         最小包围圆半径可达 D/√3 > D/2，直径圆半径恰为 D/2 无法覆盖）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .geometry import (
    wedge_halfplanes, halfplane_intersection, polygon_diameter,
    diameter_circle_covers, min_enclosing_circle, Halfplane,
)


@dataclass
class LocateResult:
    polygon: list[tuple[float, float]]          # 定位区域 D 的顶点（凸多边形）
    diameter: float                              # D 的直径
    diam_endpoints: tuple[tuple[float, float], tuple[float, float]]  # 直径两端 A,B
    diam_circle_covers: bool                     # 以 AB 为直径的圆是否覆盖 D
    violators: list[tuple[float, float]]         # 若不覆盖，落在圆外的顶点
    mec_center: tuple[float, float]              # 最小包围圆圆心（对比用）
    mec_radius: float                            # 最小包围圆半径（对比用）
    empty: bool = False                          # 交集为空（示向度互相矛盾）

    @property
    def diam_circle_radius(self) -> float:
        return self.diameter / 2.0

    def summary(self) -> str:
        if self.empty:
            return "定位区域为空：给定示向度相互矛盾（无公共真源方向）。"
        cover = "可覆盖" if self.diam_circle_covers else "不可覆盖"
        return (f"顶点数={len(self.polygon)}  直径={self.diameter:.3f} m  "
                f"直径圆半径={self.diam_circle_radius:.3f} m  最小包围圆半径={self.mec_radius:.3f} m  "
                f"直径圆{cover}整个区域"
                + ("" if self.diam_circle_covers
                   else f"（{len(self.violators)} 个顶点在圆外）"))


def locate_region(
    detections: list[tuple[float, float, float]],   # [(sx, sy, svd_deg), ...]
    delta_deg: float = 1.0,
    clip_circle_radius: Optional[float] = 1800.0,
    bound: float = 6000.0,
) -> LocateResult:
    """
    由若干 (检测点, 示向度) 求定位区域 D 及其直径 / 直径圆覆盖判定。

    - detections：每个元素 (sx, sy, svd_deg)。至少 1 个；≥2 个才可能有界。
    - delta_deg：示向度误差半张角（题面 1°）。
    - clip_circle_radius：用目标区域圆（半径 1800）再裁一次以保证有界；None 则不裁。
    - bound：初始大正方形半边长。
    """
    halfplanes: list[Halfplane] = []
    for (sx, sy, svd) in detections:
        halfplanes.extend(wedge_halfplanes(sx, sy, svd, delta_deg))

    poly = halfplane_intersection(halfplanes, bound=bound,
                                  clip_circle_radius=clip_circle_radius)
    if not poly:
        return LocateResult(polygon=[], diameter=0.0,
                            diam_endpoints=((0, 0), (0, 0)),
                            diam_circle_covers=False, violators=[],
                            mec_center=(0.0, 0.0), mec_radius=0.0, empty=True)

    diam, a, b = polygon_diameter(poly)
    covers, violators = diameter_circle_covers(poly, a, b)
    mc, mr = min_enclosing_circle(poly)
    return LocateResult(
        polygon=poly, diameter=diam, diam_endpoints=(a, b),
        diam_circle_covers=covers, violators=violators,
        mec_center=mc, mec_radius=mr, empty=False,
    )


def solve_problem1(
    true_source: tuple[float, float],
    detection_points: list[tuple[float, float]],
    error_field=None,
    delta_deg: float = 1.0,
    clip_circle_radius: Optional[float] = 1800.0,
) -> LocateResult:
    """
    正演 + 求解一条龙（供仿真 / 出图用）：
    给定真源与检测点集，用误差场（或零误差）生成各点示向度，再反演定位区域。

    error_field：若提供，须有 error_deg(x,y)；否则各点误差取 0（理想）。
    真示向度 = 从检测点指向真源的方位角；观测示向度 = 真值 + 该点误差。
    """
    import math
    from .geometry import norm_deg

    detections = []
    for (sx, sy) in detection_points:
        true_bearing = norm_deg(math.degrees(math.atan2(true_source[1] - sy,
                                                        true_source[0] - sx)))
        err = 0.0 if error_field is None else error_field.error_deg(sx, sy)
        svd = norm_deg(true_bearing + err)
        detections.append((sx, sy, svd))
    return locate_region(detections, delta_deg=delta_deg,
                         clip_circle_radius=clip_circle_radius)
