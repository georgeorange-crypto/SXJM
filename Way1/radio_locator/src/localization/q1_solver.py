"""
Q1 定位：多点示向度楔形交会（bounded-error 三角定位）。

每次 measure 得一个带 ±1° 误差的方位 → 一个楔形半平面对（wedge）。源必在楔形内。
多个楔形之交即为可行位置集 Ω（凸多边形）。用 Sutherland–Hodgman 逐个半平面裁剪
初始包围方块，得到 Ω 的精确多边形；其最小包围圆 MEC 给出定位精度与清除时机。

正观测若为 near（≤5m），额外交上以观测点为心、5m 的圆盘（用外接正多边形保守内包）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..domain.observation import Observation
from ..domain.types import ObservationType, ProblemConstants, RobotConstants, Vec2
from ..geometry.halfplane import HalfPlane
from ..geometry.mec import Circle, min_enclosing_circle
from ..geometry.polygon_clip import (bounding_square, clip_polygon,
                                      clip_polygon_halfplanes, polygon_area)
from ..geometry.rotating_calipers import diameter as poly_diameter
from ..geometry.wedge import Wedge


@dataclass
class Q1Result:
    polygon: List[Vec2]
    mec: Optional[Circle]
    diameter: float
    area: float

    def is_empty(self) -> bool:
        return len(self.polygon) < 3 or self.mec is None

    @property
    def center(self) -> Optional[Vec2]:
        return self.mec.center if self.mec else None

    @property
    def radius(self) -> float:
        return self.mec.radius if self.mec else float("inf")


def _disk_polygon(center: Vec2, radius: float, sides: int = 24) -> List[Vec2]:
    """圆盘的【内接】正多边形（保守：多边形 ⊆ 圆盘，避免把可行点裁掉需外接；
    但我们要交的是‘源在圆盘内’这一约束，应使用【外接】多边形以不丢点）。"""
    # 外接正多边形：顶点在半径 radius/cos(π/sides) 上，保证圆盘 ⊆ 多边形。
    R = radius / math.cos(math.pi / sides)
    return [(center[0] + R * math.cos(2 * math.pi * k / sides),
             center[1] + R * math.sin(2 * math.pi * k / sides)) for k in range(sides)]


def solve_q1(observations: List[Observation], problem: ProblemConstants,
             robot: RobotConstants, half_deg: Optional[float] = None) -> Q1Result:
    """
    对一组同频道观测求可行位置多边形。仅用 bearing / near 正观测（Q1 单源、必存在）。
    half_deg 缺省时取 problem.wedge_half_deg（物理误差界 + 报告量化裕量），确保真源不被裁出。
    """
    if half_deg is None:
        half_deg = problem.wedge_half_deg
    arena = problem.area_radius
    poly = bounding_square(arena)

    halfplanes: List[HalfPlane] = []
    for o in observations:
        if o.result == ObservationType.BEARING and o.bearing_deg is not None:
            w = Wedge(o.position, o.bearing_deg, half_deg)
            hlo, hhi = w.halfplanes()
            halfplanes.append(hlo)
            halfplanes.append(hhi)
    if halfplanes:
        poly = clip_polygon_halfplanes(poly, halfplanes)

    # near：交上 5m 外接多边形
    for o in observations:
        if o.result == ObservationType.TOO_STRONG:
            disk = _disk_polygon(o.position, robot.too_strong_radius)
            poly = _intersect_convex(poly, disk)
            if len(poly) < 3:
                break

    if len(poly) < 3:
        return Q1Result(polygon=[], mec=None, diameter=float("inf"), area=0.0)

    mec = min_enclosing_circle(poly)
    d, _, _ = poly_diameter(poly)
    return Q1Result(polygon=poly, mec=mec, diameter=d, area=polygon_area(poly))


def _intersect_convex(poly: List[Vec2], clipper: List[Vec2]) -> List[Vec2]:
    """用凸多边形 clipper 的每条边（作为半平面）裁剪 poly。"""
    out = poly
    n = len(clipper)
    for i in range(n):
        a = clipper[i]
        b = clipper[(i + 1) % n]
        hp = HalfPlane(anchor=a, direction=(b[0] - a[0], b[1] - a[1]))
        out = clip_polygon(out, hp)
        if len(out) < 3:
            return []
    return out
