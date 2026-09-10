"""
estimator —— 问题 1/2 的几何定位库。

问题 1：由若干检测点 + 对同一干扰源的示向度（各 ±1°），求"多边形定位区域" D，
        求其直径，并判定"以该直径为直径的圆"能否覆盖 D。
问题 2：由第一个检测点的示向度，分析交会角/距离与定位精度 L 的关系，
        给出第二检测点的候选区域与推荐点。

核心模块：
  geometry.py   角度/楔形/半平面交/凸多边形直径/直径圆覆盖判定
  problem1.py   P1 顶层：locate_region / diameter / diameter_circle_covers
  problem2.py   P2 顶层：交会角-精度公式 L、候选区域、推荐第二点
"""

from __future__ import annotations

from .geometry import (
    deg2rad, rad2deg, norm_deg, dir_vec, cross, dot,
    Halfplane, wedge_halfplanes, halfplane_intersection,
    polygon_diameter, diameter_circle_covers, convex_hull,
)
from .problem1 import (
    LocateResult, locate_region, solve_problem1,
)
from .problem2 import (
    localization_length_L, crossing_angle, second_point_candidates,
    recommend_second_point, region_diameter_two_points,
)

__all__ = [
    "deg2rad", "rad2deg", "norm_deg", "dir_vec", "cross", "dot",
    "Halfplane", "wedge_halfplanes", "halfplane_intersection",
    "polygon_diameter", "diameter_circle_covers", "convex_hull",
    "LocateResult", "locate_region", "solve_problem1",
    "localization_length_L", "crossing_angle", "second_point_candidates",
    "recommend_second_point", "region_diameter_two_points",
]
