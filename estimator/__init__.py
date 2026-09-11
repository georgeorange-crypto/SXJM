"""
estimator —— 问题 1/2 的几何定位库。

问题 1：由若干检测点 + 对同一干扰源的示向度（各 ±1°），求"多边形定位区域" D，
        求其直径，并判定"以该直径为直径的圆"能否覆盖 D。
问题 2：两层模型。
        第 1 层（解释/画图/初值）：交会角-精度公式 L、逐假设-90° 候选带、单点基线。
        第 2 层（最终答案）：有界不确定性下的 Minimax Next-Best-View——
        Ω₁（第一可行集）→ 𝒞_recv（保证接收域）→ J(S2)=最坏清除覆盖半径(MEC)
        → S2*=argmin J → 一次清除区 𝒞_clear{J≤20} / 近优带 𝒞_η → 就近择点。

核心模块：
  config.py     题面常量（附件2）与几何容差
  geometry.py   角度/楔形/半平面交/凸多边形直径/MEC/角度轮廓/外切多边形
  problem1.py   P1 顶层：locate_region / diameter / diameter_circle_covers
  problem2.py   P2 顶层：L 公式、候选带（第1层）；Ω₁/𝒞_recv/minimax（第2层）
"""

from __future__ import annotations

from .config import ProblemConfig, DEFAULT_CONFIG
from .geometry import (
    deg2rad, rad2deg, norm_deg, dir_vec, cross, dot,
    Halfplane, wedge_halfplanes, halfplane_intersection,
    polygon_diameter, diameter_circle_covers, convex_hull,
    circle_outer_halfplanes, clip_polygon_halfplanes,
    angular_span_from_point, farthest_vertex_distance, min_enclosing_circle,
)
from .problem1 import (
    LocateResult, locate_region, solve_problem1,
)
from .problem2 import (
    # 第 1 层（解释/初值）
    localization_length_L, crossing_angle, second_point_candidates,
    recommend_second_point, recommend_second_point_midpoint_baseline,
    region_diameter_two_points,
    # 第 2 层（最终答案）
    P2Status, SourceUncertaintySet, SecondPointRecommendation,
    first_feasible_set, analytic_second_point_tokekar,
    reception_guaranteed, worst_case_clearance_radius, minimax_second_point,
)

__all__ = [
    "ProblemConfig", "DEFAULT_CONFIG",
    "deg2rad", "rad2deg", "norm_deg", "dir_vec", "cross", "dot",
    "Halfplane", "wedge_halfplanes", "halfplane_intersection",
    "polygon_diameter", "diameter_circle_covers", "convex_hull",
    "circle_outer_halfplanes", "clip_polygon_halfplanes",
    "angular_span_from_point", "farthest_vertex_distance", "min_enclosing_circle",
    "LocateResult", "locate_region", "solve_problem1",
    "localization_length_L", "crossing_angle", "second_point_candidates",
    "recommend_second_point", "recommend_second_point_midpoint_baseline",
    "region_diameter_two_points",
    "P2Status", "SourceUncertaintySet", "SecondPointRecommendation",
    "first_feasible_set", "analytic_second_point_tokekar",
    "reception_guaranteed", "worst_case_clearance_radius", "minimax_second_point",
]
