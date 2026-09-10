"""
问题 2：两点交会定位的精度与第二检测点选取。

场景（思路.md §4）：机器狗先在检测点 S1 测得对某源的示向度 svd1（真源在 S1 的
±1° 楔形内）。再选第二检测点 S2 测得 svd2，两条 ±1° 楔形之交 = 一个近似平行四边形
的小区域，其"沿测向方向的伸长量"L 决定定位精度。

一阶几何（δ 为误差半张角、γ 为两测向线在真源处的交会角、r1=|S1P|、r2=|S2P|）：

    L ≈ (2δ_rad / sinγ) · √(r1² + r2² + 2 r1 r2 |cosγ|)

要点：
  - 交会角 γ→90° 时 sinγ→1 且 |cosγ|→0，L 最小 → 第二点应使两测向线尽量正交。
  - r2 越小（第二点离源越近）L 越小 → 但源位置未知，只能沿 svd1 的楔形推进。
  - r1 未知（只知道 S1、svd1，不知道 P 距离）→ 第二点是一个"候选区域"而非单点：
    对 r1 的每个可能取值，最优 S2 落在过对应 P、与 svd1 方向垂直、距 P 为 r2 的点，
    r1 扫过区间 → 这些点扫出一段"月牙形"候选带。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .geometry import deg2rad, norm_deg, dir_vec, wedge_halfplanes, \
    halfplane_intersection, polygon_diameter


def crossing_angle(s1: tuple[float, float], s2: tuple[float, float],
                   p: tuple[float, float]) -> float:
    """两测向线 S1→P 与 S2→P 在 P 处的交会角（度，[0,180]）。"""
    v1x, v1y = p[0] - s1[0], p[1] - s1[1]
    v2x, v2y = p[0] - s2[0], p[1] - s2[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = (v1x * v2x + v1y * v2y) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def localization_length_L(r1: float, r2: float, gamma_deg: float,
                          delta_deg: float = 1.0) -> float:
    """
    交会定位区域的一阶伸长量 L（米）。见模块头公式。
    gamma_deg：交会角（度）。gamma→0/180（近平行）时 sinγ→0，L→∞（病态）。
    """
    g = deg2rad(gamma_deg)
    s = math.sin(g)
    if abs(s) < 1e-9:
        return float("inf")
    delta = deg2rad(delta_deg)
    return (2.0 * delta / abs(s)) * math.sqrt(
        r1 * r1 + r2 * r2 + 2.0 * r1 * r2 * abs(math.cos(g)))


@dataclass
class SecondPointCandidate:
    r1: float                       # 假设的 S1→源 距离
    p: tuple[float, float]          # 该假设下的源位置（沿 svd1 方向）
    s2: tuple[float, float]         # 推荐第二检测点
    r2: float                       # |S2→源|
    gamma_deg: float                # 交会角
    L: float                        # 该点的定位伸长量


def second_point_candidates(
    s1: tuple[float, float],
    svd1_deg: float,
    r1_values: Optional[list[float]] = None,
    r2: float = 300.0,
    gamma_target_deg: float = 90.0,
    arena_radius: float = 1800.0,
    side: int = 0,
) -> list[SecondPointCandidate]:
    """
    因 r1（S1 到源的距离）未知，沿 svd1 的可能源位置扫过一段区间，
    对每个假设 r1 给出一个"使交会角=gamma_target、且 |S2源|=r2"的推荐第二点。
    这些点连起来就是"月牙形候选带"。

    几何：在假设源 P 处，P→S1 方向为 svd1+180°；要让 ∠(P→S1, P→S2)=γ，
    则 P→S2 方向 = (svd1+180°) ± γ，两个符号对应测向线两侧。S2 = P + r2·dir。

    - r1_values：假设的 S1→源距离序列；默认覆盖 [200,1800] 一段。
    - r2：希望的第二点到源距离（越小 L 越小，但要可达且留出移动/量程余量）。
    - gamma_target_deg：目标交会角（90° 使 L 最小）。
    - side：+1 / -1 固定取某一侧（画干净的单条带）；0=自动（两侧都在区域内时
      取 L 最小者，等价时取 +1 侧），保证同一侧连续、不出现锯齿。
    返回落在目标区域圆内的候选点。
    """
    if r1_values is None:
        r1_values = [200.0 + 100.0 * k for k in range(0, 17)]   # 200..1800

    ux, uy = dir_vec(svd1_deg)                    # S1 指向源的单位方向
    back = norm_deg(svd1_deg + 180.0)             # 源 P 指向 S1 的方向
    signs = (side,) if side in (+1, -1) else (+1, -1)

    out: list[SecondPointCandidate] = []
    for r1 in r1_values:
        px, py = s1[0] + ux * r1, s1[1] + uy * r1     # 假设源位置 P
        cand = []
        for sgn in signs:
            dx, dy = dir_vec(norm_deg(back + sgn * gamma_target_deg))
            s2 = (px + dx * r2, py + dy * r2)
            if math.hypot(s2[0], s2[1]) <= arena_radius + 1e-9:
                g = crossing_angle(s1, s2, (px, py))
                L = localization_length_L(r1, r2, g)
                cand.append((sgn, SecondPointCandidate(r1, (px, py), s2, r2, g, L)))
        if not cand:
            continue
        if side in (+1, -1):
            out.append(cand[0][1])
        else:
            # 自动：优先 +1 侧（保证连续），仅当其越界时退到 -1 侧
            plus = [c for s, c in cand if s == +1]
            out.append(plus[0] if plus else cand[0][1])
    return out


def recommend_second_point(
    s1: tuple[float, float],
    svd1_deg: float,
    r1_guess: float = 1000.0,
    r2: float = 300.0,
    arena_radius: float = 1800.0,
) -> SecondPointCandidate:
    """
    给一个可执行的单点建议（用 r1 的中位猜测）。实战中 r1 未知，
    但这个点在"候选带"中央，是稳健的默认第二检测点。
    """
    cands = second_point_candidates(s1, svd1_deg, r1_values=[r1_guess],
                                    r2=r2, arena_radius=arena_radius)
    if cands:
        return cands[0]
    # 退化：直接沿垂直方向找一个区域内的点
    ux, uy = dir_vec(svd1_deg)
    px, py = s1[0] + ux * r1_guess, s1[1] + uy * r1_guess
    for (dx, dy) in ((-uy, ux), (uy, -ux)):
        s2 = (px + dx * r2, py + dy * r2)
        if math.hypot(s2[0], s2[1]) <= arena_radius + 1e-9:
            g = crossing_angle(s1, s2, (px, py))
            return SecondPointCandidate(r1_guess, (px, py), s2, r2, g,
                                        localization_length_L(r1_guess, r2, g))
    return SecondPointCandidate(r1_guess, (px, py), (px, py), 0.0, 0.0, float("inf"))


def region_diameter_two_points(
    s1: tuple[float, float], svd1_deg: float,
    s2: tuple[float, float], svd2_deg: float,
    delta_deg: float = 1.0,
    arena_radius: float = 1800.0,
) -> tuple[float, list[tuple[float, float]]]:
    """
    两检测点各一次示向度 → 两楔形之交（近平行四边形小区域）的直径与顶点。
    用于校核一阶公式 L 与真实几何区域尺寸的吻合程度。
    """
    hps = wedge_halfplanes(*s1, svd1_deg, delta_deg) + \
        wedge_halfplanes(*s2, svd2_deg, delta_deg)
    poly = halfplane_intersection(hps, bound=6000.0, clip_circle_radius=arena_radius)
    if not poly:
        return 0.0, []
    diam, _, _ = polygon_diameter(poly)
    return diam, poly
