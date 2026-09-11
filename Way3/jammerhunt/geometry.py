"""
几何核心：方位角、交会定位（cross-fix）、定位区域（误差楔形的交）与最小包围圆、路径排序。

坐标/角度约定（附件2 §1.1/§1.2）：
- x 正东、y 正北，单位米；方位角以正东为 0°，逆时针为正，范围 [0,360)。
- 示向度 svd_deg = 从检测点指向干扰源的方位角（含 ≤1° 误差）。

本模块只做纯几何，不含任何物理/通信，便于单元测试。仅依赖标准库。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

Point = tuple[float, float]

# 示向度误差半宽（附件2 §2.3：误差 ∈ [-1°,+1°]）。定位区域用它构造。
SVD_HALF_ANGLE_DEG = 1.0


# --------------------------------------------------------------------------- #
# 角度工具
# --------------------------------------------------------------------------- #
def norm_deg(a: float) -> float:
    """归一化到 [0,360)。"""
    a = math.fmod(a, 360.0)
    if a < 0.0:
        a += 360.0
    if a >= 360.0:
        a -= 360.0
    return a


def ang_diff(a: float, b: float) -> float:
    """两方位角的最小夹角，返回 [0,180]。"""
    d = abs(norm_deg(a) - norm_deg(b))
    return d if d <= 180.0 else 360.0 - d


def signed_ang_diff(a: float, b: float) -> float:
    """有向角差 a-b，折到 (-180,180]。正=从 b 逆时针到 a。"""
    d = norm_deg(a - b)
    return d - 360.0 if d > 180.0 else d


def unit(bearing_deg: float) -> Point:
    """方位角 → 单位方向向量。"""
    r = math.radians(bearing_deg)
    return math.cos(r), math.sin(r)


def bearing_to(p_from: Point, p_to: Point) -> float:
    """从 p_from 指向 p_to 的方位角（度，[0,360)）。"""
    return norm_deg(math.degrees(math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0])))


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


# --------------------------------------------------------------------------- #
# 交会定位（两条射线/直线求交）
# --------------------------------------------------------------------------- #
def ray_intersect(p1: Point, b1: float, p2: Point, b2: float) -> Optional[Point]:
    """
    两条从 p1、p2 出发、方位角 b1、b2 的射线的交点。
    近平行返回 None。不检查“正向”（射线 vs 直线）——交会定位中源一定在正向，
    调用方如需可用返回参数判断；这里返回直线交点即可。
    """
    u1x, u1y = unit(b1)
    u2x, u2y = unit(b2)
    denom = _cross(u1x, u1y, u2x, u2y)     # u1 × u2
    if abs(denom) < 1e-12:
        return None
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    t1 = _cross(dx, dy, u2x, u2y) / denom   # p1 + t1*u1 = 交点
    return p1[0] + t1 * u1x, p1[1] + t1 * u1y


def crossing_angle(p1: Point, b1: float, p2: Point, b2: float) -> float:
    """两条示向线的交会角（0..90]，越接近 90° 定位越好（对应附件2 图2）。"""
    d = ang_diff(b1, b2)
    return min(d, 180.0 - d)


def least_squares_fix(obs: Sequence[tuple[Point, float]]) -> Optional[Point]:
    """
    ≥2 条示向观测的最小二乘交会：最小化点到各示向直线的垂距平方和。
    每条线过 p_i、方向 u_i=(c,s)，点 X 到该线的有符号垂距 = (X-p_i)×u_i。
    对 Σ[(X-p)×u]^2 求极小 → 2x2 线性方程。返回 None 表示奇异（所有线平行）。
    """
    Sxx = Sxy = Syy = bx = by = 0.0
    n = 0
    for (px, py), bdeg in obs:
        c, s = unit(bdeg)
        # (X-p)×u = (Xx-px)*s - (Xy-py)*c = s*Xx - c*Xy - (s*px - c*py)
        # 法向量 a=(s,-c)，常数 d = s*px - c*py；残差 = a·X - d
        ax, ay, d = s, -c, s * px - c * py
        Sxx += ax * ax
        Sxy += ax * ay
        Syy += ay * ay
        bx += ax * d
        by += ay * d
        n += 1
    if n < 2:
        return None
    det = Sxx * Syy - Sxy * Sxy
    if abs(det) < 1e-9:
        return None
    x = (Syy * bx - Sxy * by) / det
    y = (Sxx * by - Sxy * bx) / det
    return x, y


def best_pair_fix(obs: Sequence[tuple[Point, float]]) -> Optional[Point]:
    """
    在所有观测对中选交会角最大的一对做交点；若最好一对仍近平行则退回最小二乘。
    返回估计位置或 None。
    """
    best = None
    best_ang = -1.0
    m = len(obs)
    for i in range(m):
        for j in range(i + 1, m):
            (p1, b1), (p2, b2) = obs[i], obs[j]
            ang = crossing_angle(p1, b1, p2, b2)
            if ang > best_ang:
                pt = ray_intersect(p1, b1, p2, b2)
                if pt is not None:
                    best_ang, best = ang, pt
    if best is None:
        return least_squares_fix(obs)
    return best


# --------------------------------------------------------------------------- #
# 定位区域（两个 ±1° 误差楔形的交）与最小包围圆
# --------------------------------------------------------------------------- #
def min_enclosing_circle(pts: Sequence[Point]) -> tuple[Point, float]:
    """
    最小包围圆（点数很少，用 O(n^4) 暴力：最优圆由某两点为直径或某三点外接圆决定）。
    返回 (圆心, 半径)。空集返回 ((0,0),0)。
    """
    P = list(pts)
    if not P:
        return (0.0, 0.0), 0.0
    if len(P) == 1:
        return P[0], 0.0

    def covers(cx, cy, r):
        return all(math.hypot(px - cx, py - cy) <= r + 1e-7 for px, py in P)

    best_c, best_r = None, math.inf
    # 以两点为直径
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            cx = (P[i][0] + P[j][0]) / 2.0
            cy = (P[i][1] + P[j][1]) / 2.0
            r = math.hypot(P[i][0] - cx, P[i][1] - cy)
            if r < best_r and covers(cx, cy, r):
                best_c, best_r = (cx, cy), r
    # 以三点外接圆
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            for k in range(j + 1, len(P)):
                c = _circumcenter(P[i], P[j], P[k])
                if c is None:
                    continue
                r = math.hypot(P[i][0] - c[0], P[i][1] - c[1])
                if r < best_r and covers(c[0], c[1], r):
                    best_c, best_r = c, r
    if best_c is None:                     # 退化：全共点
        best_c, best_r = P[0], 0.0
    return best_c, best_r


def _circumcenter(a: Point, b: Point, c: Point) -> Optional[Point]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy


def localization_region(
    p1: Point, b1: float, p2: Point, b2: float, half: float = SVD_HALF_ANGLE_DEG
) -> Optional[list[Point]]:
    """
    定位区域（附件2 图2 的红色四边形）：两条示向观测各带 ±half° 误差楔形，
    真源必落在两楔形的交集里。区域为凸四边形，四个顶点是四条边界射线两两相交。
    返回 4 个顶点（无界/病态返回 None）。
    """
    corners: list[Point] = []
    for d1 in (-half, +half):
        for d2 in (-half, +half):
            pt = ray_intersect(p1, b1 + d1, p2, b2 + d2)
            if pt is None:
                return None
            corners.append(pt)
    return corners


def region_center_radius(
    p1: Point, b1: float, p2: Point, b2: float, half: float = SVD_HALF_ANGLE_DEG
) -> Optional[tuple[Point, float]]:
    """
    定位区域的最小包围圆 (圆心, 半径)。在圆心处清除，若半径 ≤ 清除半径则必命中
    （真源 ∈ 区域 ⊆ 该圆）。无界返回 None。
    """
    corners = localization_region(p1, b1, p2, b2, half)
    if corners is None:
        return None
    # 顶点离两观测点都应在“正向”（源在射线正方向）。若出现异常远的顶点视为病态。
    c, r = min_enclosing_circle(corners)
    return c, r


def best_region(
    obs: Sequence[tuple[Point, float]], half: float = SVD_HALF_ANGLE_DEG
) -> Optional[tuple[Point, float]]:
    """在所有观测对里取“定位区域最小包围圆半径最小”的一对，返回 (圆心, 半径)。"""
    best = None
    best_r = math.inf
    m = len(obs)
    for i in range(m):
        for j in range(i + 1, m):
            (p1, b1), (p2, b2) = obs[i], obs[j]
            if crossing_angle(p1, b1, p2, b2) < 2.0:      # 太平行，跳过
                continue
            cr = region_center_radius(p1, b1, p2, b2, half)
            if cr is not None and cr[1] < best_r:
                best_r, best = cr[1], cr
    return best


# --------------------------------------------------------------------------- #
# 路径排序（清除任务 / 扫描点的短巡回）
# --------------------------------------------------------------------------- #
def path_length(order: Sequence[int], pts: Sequence[Point], start: Point) -> float:
    total = 0.0
    cur = start
    for idx in order:
        total += dist(cur, pts[idx])
        cur = pts[idx]
    return total


def nearest_neighbor_order(start: Point, pts: Sequence[Point]) -> list[int]:
    """从 start 出发的最近邻巡回顺序（开放路径，不回到起点）。"""
    remaining = list(range(len(pts)))
    order: list[int] = []
    cur = start
    while remaining:
        j = min(remaining, key=lambda i: dist(cur, pts[i]))
        order.append(j)
        cur = pts[j]
        remaining.remove(j)
    return order


def two_opt(order: list[int], pts: Sequence[Point], start: Point,
            max_pass: int = 40) -> list[int]:
    """2-opt 局部改进开放路径（固定起点为 start，不固定终点）。"""
    if len(order) < 4:
        return order
    best = order[:]
    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        for i in range(len(best) - 1):
            for k in range(i + 1, len(best)):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                if path_length(cand, pts, start) + 1e-9 < path_length(best, pts, start):
                    best = cand
                    improved = True
    return best


def ordered_tour(start: Point, pts: Sequence[Point]) -> list[int]:
    """最近邻 + 2-opt，返回访问 pts 的下标顺序。"""
    if not pts:
        return []
    return two_opt(nearest_neighbor_order(start, pts), pts, start)
