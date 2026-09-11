"""
几何基础：角度约定、示向度楔形、半平面交、凸多边形直径、直径圆覆盖判定。

角度约定与题面一致（附件2 §1.2）：东=0°、逆时针为正、范围 [0,360)。
示向度楔形（思路.md §2.1）：测得示向度 svd、误差半张角 δ=1° 时，真源方向必落在
从检测点出发、方向区间 [svd-δ, svd+δ] 的角锥内。该角锥 = 两个半平面之交。

半平面表示：{ P : n·P <= c }，n 为向内法向的相反方向（这里统一用 n·P<=c 形式）。
半平面交用 Sutherland–Hodgman 逐半平面裁剪一个大初始凸多边形实现（数值稳健、实现简单，
顶点数 V 很小，复杂度足够）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------- 角度与向量 ----------------
def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


def norm_deg(a: float) -> float:
    """归一化到 [0,360)。"""
    a = math.fmod(a, 360.0)
    if a < 0:
        a += 360.0
    if a >= 360.0:
        a -= 360.0
    return a


def dir_vec(theta_deg: float) -> tuple[float, float]:
    """方位角（度，东=0、逆时针正）→ 单位方向向量。"""
    r = deg2rad(theta_deg)
    return (math.cos(r), math.sin(r))


def cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


# ---------------- 半平面 ----------------
@dataclass
class Halfplane:
    """半平面 { (x,y) : nx*x + ny*y <= c }。"""
    nx: float
    ny: float
    c: float

    def contains(self, x: float, y: float, eps: float = 1e-9) -> bool:
        return self.nx * x + self.ny * y <= self.c + eps


def wedge_halfplanes(sx: float, sy: float, svd_deg: float,
                     delta_deg: float = 1.0) -> list[Halfplane]:
    """
    检测点 (sx,sy)、示向度 svd_deg、误差半张角 delta_deg 的示向度楔形，
    表示为两个半平面之交。

    楔形 = { P : angle(P - S) ∈ [svd-δ, svd+δ] }（取张角 2δ<180° 的那一侧）。
    低边界方向 u_lo=dir(svd-δ)：楔形在其"逆时针一侧" → cross(u_lo, P-S) >= 0。
    高边界方向 u_hi=dir(svd+δ)：楔形在其"顺时针一侧" → cross(u_hi, P-S) <= 0。
    统一成 n·P<=c 形式返回。
    """
    lo = norm_deg(svd_deg - delta_deg)
    hi = norm_deg(svd_deg + delta_deg)
    ux_lo, uy_lo = dir_vec(lo)
    ux_hi, uy_hi = dir_vec(hi)

    # cross(u_lo, P-S) >= 0  ⟺  u_lo.x*(Py-Sy) - u_lo.y*(Px-Sx) >= 0
    #   ⟺ (-u_lo.y)*Px + (u_lo.x)*Py >= (-u_lo.y)*Sx + (u_lo.x)*Sy
    #   两边乘 -1 变成 <= 形式：
    #   (u_lo.y)*Px + (-u_lo.x)*Py <= (u_lo.y)*Sx + (-u_lo.x)*Sy
    hp_lo = Halfplane(uy_lo, -ux_lo, uy_lo * sx - ux_lo * sy)

    # cross(u_hi, P-S) <= 0  ⟺  (-u_hi.y)*Px + (u_hi.x)*Py <= (-u_hi.y)*Sx + (u_hi.x)*Sy
    hp_hi = Halfplane(-uy_hi, ux_hi, -uy_hi * sx + ux_hi * sy)

    return [hp_lo, hp_hi]


def circle_outer_halfplanes(cx: float, cy: float, radius: float,
                            sides: int = 128) -> list[Halfplane]:
    """
    圆 {‖P−C‖ ≤ radius} 的**外切**多边形（circumscribed）对应的半平面组。

    每个半平面的边界是圆在角 a 处的切线，法向 (cos a, sin a)：
        cos a·x + sin a·y ≤ radius + cx·cos a + cy·sin a
    这些切线半平面之交是一个 sides 边的多边形，**包含**该圆（顶点在半径
    radius/cos(π/sides) 处，向外多出约 radius·(sec(π/sides)−1)）。

    "外近似"是刻意的：用它裁剪不确定集时得到的是**超集**，据此算出的最坏直径、
    可行域是**保证成立的上界/充分条件**（宁可保守，不可漏掉真解）。
    cx=cy=0 时退化为 Halfplane(cos a, sin a, radius)，与 halfplane_intersection
    里对原点圆的裁剪完全一致。
    """
    hps: list[Halfplane] = []
    for k in range(sides):
        a = 2.0 * math.pi * k / sides
        ca, sa = math.cos(a), math.sin(a)
        hps.append(Halfplane(ca, sa, radius + cx * ca + cy * sa))
    return hps


def _clip_polygon(poly: list[tuple[float, float]], hp: Halfplane,
                  eps: float = 1e-9) -> list[tuple[float, float]]:
    """用一个半平面裁剪凸多边形（Sutherland–Hodgman）。"""
    if not poly:
        return poly
    out: list[tuple[float, float]] = []
    n = len(poly)
    for i in range(n):
        cx, cy = poly[i]
        nx_, ny_ = poly[(i + 1) % n]
        c_in = (hp.nx * cx + hp.ny * cy) <= hp.c + eps
        n_in = (hp.nx * nx_ + hp.ny * ny_) <= hp.c + eps
        if c_in:
            out.append((cx, cy))
        if c_in != n_in:
            # 求边与半平面边界的交点
            d_c = hp.c - (hp.nx * cx + hp.ny * cy)
            d_n = hp.c - (hp.nx * nx_ + hp.ny * ny_)
            denom = (d_c - d_n)
            if abs(denom) > 1e-18:
                t = d_c / denom
                ix = cx + t * (nx_ - cx)
                iy = cy + t * (ny_ - cy)
                out.append((ix, iy))
    return out


def _bounding_box_polygon(half: float) -> list[tuple[float, float]]:
    """以原点为中心、半边长 half 的正方形，作为半平面交的初始大凸多边形。"""
    return [(-half, -half), (half, -half), (half, half), (-half, half)]


def halfplane_intersection(halfplanes: list[Halfplane],
                           bound: float = 5000.0,
                           clip_circle_radius: Optional[float] = None
                           ) -> list[tuple[float, float]]:
    """
    求若干半平面之交（凸多边形）。用一个大初始正方形逐个裁剪。
    - bound：初始正方形半边长（应大于任何可能的定位区域，默认 5000 m 覆盖 1800 圆及外扩）。
    - clip_circle_radius：若给定，用该半径的圆（多边形近似）再裁一次，
      对应"真源必在目标圆内"的先验，能把无界楔形限制成有界区域。
    返回凸多边形顶点（逆时针或顺时针），为空表示交集为空。
    """
    poly = _bounding_box_polygon(bound)
    for hp in halfplanes:
        poly = _clip_polygon(poly, hp)
        if not poly:
            return []
    if clip_circle_radius is not None:
        # 用正 128 边形近似圆做裁剪
        m = 128
        for k in range(m):
            a = 2 * math.pi * k / m
            nx_, ny_ = math.cos(a), math.sin(a)
            # 圆内：nx*x+ny*y <= R
            poly = _clip_polygon(poly, Halfplane(nx_, ny_, clip_circle_radius))
            if not poly:
                return []
    return _dedup(poly)


def clip_polygon_halfplanes(poly: list[tuple[float, float]],
                            halfplanes: list[Halfplane],
                            eps: float = 1e-9) -> list[tuple[float, float]]:
    """
    用若干半平面依次裁剪一个**已有**凸多边形（区别于 halfplane_intersection：
    后者从一个大正方形开始）。用于把不确定集 Ω₁ 再与第二楔形 W₂ 求交得到 Ω₂。
    返回裁剪后的凸多边形顶点；为空表示交集为空。
    """
    for hp in halfplanes:
        poly = _clip_polygon(poly, hp, eps)
        if not poly:
            return []
    return _dedup(poly)


def _dedup(poly: list[tuple[float, float]], eps: float = 1e-7) -> list[tuple[float, float]]:
    """去掉相邻重复点。"""
    if not poly:
        return poly
    out = [poly[0]]
    for p in poly[1:]:
        if abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


# ---------------- 凸包与直径 ----------------
def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew monotone chain 凸包，返回逆时针顶点。"""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def build(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and cross(h[-1][0] - h[-2][0], h[-1][1] - h[-2][1],
                                        p[0] - h[-2][0], p[1] - h[-2][1]) <= 0:
                h.pop()
            h.append(p)
        return h
    lower = build(pts)
    upper = build(reversed(pts))
    return lower[:-1] + upper[:-1]


def polygon_diameter(poly: list[tuple[float, float]]
                     ) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """
    凸多边形直径（最远点对距离）及对应两点。顶点数很小，直接 O(V²) 暴力。
    返回 (直径, A, B)。空/单点返回 (0, p, p)。
    """
    if not poly:
        return 0.0, (0.0, 0.0), (0.0, 0.0)
    if len(poly) == 1:
        return 0.0, poly[0], poly[0]
    best = -1.0
    ba, bb = poly[0], poly[0]
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            dx = poly[i][0] - poly[j][0]
            dy = poly[i][1] - poly[j][1]
            d = math.hypot(dx, dy)
            if d > best:
                best, ba, bb = d, poly[i], poly[j]
    return best, ba, bb


def angular_span_from_point(poly: list[tuple[float, float]],
                            sx: float, sy: float) -> tuple[float, float]:
    """
    凸多边形 poly 从外部观察点 (sx,sy) 看过去所张的**角度区间**（度）。

    返回 (lo, hi)，hi = lo + span，span ∈ [0,360]，hi 可能 > 360（调用方对楔形
    方向自行 norm_deg）。做法：把各顶点相对 (sx,sy) 的方位角排序，找最大空隙，
    覆盖弧 = 全圆减去最大空隙。若 (sx,sy) 落在 poly 内部（无有效空隙），返回整圈。

    用途：第二检测点 S2 处，源方向 arg(G−S2) 随 G∈Ω₁ 只在这个区间内变化，
    因此 J(S2) 的最坏情形只需在 [lo−δ, hi+δ] 这个一维角度范围里扫描 φ。
    """
    if not poly:
        return (0.0, 0.0)
    angs = sorted(norm_deg(rad2deg(math.atan2(y - sy, x - sx))) for (x, y) in poly)
    n = len(angs)
    if n == 1:
        return (angs[0], angs[0])
    gaps = [(angs[i + 1] - angs[i], i) for i in range(n - 1)]
    gaps.append((angs[0] + 360.0 - angs[n - 1], n - 1))   # 首尾环绕空隙
    max_gap, gi = max(gaps)
    lo = angs[(gi + 1) % n]
    span = 360.0 - max_gap
    return (lo, lo + span)


def farthest_vertex_distance(poly: list[tuple[float, float]],
                             x: float, y: float) -> float:
    """点 (x,y) 到凸多边形顶点的最大距离。因 ‖·−(x,y)‖ 在凸集上取最大值必在顶点，
    这就等于 max_{G∈poly}‖G−(x,y)‖，用于判定"保证接收域"成员：≤ ρ_acc。"""
    if not poly:
        return 0.0
    return max(math.hypot(vx - x, vy - y) for (vx, vy) in poly)


def diameter_circle_covers(poly: list[tuple[float, float]],
                           a: tuple[float, float],
                           b: tuple[float, float],
                           eps: float = 1e-6
                           ) -> tuple[bool, list[tuple[float, float]]]:
    """
    判定"以 AB 为直径的圆"是否覆盖凸多边形 poly。
    Thales：P 在以 AB 为直径的圆内(含边界) ⟺ ∠APB ≥ 90° ⟺ (P-A)·(P-B) ≤ 0。
    返回 (是否全覆盖, 违反的顶点列表)。
    """
    violators = []
    for (px, py) in poly:
        d = dot(px - a[0], py - a[1], px - b[0], py - b[1])
        # 允许相对误差：用直径平方做归一化
        scale = max(1.0, (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        if d > eps * scale:
            violators.append((px, py))
    return (len(violators) == 0), violators


def min_enclosing_circle(points: list[tuple[float, float]]
                         ) -> tuple[tuple[float, float], float]:
    """
    Welzl 最小包围圆（期望 O(n)）。返回 (圆心, 半径)。用于与直径圆对比。
    """
    import random
    pts = list(points)
    random.Random(12345).shuffle(pts)

    def circle_two(p, q):
        cx = (p[0] + q[0]) / 2.0
        cy = (p[1] + q[1]) / 2.0
        r = math.hypot(p[0] - q[0], p[1] - q[1]) / 2.0
        return (cx, cy), r

    def circle_three(p, q, s):
        ax, ay = p; bx, by = q; cx_, cy_ = s
        d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        if abs(d) < 1e-18:
            return None
        ux = ((ax*ax+ay*ay)*(by-cy_) + (bx*bx+by*by)*(cy_-ay) + (cx_*cx_+cy_*cy_)*(ay-by)) / d
        uy = ((ax*ax+ay*ay)*(cx_-bx) + (bx*bx+by*by)*(ax-cx_) + (cx_*cx_+cy_*cy_)*(bx-ax)) / d
        return (ux, uy), math.hypot(ax-ux, ay-uy)

    def in_circle(c, r, p, eps=1e-9):
        return math.hypot(p[0]-c[0], p[1]-c[1]) <= r + eps

    c = (0.0, 0.0); r = 0.0
    for i, p in enumerate(pts):
        if in_circle(c, r, p):
            continue
        c, r = p, 0.0
        for j in range(i):
            q = pts[j]
            if in_circle(c, r, q):
                continue
            c, r = circle_two(p, q)
            for k in range(j):
                s = pts[k]
                if in_circle(c, r, s):
                    continue
                cc = circle_three(p, q, s)
                if cc is not None:
                    c, r = cc
    return c, r
