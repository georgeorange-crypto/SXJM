"""
estimator 自测：几何 + 问题1 + 问题2。

运行： python -m estimator.test_estimator
覆盖：单元（已知形状/闭式值）、性质/不变量（随机真源必落在定位区内）、边界、
      与一阶公式的一致性校核。
"""

from __future__ import annotations

import math
import random

from .geometry import (
    norm_deg, dir_vec, cross, dot, wedge_halfplanes, halfplane_intersection,
    polygon_diameter, diameter_circle_covers, convex_hull, min_enclosing_circle,
    Halfplane,
)
from .problem1 import locate_region, solve_problem1, LocateResult
from .problem2 import (
    localization_length_L, crossing_angle, second_point_candidates,
    recommend_second_point, region_diameter_two_points,
)

_PASS = 0
_FAIL = 0


def check(cond: bool, msg: str):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        print(f"  FAIL: {msg}")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _bearing(sx, sy, px, py):
    return norm_deg(math.degrees(math.atan2(py - sy, px - sx)))


def _point_in_poly(poly, x, y, eps=1e-6):
    """凸多边形内含判定（顶点逆/顺时针都行：用符号一致性）。"""
    n = len(poly)
    if n < 3:
        return False
    signs = []
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cr = cross(bx - ax, by - ay, x - ax, y - ay)
        signs.append(cr)
    pos = all(s >= -eps for s in signs)
    neg = all(s <= eps for s in signs)
    return pos or neg


# ---------------- A. 几何单元 ----------------
def test_geometry_basics():
    check(approx(norm_deg(-10), 350), "norm_deg(-10)=350")
    check(approx(norm_deg(370), 10), "norm_deg(370)=10")
    check(approx(norm_deg(360), 0), "norm_deg(360)=0")
    vx, vy = dir_vec(0)
    check(approx(vx, 1) and approx(vy, 0), "dir_vec(0)=E")
    vx, vy = dir_vec(90)
    check(approx(vx, 0, 1e-9) and approx(vy, 1), "dir_vec(90)=N")


def test_wedge_contains():
    rng = random.Random(1)
    ok = True
    for _ in range(2000):
        sx, sy = rng.uniform(-1800, 1800), rng.uniform(-1800, 1800)
        px, py = rng.uniform(-1800, 1800), rng.uniform(-1800, 1800)
        if math.hypot(px - sx, py - sy) < 1e-6:
            continue
        tb = _bearing(sx, sy, px, py)
        err = rng.uniform(-1, 1)
        svd = norm_deg(tb + err)   # 观测=真值+误差；真源相对观测偏 -err∈[-1,1]
        hps = wedge_halfplanes(sx, sy, svd, 1.0)
        if not all(hp.contains(px, py) for hp in hps):
            ok = False
            break
    check(ok, "真源在 ±1° 楔形内（误差 ∈[-1,1]）")

    # 偏差 >1° 必在楔形外
    sx, sy = 0.0, 0.0
    tb = 30.0
    off = (2000 * math.cos(math.radians(tb + 1.5)),
           2000 * math.sin(math.radians(tb + 1.5)))
    hps = wedge_halfplanes(sx, sy, tb, 1.0)
    check(not all(hp.contains(off[0], off[1]) for hp in hps),
          "偏 1.5° 的点在 ±1° 楔形外")


def test_halfplane_intersection_square():
    # 四个半平面围成 [-1,1]^2
    hps = [Halfplane(1, 0, 1), Halfplane(-1, 0, 1),
           Halfplane(0, 1, 1), Halfplane(0, -1, 1)]
    poly = halfplane_intersection(hps, bound=10)
    check(len(poly) == 4, "方形交=4 顶点")
    diam, a, b = polygon_diameter(poly)
    check(approx(diam, 2 * math.sqrt(2), 1e-6), f"方形直径=2√2, got {diam}")


def test_convex_hull_and_diameter():
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5), (0.3, 0.2)]
    h = convex_hull(pts)
    check(len(h) == 4, "凸包去掉内部点→4 顶点")
    diam, a, b = polygon_diameter(h)
    check(approx(diam, math.sqrt(2)), "单位方直径=√2")


def test_diameter_circle_cover_cases():
    # 钝角三角形：最长边即直径，必覆盖
    poly = [(0.0, 0.0), (200.0, 0.0), (50.0, 20.0)]
    d, a, b = polygon_diameter(poly)
    cov, viol = diameter_circle_covers(poly, a, b)
    check(cov, "钝角三角形直径圆可覆盖")
    # 等边三角：不覆盖（Jung）
    s = 100.0
    tri = [(0, 0), (s, 0), (s / 2, s * math.sqrt(3) / 2)]
    d, a, b = polygon_diameter(tri)
    cov, viol = diameter_circle_covers(tri, a, b)
    check(not cov and len(viol) >= 1, "等边三角直径圆不覆盖（反例）")
    # 最小包围圆半径 = s/√3 > s/2
    _, mr = min_enclosing_circle(tri)
    check(approx(mr, s / math.sqrt(3), 1e-3), f"等边三角最小包围圆 r=s/√3, got {mr}")
    check(mr > d / 2, "最小包围圆半径 > 直径/2")


def test_min_enclosing_circle_known():
    c, r = min_enclosing_circle([(-3, 0), (3, 0), (0, 1)])
    check(approx(c[0], 0, 1e-6) and approx(r, 3, 1e-6),
          f"MEC of wide triangle: center x=0 r=3, got c={c} r={r}")


# ---------------- B. 问题1 不变量 ----------------
def test_p1_source_always_inside():
    """核心不变量：任意 ±1° 误差下，真源必落在定位区 D 内。"""
    rng = random.Random(7)
    ok = 0
    tot = 0
    for _ in range(1500):
        px = rng.uniform(-1600, 1600)
        py = rng.uniform(-1600, 1600)
        if math.hypot(px, py) > 1700:
            continue
        n = rng.randint(2, 6)
        dets = []
        for _k in range(n):
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(1200, 1780)
            sx, sy = r * math.cos(ang), r * math.sin(ang)
            tb = _bearing(sx, sy, px, py)
            svd = norm_deg(tb + rng.uniform(-1, 1))
            dets.append((sx, sy, svd))
        res = locate_region(dets)
        if res.empty:
            continue
        tot += 1
        if _point_in_poly(res.polygon, px, py, eps=1e-3):
            ok += 1
    check(tot > 0 and ok == tot, f"真源必在 D 内：{ok}/{tot}")


def test_p1_shrinks_with_more_points():
    rng = random.Random(11)
    P = (200.0, -300.0)
    def diam_for(n):
        ds = []
        for _ in range(40):
            dets = []
            for _k in range(n):
                ang = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(1200, 1780)
                sx, sy = r * math.cos(ang), r * math.sin(ang)
                svd = norm_deg(_bearing(sx, sy, *P) + rng.uniform(-1, 1))
                dets.append((sx, sy, svd))
            res = locate_region(dets)
            if not res.empty:
                ds.append(res.diameter)
        ds.sort()
        return ds[len(ds) // 2]
    d2 = diam_for(2)
    d6 = diam_for(6)
    check(d6 <= d2, f"中位直径随点数不增：n=2→{d2:.1f}, n=6→{d6:.1f}")


def test_p1_empty_on_contradiction():
    # 两个从同一点出发、方向相差很大的示向度 → 交为空/退化
    dets = [(0.0, 0.0, 10.0), (0.0, 0.0, 200.0)]
    res = locate_region(dets, clip_circle_radius=1800.0)
    check(res.empty or res.diameter < 1e-6 or len(res.polygon) == 0,
          "同点矛盾示向度 → 空/退化")


def test_p1_solve_zero_error():
    P = (150.0, 220.0)
    dets = [(-1500, -1000), (1400, -900), (0, 1600)]
    res = solve_problem1(P, dets, error_field=None)
    check(not res.empty, "零误差可解")
    check(_point_in_poly(res.polygon, P[0], P[1], eps=1e-2), "零误差真源在 D 内")
    # 各楔形仍有 ±1° 张角，故区域非零；距离 ~1500 m、±1° → 直径量级几十米
    check(res.diameter < 150, f"零误差 3 点定位区受 ±1° 限制而有界: {res.diameter:.2f}")


# ---------------- C. 问题2 ----------------
def test_p2_L_min_at_90():
    Ls = {g: localization_length_L(1000, 300, g) for g in range(10, 171, 5)}
    gmin = min(Ls, key=Ls.get)
    check(gmin == 90, f"L 在 γ=90° 最小, got {gmin}")


def test_p2_L_monotone_in_r2():
    prev = -1.0
    ok = True
    for r2 in (50, 100, 200, 400, 800):
        L = localization_length_L(1000, r2, 90.0)
        if L < prev:
            ok = False
        prev = L
    check(ok, "γ=90° 时 L 随 r2 单调增")


def test_p2_crossing_angle():
    # S1 沿 +x 看 P；S2 在 P 正上方 → 两视线正交
    g = crossing_angle((0, 0), (1000, 1000), (1000, 0))
    check(approx(g, 90, 1e-6), f"crossing_angle 正交=90, got {g}")
    # 共线 → 0°
    g0 = crossing_angle((0, 0), (500, 0), (1000, 0))
    check(approx(g0, 0, 1e-6), f"crossing_angle 同侧共线=0, got {g0}")


def test_p2_first_order_matches_geometry():
    """一阶 L 与两楔形真实交区直径应同量级且接近。"""
    P = (250.0, 300.0)
    S1 = (-1200.0, -1250.0)
    svd1 = _bearing(*S1, *P)
    rec = recommend_second_point(S1, svd1,
                                 r1_guess=math.hypot(P[0] - S1[0], P[1] - S1[1]),
                                 r2=300.0)
    S2 = rec.s2
    svd2 = _bearing(*S2, *P)
    d, poly = region_diameter_two_points(S1, svd1, S2, svd2)
    check(abs(d - rec.L) <= 0.15 * rec.L,
          f"一阶 L={rec.L:.2f} 与真实交区直径={d:.2f} 接近")
    check(abs(rec.gamma_deg - 90.0) < 1e-6, "推荐点交会角=90°")


def test_p2_candidates_in_arena():
    S1 = (-1000.0, -1100.0)
    cands = second_point_candidates(S1, 40.0, r2=300.0, arena_radius=1800.0)
    check(len(cands) > 0, "候选点非空")
    check(all(math.hypot(c.s2[0], c.s2[1]) <= 1800.0 + 1e-6 for c in cands),
          "候选点都在区域圆内")


def main():
    tests = [
        test_geometry_basics, test_wedge_contains,
        test_halfplane_intersection_square, test_convex_hull_and_diameter,
        test_diameter_circle_cover_cases, test_min_enclosing_circle_known,
        test_p1_source_always_inside, test_p1_shrinks_with_more_points,
        test_p1_empty_on_contradiction, test_p1_solve_zero_error,
        test_p2_L_min_at_90, test_p2_L_monotone_in_r2, test_p2_crossing_angle,
        test_p2_first_order_matches_geometry, test_p2_candidates_in_arena,
    ]
    print(f"estimator 自测：{len(tests)} 组")
    for t in tests:
        t()
    print(f"\n结果：{_PASS} 通过 / {_FAIL} 失败")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
