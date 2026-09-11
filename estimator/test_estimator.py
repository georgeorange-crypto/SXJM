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
    first_feasible_set, reception_guaranteed, worst_case_clearance_radius,
    minimax_second_point, analytic_second_point_tokekar, P2Status,
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


# ---------------- D. 问题2 第2层：Minimax NBV ----------------
def _omega1_dense_samples(poly, k=12):
    """凸多边形 Ω₁ 的稠密采样（顶点 + 边上等分 + 向形心的收缩），用于暴力校核。"""
    n = len(poly)
    if n == 0:
        return []
    samples = list(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        for t in range(1, k):
            f = t / k
            samples.append((ax + (bx - ax) * f, ay + (by - ay) * f))
    cx = sum(p[0] for p in poly) / n
    cy = sum(p[1] for p in poly) / n
    for (x, y) in list(samples):
        for f in (0.25, 0.5, 0.75):
            samples.append((cx + (x - cx) * f, cy + (y - cy) * f))
    return samples


def _brute_reception(s2, s1, poly, recv_min=1000.0, tol=1e-6):
    """暴力判定：所有采样 G 都满足 ‖S2−G‖ ≤ max(recv_min, ‖G−S1‖)。"""
    for (gx, gy) in _omega1_dense_samples(poly):
        d = math.hypot(s2[0] - gx, s2[1] - gy)
        if d > max(recv_min, math.hypot(gx - s1[0], gy - s1[1])) + tol:
            return False
    return True


def test_p2_first_feasible_set_contains_consistent_sources():
    """Ω₁（外切多边形超集）必须包含所有与首次示向度相容的可能源。"""
    S1 = (0.0, 0.0)
    svd1 = 90.0
    om = first_feasible_set(S1, svd1)
    check(not om.empty and len(om.poly) >= 3, "Ω₁ 非空且为多边形")
    check(om.r_max <= 1500.0 / math.cos(math.pi / 128) + 1.0,
          f"r_max ≤ 1500(+外切余量), got {om.r_max:.2f}")
    check(om.mec_radius <= om.r_max + 1e-6, "MEC(Ω₁) 半径 ≤ r_max")
    rng = random.Random(5)
    ok = 0
    tot = 0
    for _ in range(1500):
        err = rng.uniform(-1.0, 1.0)
        r1 = rng.uniform(10.0, 1490.0)
        ux, uy = dir_vec(norm_deg(svd1 + err))
        P = (S1[0] + ux * r1, S1[1] + uy * r1)
        tot += 1
        if _point_in_poly(om.poly, P[0], P[1], eps=1e-3):
            ok += 1
    check(ok == tot, f"相容源都在 Ω₁ 内（外近似超集）：{ok}/{tot}")


def test_p2_reception_guaranteed_exact():
    """精确 reception_guaranteed 判据 vs 暴力采样：认证必须是充分的。"""
    S1 = (1700.0, 0.0)
    om = first_feasible_set(S1, 90.0)
    check(reception_guaranteed(S1, om), "S1 恒在保证接收域 𝒞_recv 内")
    rng = random.Random(3)
    sound = True
    agree = 0
    tot = 0
    for _ in range(400):
        s2 = (rng.uniform(S1[0] - 1200, S1[0] + 1200), rng.uniform(-1200, 1200))
        ex = reception_guaranteed(s2, om)
        br = _brute_reception(s2, S1, om.poly)
        if ex and not br:            # 认证可行却被暴力发现违反 → 不安全，绝不允许
            sound = False
        tot += 1
        if ex == br:
            agree += 1
    check(sound, "reception_guaranteed 可行认证充分（exact ⇒ brute）")
    check(agree >= int(0.95 * tot), f"exact 与 brute 判定基本一致：{agree}/{tot}")


def test_p2_worst_case_clearance_bounded_by_mec():
    """J(S2)=最坏 MEC(Ω₂) 半径，因 Ω₂⊆Ω₁ 必落在 [0, MEC(Ω₁)]。"""
    om = first_feasible_set((0.0, 0.0), 90.0)
    rng = random.Random(9)
    ok = True
    for _ in range(30):
        s2 = (rng.uniform(-1500, 1500), rng.uniform(-1500, 1500))
        J, _phi, _poly = worst_case_clearance_radius(s2, om, phi_step_deg=1.0, refine=False)
        if not (0.0 <= J <= om.mec_radius + 1e-6):
            ok = False
            break
    check(ok, "J(S2) ∈ [0, MEC(Ω₁)]")


def test_p2_minimax_near_vs_far():
    """远源(r 可达 1500)不可一次清除；近源(受竞技场裁剪)可一次清除。"""
    far = minimax_second_point((0.0, 0.0), 90.0, grid_n=11)
    near = minimax_second_point((1700.0, 0.0), 90.0, grid_n=11)
    for rec, name in ((far, "far"), (near, "near")):
        check(rec.status == P2Status.OK and rec.feasible, f"{name}: status OK 且可行")
        check(rec.j_star <= rec.rho_star + 1e-6, f"{name}: J* ≤ ρ*(Ω₁)")
        check(reception_guaranteed(rec.s2_star, rec.omega1), f"{name}: S2* ∈ 𝒞_recv")
        check((rec.j_star <= rec.clear_radius + 1e-6) == rec.clearable_one_move,
              f"{name}: clearable_one_move 与 J*≤20 一致")
    check(near.j_star < far.j_star,
          f"近源 J* 更小: near={near.j_star:.1f} far={far.j_star:.1f}")
    check(near.clearable_one_move and len(near.clearable_region) > 0,
          "近源存在一次清除区 𝒞_clear")
    check(not far.clearable_one_move, "远源无法保证一次清除（需三次测向）")
    check(all(p in near.reception_region for p in near.clearable_region),
          "𝒞_clear ⊆ 𝒞_recv")


def test_p2_analytic_reference_formula():
    """Tokekar 解析参考点 S_ana = S1 + mid·u ± half·n 的坐标公式。"""
    S1 = (100.0, -50.0)
    Sp = analytic_second_point_tokekar(S1, 90.0, 100.0, 500.0, +1)  # u=(0,1) n=(-1,0)
    Sm = analytic_second_point_tokekar(S1, 90.0, 100.0, 500.0, -1)  # mid=300 half=200
    check(approx(Sp[0], S1[0] - 200.0, 1e-6) and approx(Sp[1], S1[1] + 300.0, 1e-6),
          f"S_ana+ 公式, got {Sp}")
    check(approx(Sm[0], S1[0] + 200.0, 1e-6) and approx(Sm[1], S1[1] + 300.0, 1e-6),
          f"S_ana- 公式, got {Sm}")


def main():
    tests = [
        test_geometry_basics, test_wedge_contains,
        test_halfplane_intersection_square, test_convex_hull_and_diameter,
        test_diameter_circle_cover_cases, test_min_enclosing_circle_known,
        test_p1_source_always_inside, test_p1_shrinks_with_more_points,
        test_p1_empty_on_contradiction, test_p1_solve_zero_error,
        test_p2_L_min_at_90, test_p2_L_monotone_in_r2, test_p2_crossing_angle,
        test_p2_first_order_matches_geometry, test_p2_candidates_in_arena,
        test_p2_first_feasible_set_contains_consistent_sources,
        test_p2_reception_guaranteed_exact,
        test_p2_worst_case_clearance_bounded_by_mec,
        test_p2_minimax_near_vs_far, test_p2_analytic_reference_formula,
    ]
    print(f"estimator 自测：{len(tests)} 组")
    for t in tests:
        t()
    print(f"\n结果：{_PASS} 通过 / {_FAIL} 失败")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
