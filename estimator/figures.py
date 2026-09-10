"""
问题 1/2 的出图模块（论文插图）。

统一风格、英文/数学标注（保证任意环境可靠渲染），高分辨率 PNG + 矢量 PDF 双输出，
便于直接插入论文。运行：

    python -m estimator.figures            # 生成全部图到 estimator/figures/
    python -m estimator.figures --outdir some/dir --dpi 220

图清单：
  P1:
    p1_wedge_single         单检测点 ±1° 示向度楔形
    p1_two_wedges           两楔形之交（平行四边形定位区）
    p1_multi_region         多点之交（凸多边形 D）+ 直径 + 直径圆
    p1_cover_success        直径圆覆盖成功（钝/长条形 D）
    p1_cover_fail           直径圆覆盖失败（近等边三角形 D）+ 最小包围圆对比
    p1_shrink_vs_n          定位区域直径随检测点数下降
    p1_geometry_effect      基线/张开度对区域大小的影响
  P2:
    p2_L_vs_gamma           L–交会角曲线（90° 最优）
    p2_L_heat_gamma_r2      L 关于 (γ, r2) 的热力图
    p2_L_heat_r1_r2         L 关于 (r1, r2) 的热力图（γ=90°）
    p2_candidate_crescent   第二检测点候选月牙带（r1 未知）
    p2_good_vs_bad_second   好/坏第二点的定位区对比
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon
import numpy as np

from .geometry import (
    dir_vec, norm_deg, wedge_halfplanes, halfplane_intersection,
    polygon_diameter, diameter_circle_covers, min_enclosing_circle,
)
from .problem1 import locate_region, solve_problem1
from .problem2 import (
    localization_length_L, crossing_angle, second_point_candidates,
    recommend_second_point, region_diameter_two_points,
)


# ---------------- 全局样式 ----------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.framealpha": 0.9,
    "legend.fontsize": 9,
    "mathtext.fontset": "cm",
    "axes.axisbelow": True,
})

C_SOURCE = "#d62728"     # 真源
C_DET = "#1f77b4"        # 检测点
C_WEDGE = "#1f77b4"      # 楔形
C_REGION = "#2ca02c"     # 定位区域
C_DIAM = "#ff7f0e"       # 直径
C_CIRCLE = "#9467bd"     # 直径圆
C_MEC = "#8c564b"        # 最小包围圆
C_BAND = "#17becf"       # 候选带


def _save(fig, outdir: str, name: str):
    png = os.path.join(outdir, name + ".png")
    pdf = os.path.join(outdir, name + ".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def _draw_wedge(ax, s, svd, length=2600.0, delta=1.0, color=C_WEDGE,
                alpha=0.16, label=None):
    """在 ax 上把一个 ±delta 示向度楔形画成一个填充扇形（近似为三角形带）。"""
    lo = norm_deg(svd - delta)
    hi = norm_deg(svd + delta)
    # 用一系列角度采样扇形边界
    angs = np.linspace(math.radians(svd - delta), math.radians(svd + delta), 8)
    pts = [(s[0], s[1])]
    for a in angs:
        pts.append((s[0] + length * math.cos(a), s[1] + length * math.sin(a)))
    poly = MplPolygon(pts, closed=True, facecolor=color, alpha=alpha,
                      edgecolor=color, linewidth=0.8, label=label)
    ax.add_patch(poly)
    # 中心测向射线（虚线）
    cx, cy = dir_vec(svd)
    ax.plot([s[0], s[0] + length * cx], [s[1], s[1] + length * cy],
            color=color, lw=0.8, ls=":")


def _draw_polygon(ax, poly, color=C_REGION, alpha=0.35, label=None, lw=1.6):
    if not poly:
        return
    p = MplPolygon(poly, closed=True, facecolor=color, alpha=alpha,
                   edgecolor=color, linewidth=lw, label=label)
    ax.add_patch(p)


def _arena(ax, R=1800.0, alpha=0.9):
    ax.add_patch(Circle((0, 0), R, fill=False, ls="-.", lw=1.0,
                        edgecolor="0.55", alpha=alpha))


# ==================== P1 ====================
def fig_p1_wedge_single(outdir):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    S = (0.0, 0.0)
    P = (1200.0, 500.0)
    svd = norm_deg(math.degrees(math.atan2(P[1] - S[1], P[0] - S[0])))
    _draw_wedge(ax, S, svd, length=2200, delta=1.0,
                label=r"$\pm1^\circ$ bearing wedge $W$")
    ax.plot(*S, "^", color=C_DET, ms=12, label="detector $S$")
    ax.plot(*P, "*", color=C_SOURCE, ms=17, label="true source $P$")
    # 标注张角
    ax.annotate(r"half-angle $\delta=1^\circ$",
                xy=(S[0] + 900 * math.cos(math.radians(svd)),
                    S[1] + 900 * math.sin(math.radians(svd))),
                xytext=(300, 1300), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_title("P1: a single measurement gives a $\\pm1^\\circ$ wedge")
    ax.set_xlabel("x  (m,  East)"); ax.set_ylabel("y  (m,  North)")
    ax.set_xlim(-300, 2400); ax.set_ylim(-500, 1900)
    ax.set_aspect("equal"); ax.legend(loc="upper left")
    _save(fig, outdir, "p1_wedge_single")


def fig_p1_two_wedges(outdir):
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    P = (300.0, 400.0)
    dets = [(-1300.0, -900.0), (1200.0, -800.0)]
    detections = []
    for S in dets:
        svd = norm_deg(math.degrees(math.atan2(P[1] - S[1], P[0] - S[0])))
        detections.append((S[0], S[1], svd))
        _draw_wedge(ax, S, svd, length=3500, delta=1.0)
        ax.plot(S[0], S[1], "^", color=C_DET, ms=11)
    res = locate_region(detections)
    _draw_polygon(ax, res.polygon, label="localization region $D=W_1\\cap W_2$")
    ax.plot(*P, "*", color=C_SOURCE, ms=17, label="true source $P$")
    ax.plot([], [], "^", color=C_DET, label="detectors $S_1,S_2$")
    ax.set_title(f"P1: two wedges intersect to a small region  (diam $\\approx$ {res.diameter:.1f} m)")
    ax.set_xlabel("x  (m)"); ax.set_ylabel("y  (m)")
    ax.set_xlim(-1600, 1600); ax.set_ylim(-1200, 900)
    ax.set_aspect("equal"); ax.legend(loc="lower center")
    # 放大插图：定位区
    _inset_region(fig, ax, res.polygon, P)
    _save(fig, outdir, "p1_two_wedges")


def _inset_region(fig, ax, poly, P=None):
    """在右上角加一个定位区放大插图。"""
    if not poly:
        return
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.75 + 5
    axin = ax.inset_axes([0.63, 0.52, 0.35, 0.35])
    _draw_polygon(axin, poly)
    if P is not None:
        axin.plot(*P, "*", color=C_SOURCE, ms=10)
    axin.set_xlim(cx - span, cx + span); axin.set_ylim(cy - span, cy + span)
    axin.set_aspect("equal")
    axin.set_title("zoom: $D$", fontsize=8)
    axin.tick_params(labelsize=6)


def fig_p1_multi_region(outdir):
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    P = (-200.0, 350.0)
    dets = [(-1400., -1000.), (1300., -900.), (1000., 1200.), (-1300., 900.)]
    detections = []
    for S in dets:
        svd = norm_deg(math.degrees(math.atan2(P[1] - S[1], P[0] - S[0])))
        detections.append((S[0], S[1], svd))
        _draw_wedge(ax, S, svd, length=3600, delta=1.0, alpha=0.10)
        ax.plot(S[0], S[1], "^", color=C_DET, ms=10)
    res = locate_region(detections)
    _draw_polygon(ax, res.polygon, label="region $D=\\bigcap_i W_i$")
    a, b = res.diam_endpoints
    ax.plot([a[0], b[0]], [a[1], b[1]], "-o", color=C_DIAM, lw=2.0, ms=5,
            label=f"diameter = {res.diameter:.1f} m")
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    ax.add_patch(Circle(mid, res.diameter / 2, fill=False, lw=1.4,
                        edgecolor=C_CIRCLE, ls="--",
                        label="diameter-circle"))
    ax.plot(*P, "*", color=C_SOURCE, ms=16, label="true source $P$")
    ax.plot([], [], "^", color=C_DET, label="detectors")
    ax.set_title("P1: four measurements $\\Rightarrow$ convex polygon $D$")
    ax.set_xlabel("x  (m)"); ax.set_ylabel("y  (m)")
    ax.set_xlim(-1700, 1700); ax.set_ylim(-1300, 1500)
    ax.set_aspect("equal"); ax.legend(loc="lower left")
    _inset_region_diam(fig, ax, res)
    _save(fig, outdir, "p1_multi_region")


def _inset_region_diam(fig, ax, res):
    poly = res.polygon
    if not poly:
        return
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.7 + 3
    axin = ax.inset_axes([0.63, 0.06, 0.34, 0.34])
    _draw_polygon(axin, poly)
    a, b = res.diam_endpoints
    axin.plot([a[0], b[0]], [a[1], b[1]], "-o", color=C_DIAM, lw=1.6, ms=3)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    axin.add_patch(Circle(mid, res.diameter / 2, fill=False, lw=1.0,
                          edgecolor=C_CIRCLE, ls="--"))
    axin.set_xlim(cx - span, cx + span); axin.set_ylim(cy - span, cy + span)
    axin.set_aspect("equal"); axin.set_title("zoom", fontsize=8)
    axin.tick_params(labelsize=6)


def _cover_panel(ax, poly, title):
    diam, a, b = polygon_diameter(poly)
    covers, viol = diameter_circle_covers(poly, a, b)
    mc, mr = min_enclosing_circle(poly)
    _draw_polygon(ax, poly, label="region $D$")
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    ax.plot([a[0], b[0]], [a[1], b[1]], "-o", color=C_DIAM, lw=2, ms=5,
            label=f"diameter $={diam:.1f}$")
    ax.add_patch(Circle(mid, diam / 2, fill=False, lw=1.6, edgecolor=C_CIRCLE,
                        ls="--", label=f"diam-circle $r={diam/2:.1f}$"))
    ax.add_patch(Circle(mc, mr, fill=False, lw=1.4, edgecolor=C_MEC, ls=":",
                        label=f"min-enclosing $r={mr:.1f}$"))
    if viol:
        vx = [p[0] for p in viol]; vy = [p[1] for p in viol]
        ax.plot(vx, vy, "x", color="red", ms=11, mew=2.5,
                label="outside diam-circle")
    ax.set_aspect("equal")
    verdict = "covered" if covers else "NOT covered"
    ax.set_title(f"{title}\n(diam-circle {verdict})")
    ax.legend(loc="upper right", fontsize=8)


def fig_p1_cover_success(outdir):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    # 长条/钝角形状：直径圆一定覆盖
    poly = [(-120, -18), (130, -30), (150, 12), (-40, 40), (-140, 20)]
    _cover_panel(ax, poly, "P1: elongated region")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    _save(fig, outdir, "p1_cover_success")


def fig_p1_cover_fail(outdir):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    # 近等边三角形：Jung 反例，直径圆无法覆盖
    s = 120.0
    poly = [(0.0, 0.0), (s, 0.0), (s / 2, s * math.sqrt(3) / 2)]
    _cover_panel(ax, poly, "P1: near-equilateral region (counterexample)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.annotate("apex sits OUTSIDE\nthe diameter-circle",
                xy=(poly[2][0], poly[2][1]), xytext=(6, 66),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))
    _save(fig, outdir, "p1_cover_fail")


def fig_p1_shrink_vs_n(outdir):
    """定位区直径随检测点数量增加而收敛（多随机布局取中位）。"""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    rng = np.random.default_rng(2026)
    P = np.array([120.0, -260.0])
    ns = list(range(2, 11))
    trials = 60
    med, p90, lo = [], [], []
    for n in ns:
        di: list[float] = []
        for _ in range(trials):
            dets = []
            for _k in range(n):
                ang = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(1200, 1780)
                S = (r * math.cos(ang), r * math.sin(ang))
                svd = norm_deg(math.degrees(math.atan2(P[1] - S[1], P[0] - S[0]))
                               + rng.uniform(-1, 1))
                dets.append((S[0], S[1], svd))
            res = locate_region(dets)
            if not res.empty:
                di.append(res.diameter)
        di.sort()
        med.append(np.median(di)); p90.append(np.percentile(di, 90))
        lo.append(np.percentile(di, 10))
    ax.plot(ns, med, "-o", color=C_REGION, label="median diam(D)")
    ax.fill_between(ns, lo, p90, color=C_REGION, alpha=0.18, label="10–90 pct")
    ax.set_xlabel("number of detection points $n$")
    ax.set_ylabel("localization-region diameter (m)")
    ax.set_title("P1: region shrinks as measurements accumulate")
    ax.set_yscale("log"); ax.legend()
    _save(fig, outdir, "p1_shrink_vs_n")


def fig_p1_geometry_effect(outdir):
    """两点定位：基线张开角对区域直径的影响。"""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    P = (0.0, 0.0)
    r = 1500.0
    seps = np.linspace(5, 175, 60)   # 两检测点相对源的张开角（度）
    diover = []
    for sep in seps:
        S1 = (r * math.cos(math.radians(90 - sep / 2)),
              r * math.sin(math.radians(90 - sep / 2)))
        S2 = (r * math.cos(math.radians(90 + sep / 2)),
              r * math.sin(math.radians(90 + sep / 2)))
        svd1 = norm_deg(math.degrees(math.atan2(P[1] - S1[1], P[0] - S1[0])))
        svd2 = norm_deg(math.degrees(math.atan2(P[1] - S2[1], P[0] - S2[0])))
        d, poly = region_diameter_two_points(S1, svd1, S2, svd2)
        diover.append(d if d > 0 else np.nan)
    ax.plot(seps, diover, "-", color=C_DIAM, lw=2)
    ax.axvline(90, color="0.5", ls="--", lw=1)
    ax.annotate("best near $90^\\circ$ crossing", xy=(90, np.nanmin(diover)),
                xytext=(100, np.nanmin(diover) * 6), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_xlabel("angular separation of the two detectors seen from $P$  (deg)")
    ax.set_ylabel("region diameter (m)")
    ax.set_title("P1: two-point geometry — crossing angle drives accuracy")
    ax.set_yscale("log")
    _save(fig, outdir, "p1_geometry_effect")


# ==================== P2 ====================
def fig_p2_L_vs_gamma(outdir):
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    gammas = np.linspace(5, 175, 400)
    for r2, style in [(150, "-"), (300, "--"), (600, ":")]:
        L = [localization_length_L(1000, r2, g) for g in gammas]
        ax.plot(gammas, L, style, lw=2, label=f"$r_2={r2}$ m")
    ax.axvline(90, color="0.5", ls="--", lw=1)
    ax.set_xlabel("crossing angle $\\gamma$  (deg)")
    ax.set_ylabel("localization length  $L$  (m)")
    ax.set_title("P2: $L(\\gamma)=\\dfrac{2\\delta}{\\sin\\gamma}\\sqrt{r_1^2+r_2^2+2r_1r_2|\\cos\\gamma|}$   ($r_1=1000$)")
    ax.set_yscale("log"); ax.legend()
    ax.annotate("minimum at $\\gamma=90^\\circ$", xy=(90, localization_length_L(1000, 300, 90)),
                xytext=(95, localization_length_L(1000, 300, 90) * 4), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.4"))
    _save(fig, outdir, "p2_L_vs_gamma")


def fig_p2_L_heat_gamma_r2(outdir):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    gammas = np.linspace(20, 160, 200)
    r2s = np.linspace(80, 900, 200)
    G, R2 = np.meshgrid(gammas, r2s)
    L = np.vectorize(lambda g, r2: localization_length_L(1000, r2, g))(G, R2)
    pcm = ax.pcolormesh(G, R2, L, shading="auto", cmap="viridis",
                        vmax=np.percentile(L, 96))
    cs = ax.contour(G, R2, L, levels=8, colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    fig.colorbar(pcm, ax=ax, label="$L$ (m)")
    ax.axvline(90, color="white", ls="--", lw=1)
    ax.set_xlabel("crossing angle $\\gamma$ (deg)")
    ax.set_ylabel("second-point distance $r_2$ (m)")
    ax.set_title("P2: $L(\\gamma, r_2)$  at $r_1=1000$ m  (lower = better)")
    _save(fig, outdir, "p2_L_heat_gamma_r2")


def fig_p2_L_heat_r1_r2(outdir):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    r1s = np.linspace(200, 1800, 200)
    r2s = np.linspace(80, 900, 200)
    R1, R2 = np.meshgrid(r1s, r2s)
    L = np.vectorize(lambda r1, r2: localization_length_L(r1, r2, 90.0))(R1, R2)
    pcm = ax.pcolormesh(R1, R2, L, shading="auto", cmap="magma",
                        vmax=np.percentile(L, 97))
    cs = ax.contour(R1, R2, L, levels=8, colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    fig.colorbar(pcm, ax=ax, label="$L$ (m)")
    ax.set_xlabel("first-point distance $r_1$ (m)")
    ax.set_ylabel("second-point distance $r_2$ (m)")
    ax.set_title("P2: $L(r_1, r_2)$ at $\\gamma=90^\\circ$  — small $r_2$ dominates")
    _save(fig, outdir, "p2_L_heat_r1_r2")


def fig_p2_candidate_crescent(outdir):
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    _arena(ax)
    S1 = (-1200.0, -1300.0)
    svd1 = 42.0
    ux, uy = dir_vec(svd1)
    # svd1 楔形
    _draw_wedge(ax, S1, svd1, length=3600, delta=1.0, alpha=0.12,
                label="$\\pm1^\\circ$ wedge from $S_1$")
    ax.plot(*S1, "^", color=C_DET, ms=12, label="first point $S_1$")
    # 沿 svd1 的可能源位置
    r1s = list(np.linspace(400, 2600, 40))
    Ps = [(S1[0] + ux * r1, S1[1] + uy * r1) for r1 in r1s]
    ax.plot([p[0] for p in Ps], [p[1] for p in Ps], "-", color=C_SOURCE,
            lw=1.4, alpha=0.6, label="possible sources along $svd_1$")

    # 候选第二点区域：r2 在 [150,500] 之间扫，得到两侧的填充"月牙带"
    r2_lo, r2_hi = 150.0, 500.0
    first = True
    for sgn in (+1, -1):
        lo = second_point_candidates(S1, svd1, r1_values=r1s, r2=r2_lo,
                                    arena_radius=1800.0, side=sgn)
        hi = second_point_candidates(S1, svd1, r1_values=r1s, r2=r2_hi,
                                    arena_radius=1800.0, side=sgn)
        if lo and hi:
            band = [c.s2 for c in lo] + [c.s2 for c in reversed(hi)]
            poly = MplPolygon(band, closed=True, facecolor=C_BAND, alpha=0.30,
                              edgecolor=C_BAND, lw=1.2,
                              label=("candidate $S_2$ region\n($r_2\\in[150,500]$ m)"
                                     if first else None))
            ax.add_patch(poly)
            first = False
    # 中位推荐点（r1 中值猜测）
    rec = recommend_second_point(S1, svd1, r1_guess=1500.0, r2=300.0)
    ax.plot(*rec.s2, "s", color="#ff7f0e", ms=11,
            label=f"recommended $S_2$ ($\\gamma={rec.gamma_deg:.0f}^\\circ$)")

    ax.set_title("P2: second-point candidate region (since $r_1$ unknown)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_xlim(-1950, 1950); ax.set_ylim(-1950, 1950)
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    _save(fig, outdir, "p2_candidate_crescent")


def fig_p2_good_vs_bad_second(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6))
    P = (250.0, 300.0)
    S1 = (-1200.0, -1250.0)
    svd1 = norm_deg(math.degrees(math.atan2(P[1] - S1[1], P[0] - S1[0])))
    # 好第二点：交会角≈90
    good = recommend_second_point(S1, svd1, r1_guess=math.hypot(P[0]-S1[0], P[1]-S1[1]),
                                  r2=300.0)
    S2g = good.s2
    # 坏第二点：几乎与 S1 共线（交会角很小）
    ux, uy = dir_vec(svd1)
    S2b = (S1[0] + ux * (math.hypot(P[0]-S1[0], P[1]-S1[1]) + 300),
           S1[1] + uy * (math.hypot(P[0]-S1[0], P[1]-S1[1]) + 300))
    for ax, S2, tag in ((axes[0], S2g, "good ($\\gamma\\approx90^\\circ$)"),
                        (axes[1], S2b, "bad (near-collinear)")):
        svd2 = norm_deg(math.degrees(math.atan2(P[1] - S2[1], P[0] - S2[0])))
        _draw_wedge(ax, S1, svd1, length=4200, delta=1.0, alpha=0.12)
        _draw_wedge(ax, S2, svd2, length=4200, delta=1.0, alpha=0.12, color="#ff7f0e")
        d, poly = region_diameter_two_points(S1, svd1, S2, svd2)
        g = crossing_angle(S1, S2, P)
        _draw_polygon(ax, poly, label=f"region  diam={d:.1f} m")
        ax.plot(*S1, "^", color=C_DET, ms=11)
        ax.plot(*S2, "s", color="#ff7f0e", ms=10)
        ax.plot(*P, "*", color=C_SOURCE, ms=15)
        ax.set_title(f"{tag}\n$\\gamma={g:.1f}^\\circ$, diam$={d:.0f}$ m")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_xlim(-1700, 1700); ax.set_ylim(-1700, 1400)
        ax.set_aspect("equal"); ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("P2: choosing $S_2$ for a near-orthogonal crossing shrinks the region")
    _save(fig, outdir, "p2_good_vs_bad_second")


ALL_FIGS = [
    fig_p1_wedge_single, fig_p1_two_wedges, fig_p1_multi_region,
    fig_p1_cover_success, fig_p1_cover_fail, fig_p1_shrink_vs_n,
    fig_p1_geometry_effect,
    fig_p2_L_vs_gamma, fig_p2_L_heat_gamma_r2, fig_p2_L_heat_r1_r2,
    fig_p2_candidate_crescent, fig_p2_good_vs_bad_second,
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成 P1/P2 论文插图")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    plt.rcParams["savefig.dpi"] = args.dpi
    print(f"[figures] outdir = {args.outdir}")
    for fn in ALL_FIGS:
        fn(args.outdir)
    print(f"[figures] done: {len(ALL_FIGS)} figures (PNG + PDF each)")


if __name__ == "__main__":
    main()
