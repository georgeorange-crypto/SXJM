"""
扫描点布置与覆盖保证（本题“确保清除所有源”的理论支柱）。

为什么需要“覆盖”而不是随便扫几下：干扰源【个数未知】。要断言“某频道无源”，
必须保证——若该频道真有源（在区域内任意位置、R_eff 任意 ∈[1000,1500]、朝向任意），
我的扫描点集合里【至少有一个能收到它】。否则漏检且无从知晓，无法“确保全清”。

两类保证（均按最坏 R_eff=1000 设计并数值校验）：

1) 全向（第三问）—— 1-覆盖：
   区域内任意点 s，都存在扫描点 p 使 dist(p,s) ≤ 1000。
   ⟺ 用半径 1000 的圆盘覆盖半径 1800 的圆盘。给出 圆心+K 环 的小集合并验证。

2) 定向（第四问）—— 3-覆盖（角度环绕）：
   定向源覆盖角是过源的某个 180° 半平面（朝向未知）。要保证任意朝向都能被某扫描点收到，
   需在源周围、距离 ≤1000 内有若干扫描点，其“从源看出去的方位角”最大圆缺口 < 180°
   （等价于源落在这些点凸包内部）。这样任何 180° 盲区都挡不住全部扫描点。
   用三角格（边长 h<1000）裁剪到区域即可满足：区域内任意点必落在某个顶点距其 <h≤1000
   的小三角形内部 → 三顶点方位缺口 <180°。数值校验最大缺口。

本模块只做几何+校验，stdlib only。运行 `python -m jammerhunt.coverage` 打印两套点集与校验结论。
"""

from __future__ import annotations

import math
from typing import Optional

Point = tuple[float, float]

ARENA_RADIUS_M = 1800.0
WORST_REFF_M = 1000.0                    # 最坏有效接收半径（附件2 §2.1 下界）
DIR_HALF_ANGLE_DEG = 90.0


# --------------------------------------------------------------------------- #
# 角度/几何小工具（与 geometry.py 保持一致，但本模块自足以便独立校验）
# --------------------------------------------------------------------------- #
def _norm(a: float) -> float:
    a = math.fmod(a, 360.0)
    return a + 360.0 if a < 0 else a


def _bearing(p_from: Point, p_to: Point) -> float:
    return _norm(math.degrees(math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0])))


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def max_angular_gap(bearings: list[float]) -> float:
    """一组方位角（度）排序后的最大圆缺口（相邻角差，含首尾环绕）。返回 [0,360]。"""
    if not bearings:
        return 360.0
    if len(bearings) == 1:
        return 360.0
    b = sorted(_norm(x) for x in bearings)
    gap = 0.0
    for i in range(1, len(b)):
        gap = max(gap, b[i] - b[i - 1])
    gap = max(gap, 360.0 - b[-1] + b[0])   # 环绕缺口
    return gap


# --------------------------------------------------------------------------- #
# 采样（校验用）：半径 1800 圆盘上的稠密极坐标网格
# --------------------------------------------------------------------------- #
def _disk_samples(R: float = ARENA_RADIUS_M, dr: float = 25.0, dtheta_deg: float = 2.0) -> list[Point]:
    pts: list[Point] = [(0.0, 0.0)]
    r = dr
    while r <= R + 1e-9:
        n = max(6, int(round(360.0 / dtheta_deg)))
        for i in range(n):
            t = math.radians(i * 360.0 / n)
            pts.append((r * math.cos(t), r * math.sin(t)))
        r += dr
    return pts


# --------------------------------------------------------------------------- #
# 覆盖校验
# --------------------------------------------------------------------------- #
def verify_one_cover(points: list[Point], R: float = ARENA_RADIUS_M,
                     reff: float = WORST_REFF_M, dr: float = 25.0,
                     dtheta_deg: float = 2.0) -> tuple[bool, float]:
    """
    1-覆盖校验：返回 (是否每个采样点都在某扫描点 reff 内, 最坏“到最近扫描点距离”)。
    最坏距离 ≤ reff 即通过。
    """
    worst = 0.0
    for s in _disk_samples(R, dr, dtheta_deg):
        dmin = min(_dist(s, p) for p in points)
        worst = max(worst, dmin)
    return worst <= reff + 1e-9, worst


def verify_three_cover(points: list[Point], R: float = ARENA_RADIUS_M,
                       reff: float = WORST_REFF_M, dr: float = 25.0,
                       dtheta_deg: float = 2.0,
                       gap_limit_deg: float = 180.0) -> tuple[bool, float]:
    """
    3-覆盖（角度环绕）校验：对每个采样点 s，取 reff 内的扫描点，算它们“从 s 看”的方位角
    最大圆缺口。返回 (是否全部 < gap_limit, 最坏(最大)缺口)。
    缺口 < 180° ⟹ 任意 180° 盲区都挡不住全部扫描点 ⟹ 定向源必被至少一个点收到。
    """
    worst_gap = 0.0
    for s in _disk_samples(R, dr, dtheta_deg):
        near = [p for p in points if _dist(s, p) <= reff + 1e-9 and _dist(s, p) > 1e-6]
        if len(near) < 3:
            return False, 360.0
        bearings = [_bearing(s, p) for p in near]
        worst_gap = max(worst_gap, max_angular_gap(bearings))
    return worst_gap < gap_limit_deg - 1e-6, worst_gap


# --------------------------------------------------------------------------- #
# 构造：全向 1-覆盖（圆心 + K 环），搜索满足余量的最少点集
# --------------------------------------------------------------------------- #
def _ring(k: int, radius: float, phase_deg: float = 0.0) -> list[Point]:
    return [(radius * math.cos(math.radians(phase_deg + i * 360.0 / k)),
             radius * math.sin(math.radians(phase_deg + i * 360.0 / k)))
            for i in range(k)]


def omni_scan_points(R: float = ARENA_RADIUS_M, reff: float = WORST_REFF_M,
                     margin_m: float = 50.0) -> list[Point]:
    """
    返回全向 1-覆盖扫描点（圆心 + 单环），在 K=6,7,8 与环半径上搜索，
    取“最坏最近距离 ≤ reff-margin”的最少点集。找不到则回退到 K=8 稠环。
    """
    target = reff - margin_m
    best: Optional[list[Point]] = None
    for k in (6, 7, 8):
        a = 0.60 * R
        while a <= R - 20.0:
            pts = [(0.0, 0.0)] + _ring(k, a)
            ok, worst = verify_one_cover(pts, R, reff)
            if worst <= target:
                if best is None or len(pts) < len(best):
                    best = pts
                break
            a += 20.0
        if best is not None:
            break
    if best is None:                       # 保险：更密的双环
        best = [(0.0, 0.0)] + _ring(8, 0.62 * R) + _ring(8, 0.90 * R, phase_deg=22.5)
    return best


# --------------------------------------------------------------------------- #
# 构造：定向 3-覆盖（三角格裁剪），搜索满足余量的最大边长（→最少点）
# --------------------------------------------------------------------------- #
def _triangular_lattice(h: float, Rmax: float) -> list[Point]:
    """以原点为格点的三角格，裁剪到半径 Rmax 内。"""
    pts: list[Point] = []
    dy = h * math.sqrt(3.0) / 2.0
    jmax = int(math.ceil(Rmax / dy)) + 1
    for j in range(-jmax, jmax + 1):
        y = j * dy
        x_off = (h / 2.0) if (j % 2) else 0.0
        imax = int(math.ceil((Rmax + abs(x_off)) / h)) + 1
        for i in range(-imax, imax + 1):
            x = i * h + x_off
            if x * x + y * y <= Rmax * Rmax + 1e-6:
                pts.append((x, y))
    return pts


def directional_scan_points(src_radius: float = ARENA_RADIUS_M,
                            reff: float = WORST_REFF_M,
                            gap_margin_deg: float = 6.0) -> list[Point]:
    """
    返回定向 3-覆盖扫描点（三角格裁剪）。

    设计按【物理下界 reff=1000】——真实 R_eff≥1000，用 1000 设计对任何真实 R_eff 都成立
    （更大的 R_eff 只会让覆盖更容易），无需再留 reff 余量；仅对角度缺口留 gap_margin。

    src_radius：假定干扰源落在半径 ≤ src_radius 的范围内。默认 1800（全盘、最保守）。
    注：定向源越贴边、朝向越向外，其覆盖半平面越是落在区域之外——极限情形任何区域内策略
    都无法探测（该源对机器狗“不可见”）。故可用 src_radius<1800 得到更省的点集（见思路.md）。

    搜索：边长 h 由大到小（点少→多），取首个通过“3 点均在 reff 内、最大缺口 < 180-gap_margin”
    的 h。裁剪半径 src_radius+0.55h，仅比源范围外扩不到一格，避免无谓外圈点。
    """
    limit = 180.0 - gap_margin_deg
    best: Optional[list[Point]] = None
    h = min(reff, 0.5 * reff + 400.0)      # 起点略低于 reff，避免顶点环恰卡在探测边界
    while h >= 500.0:
        pts = _triangular_lattice(h, src_radius + 0.55 * h)
        ok3, worst_gap = verify_three_cover(pts, src_radius, reff, dr=20.0, dtheta_deg=2.0,
                                            gap_limit_deg=limit)
        if ok3:
            best = pts
            break
        h -= 20.0
    if best is None:                       # 保险：很密的格
        best = _triangular_lattice(600.0, src_radius + 400.0)
    return best


# --------------------------------------------------------------------------- #
# 自检打印
# --------------------------------------------------------------------------- #
def _summary() -> str:
    omni = omni_scan_points()
    ok1, worst1 = verify_one_cover(omni)
    dir_pts = directional_scan_points()
    ok3, gap3 = verify_three_cover(dir_pts, reff=WORST_REFF_M, gap_limit_deg=180.0)
    lines = [
        "== 全向 1-覆盖（第三问扫描点）==",
        f"  点数 = {len(omni)}",
        f"  最坏“到最近扫描点”距离 = {worst1:.1f} m  (需 ≤ {WORST_REFF_M:.0f})  → {'OK' if ok1 else 'FAIL'}",
        "",
        "== 定向 3-覆盖（第四问扫描点）==",
        f"  点数 = {len(dir_pts)}",
        f"  最坏方位缺口 = {gap3:.1f}°  (需 < 180°)  → {'OK' if ok3 else 'FAIL'}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(_summary())
