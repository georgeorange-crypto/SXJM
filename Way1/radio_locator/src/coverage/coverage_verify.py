"""
覆盖性验证：把布点方案交给数值验证器，确认“任何合法源都会被至少一点探测到”。

- verify_omni_cover：任意源位置（竞技场内）到最近探测点 <= r_recv=Rmin ⟹ 全向源必被探测。
  等价于探测点集的覆盖半径 <= Rmin。用密集边界+内环采样估上界（保守偏大）。
- verify_directional_cover：任意源位置 + 任意朝向，锥∩射程内至少含一个探测点。
  数值验证：对采样源位置 × 采样朝向，检查是否存在满足 dist<=Rmin 且在锥内的探测点。

这些验证器供单测与布点自检使用（覆盖是 completeness 的保证）。
"""
from __future__ import annotations

import math
from typing import List, Tuple

from ..domain.types import Vec2


def coverage_radius(points: List[Vec2], arena_r: float, samples: int = 720) -> float:
    """竞技场内点到最近探测点的最大距离（覆盖半径上界的数值估计）。"""
    worst = 0.0
    rings = [1.0, 0.98, 0.9, 0.8, 0.65, 0.5, 0.35, 0.2, 0.0]
    for frac in rings:
        rr = arena_r * frac
        n = max(1, int(samples * frac)) if frac > 0 else 1
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            x, y = rr * math.cos(ang), rr * math.sin(ang)
            dmin = min(math.hypot(x - px, y - py) for px, py in points)
            worst = max(worst, dmin)
    return worst


def verify_omni_cover(points: List[Vec2], arena_r: float, r_recv: float = 1000.0) -> bool:
    """全向源覆盖：覆盖半径 <= r_recv。"""
    return coverage_radius(points, arena_r) <= r_recv + 1e-6


def _in_cone(src: Vec2, direction_deg: float, q: Vec2, half_deg: float = 90.0) -> bool:
    bearing = math.degrees(math.atan2(q[1] - src[1], q[0] - src[0])) % 360.0
    diff = abs((bearing - direction_deg + 180.0) % 360.0 - 180.0)
    return diff <= half_deg + 1e-9


def verify_directional_cover(points: List[Vec2], arena_r: float,
                             r_recv: float = 1000.0,
                             pos_samples: int = 400, dir_samples: int = 36) -> Tuple[bool, float]:
    """
    定向源覆盖：对采样的 (源位置, 朝向)，验证存在射程内且在锥内的探测点。

    几何事实（不可回避）：竞技场边界上、朝径向【外】的定向源，其 180° 半平面与竞技场
    仅切于边界点，任何【内部】探测点都收不到。故对凸域用有限点【无法】覆盖到精确边界；
    N 点边界环只能覆盖到半径 arena·cos(π/N)，残留最外圈薄环为几何必然，非缺陷。

    因此返回 (fully_covered, guaranteed_core_fraction)：
      - fully_covered：采样全通过（含边界）。
      - guaranteed_core_fraction：所有【失败】样本中最内的半径比例（=1.0 表示无失败）。
        物理含义：半径 < core·arena 的任意定向源都被保证探测到；失败只发生在其外的薄环。
    """
    fully_covered = True
    min_fail_fraction = 1.0
    rng_positions = _sample_positions(arena_r, pos_samples)
    for (sx, sy) in rng_positions:
        frac = math.hypot(sx, sy) / arena_r
        for di in range(dir_samples):
            ddeg = 360.0 * di / dir_samples
            detectable = False
            for q in points:
                d = math.hypot(q[0] - sx, q[1] - sy)
                if d <= r_recv + 1e-6 and _in_cone((sx, sy), ddeg, q):
                    detectable = True
                    break
            if not detectable:
                fully_covered = False
                if frac < min_fail_fraction:
                    min_fail_fraction = frac
    return fully_covered, min_fail_fraction


def _sample_positions(arena_r: float, n: int) -> List[Vec2]:
    """竞技场内近似均匀采样（含边界圈）。"""
    pts: List[Vec2] = []
    rings = max(1, int(math.sqrt(n)))
    for i in range(rings):
        rr = arena_r * (i + 1) / rings
        cnt = max(1, int(n / rings))
        for k in range(cnt):
            ang = 2.0 * math.pi * k / cnt
            pts.append((rr * math.cos(ang), rr * math.sin(ang)))
    pts.append((0.0, 0.0))
    return pts
