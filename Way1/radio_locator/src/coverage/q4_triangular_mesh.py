"""
Q4 覆盖布点：三角网格，保证任何【定向源】（覆盖角 180°）至少被一个顶点探测到。

难点：定向源只在半平面（180° 锥）内发射。一个探测点若落在源的背面锥外，即使很近
也收不到。故不能只保证“落在某点 r_safe 内”，还要保证“落在某个【在锥内】的点的射程内”。

关键几何事实：源的 180° 锥是一个半平面。对任意方向的半平面，只要探测点足够密，
半平面必含至少一个射程内的顶点。充分条件（保守）：三角网格边长 h 满足
  h <= Rmin = 1000，并且网格覆盖竞技场。
证明思路：以源为心、半径 Rmin 的半圆盘（锥∩射程）面积 = π Rmin²/2。正三角网格中
任一半径 Rmin 的半圆盘，只要 h<=Rmin，必包含至少一个格点（半圆盘直径 2Rmin>=2h，
且半圆盘“高度” Rmin>=h 覆盖至少一行网格，行内点距 h<=Rmin 必有一点落入半平面）。
取 h=950<1000 留裕量。

输出顶点集与 Q3 相同为探测点序列，交给规划器排程。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from ..domain.types import Vec2
from .q3_hex_cover import CoverPlan, _farthest_uncovered_distance


def triangular_mesh(arena_r: float, edge: float = 950.0,
                    boundary_points: int = 24) -> List[Vec2]:
    """
    正三角网格顶点（行距 = edge*√3/2，奇偶行错开 edge/2），保留竞技场内及边缘顶点。
    edge <= Rmin=1000 保证【内部】定向源可检测；默认 950 留裕量。

    额外加一圈 boundary_points 个边界点：竞技场边界上朝径向外的定向源，只有边界附近的
    点能探测到。N 点边界环把“保证覆盖”推到半径 arena·cos(π/N)（N=24 → 0.991R）。
    精确边界（100%R）用有限内部点几何上不可覆盖——这是凸域的固有极限，非布点缺陷。
    """
    s = edge
    dy = s * math.sqrt(3.0) / 2.0
    pts: List[Vec2] = []
    j = 0
    y = -arena_r
    # 向外扩一圈以覆盖边界
    y = -arena_r - dy
    while y <= arena_r + dy:
        offset = 0.0 if (j % 2 == 0) else s / 2.0
        x = -arena_r - s
        while x <= arena_r + s:
            px = x + offset
            if px * px + y * y <= arena_r * arena_r + 1e-6:
                pts.append((px, y))
            x += s
        y += dy
        j += 1
    # 保证圆心与边界若干点入选（可检测性对边界源尤为关键）
    if (0.0, 0.0) not in pts:
        pts.append((0.0, 0.0))
    # 边界环：把定向源保证覆盖半径推到 arena·cos(π/N)。
    if boundary_points > 0:
        for k in range(boundary_points):
            ang = 2.0 * math.pi * k / boundary_points
            pts.append((arena_r * math.cos(ang), arena_r * math.sin(ang)))
    return pts


@dataclass(frozen=True)
class MeshPlan:
    points: List[Vec2]
    edge: float
    omni_covers: bool          # 是否也满足全向覆盖（更强）


def q4_mesh_cover(arena_r: float, edge: float = 950.0) -> MeshPlan:
    pts = triangular_mesh(arena_r, edge)
    # 全向覆盖半径（用于兼判全向源）：网格覆盖半径 = edge/√3
    worst = _farthest_uncovered_distance(pts, arena_r)
    omni_ok = worst <= 1000.0 + 1e-6
    return MeshPlan(points=pts, edge=edge, omni_covers=omni_ok)
