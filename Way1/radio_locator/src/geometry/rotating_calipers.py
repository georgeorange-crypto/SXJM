"""
旋转卡壳求凸多边形直径（最远点对），O(n)。

输入需为凸多边形逆时针顶点。若输入未必凸，先取 convex_hull。
"""
from __future__ import annotations

from typing import List, Tuple

from .vector import Vec2, dist2, sub, cross
from .convex_hull import convex_hull


def diameter(points: List[Vec2]) -> Tuple[float, Vec2, Vec2]:
    """
    返回 (直径, 点A, 点B)。对任意点集自动求凸包后旋转卡壳。
    退化：0 点→(0,原点,原点)；1 点→(0,该点,该点)；2 点→两点距离。
    """
    hull = convex_hull(points)
    m = len(hull)
    if m == 0:
        return 0.0, (0.0, 0.0), (0.0, 0.0)
    if m == 1:
        return 0.0, hull[0], hull[0]
    if m == 2:
        import math
        return math.hypot(hull[0][0] - hull[1][0], hull[0][1] - hull[1][1]), hull[0], hull[1]

    # 旋转卡壳
    best = 0.0
    pa, pb = hull[0], hull[1]
    j = 1
    for i in range(m):
        ni = (i + 1) % m
        edge = sub(hull[ni], hull[i])
        # 推进对踵点 j，使其到边 i 的距离最大（用叉积单调性）
        while True:
            nj = (j + 1) % m
            if cross(edge, sub(hull[nj], hull[i])) > cross(edge, sub(hull[j], hull[i])):
                j = nj
            else:
                break
        # 候选点对：(i,j) 与 (ni,j)
        for a in (i, ni):
            d2 = dist2(hull[a], hull[j])
            if d2 > best:
                best = d2
                pa, pb = hull[a], hull[j]

    return best ** 0.5, pa, pb
