"""Andrew monotone chain 凸包，O(n log n)。返回逆时针顶点，无重复。"""
from __future__ import annotations

from typing import List

from .vector import Vec2, cross, sub


def convex_hull(points: List[Vec2]) -> List[Vec2]:
    """返回逆时针凸包顶点。点数 <=2 时原样返回（去重）。"""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 1:
        return list(pts)

    def half(seq: List[Vec2]) -> List[Vec2]:
        chain: List[Vec2] = []
        for p in seq:
            while len(chain) >= 2 and cross(sub(chain[-1], chain[-2]), sub(p, chain[-2])) <= 0:
                chain.pop()
            chain.append(p)
        return chain

    lower = half(pts)
    upper = half(list(reversed(pts)))
    # 去掉各自最后一点（与另一链起点重合）
    return lower[:-1] + upper[:-1]
