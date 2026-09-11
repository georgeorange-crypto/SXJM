"""
最小包围圆（Minimum Enclosing Circle, MEC）—— Welzl 算法，期望 O(n)。

这是 Correctness Layer 的核心工具：清除证书 R_MEC(Ω) <= 20 依赖它。
必须保证返回的圆【包含所有输入点】（含数值裕量），宁可略大不可略小。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from .vector import Vec2


@dataclass(frozen=True)
class Circle:
    center: Vec2
    radius: float

    def contains(self, p: Vec2, eps: float = 1e-7) -> bool:
        return math.hypot(p[0] - self.center[0], p[1] - self.center[1]) <= self.radius + eps


def _circle_from_2(a: Vec2, b: Vec2) -> Circle:
    cx = (a[0] + b[0]) / 2.0
    cy = (a[1] + b[1]) / 2.0
    r = math.hypot(a[0] - b[0], a[1] - b[1]) / 2.0
    return Circle((cx, cy), r)


def _circle_from_3(a: Vec2, b: Vec2, c: Vec2) -> Optional[Circle]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-18:
        return None  # 共线
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    center = (ux, uy)
    r = math.hypot(ax - ux, ay - uy)
    return Circle(center, r)


def _trivial(boundary: List[Vec2]) -> Circle:
    if not boundary:
        return Circle((0.0, 0.0), 0.0)
    if len(boundary) == 1:
        return Circle(boundary[0], 0.0)
    if len(boundary) == 2:
        return _circle_from_2(boundary[0], boundary[1])
    # 3 点：尝试外接圆，退化则取两两中最大
    c = _circle_from_3(boundary[0], boundary[1], boundary[2])
    if c is not None:
        return c
    best = _circle_from_2(boundary[0], boundary[1])
    for pair in ((0, 2), (1, 2)):
        cc = _circle_from_2(boundary[pair[0]], boundary[pair[1]])
        if cc.radius > best.radius:
            best = cc
    return best


def min_enclosing_circle(points: List[Vec2], seed: int = 12345) -> Circle:
    """
    Welzl 算法（迭代式，避免递归深度问题）。返回包含所有点的最小圆。
    对空集返回半径 0 的原点圆。
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    if not pts:
        return Circle((0.0, 0.0), 0.0)
    rng = random.Random(seed)
    rng.shuffle(pts)

    c = Circle(pts[0], 0.0)
    for i in range(1, len(pts)):
        if c.contains(pts[i]):
            continue
        # pts[i] 在边界上，重解含 pts[i] 的 MEC
        c = Circle(pts[i], 0.0)
        for j in range(i):
            if c.contains(pts[j]):
                continue
            c = _circle_from_2(pts[i], pts[j])
            for k in range(j):
                if c.contains(pts[k]):
                    continue
                c3 = _circle_from_3(pts[i], pts[j], pts[k])
                if c3 is not None:
                    c = c3
    return c


def enclosing_radius(points: List[Vec2]) -> float:
    """便捷函数：只要 MEC 半径。"""
    return min_enclosing_circle(points).radius
