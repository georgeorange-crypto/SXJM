"""Minimum enclosing circle (Welzl), the clear-guard primitive.

DESIGN.md §3.3 / §6.5: a channel is *clearable* iff ``MEC(F_c).r <= threshold``
(Way2 uses 18, leaving 2 m of the 20 m clear radius as margin). Because ``F_c``
is stored as a convex superset polygon, ``min_enclosing_circle`` of its vertices
is an outer bound of the true feasible set's MEC — sound for the clear guarantee.
"""

from __future__ import annotations

import random
from math import sqrt
from typing import List, Optional, Sequence, Tuple

from .primitives import Point, dist


def _in_circle(p: Point, center: Point, radius: float, eps: float) -> bool:
    return dist(p, center) <= radius + eps


def circle_from_two(a: Point, b: Point) -> Tuple[Point, float]:
    """Smallest circle through two points: the diameter circle."""
    center = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    return center, 0.5 * dist(a, b)


def circumcircle(a: Point, b: Point, c: Point) -> Optional[Tuple[Point, float]]:
    """Circle through three points; ``None`` if (nearly) collinear."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    center = (ux, uy)
    return center, dist(center, a)


def min_enclosing_circle(
    points: Sequence[Point], eps: float = 1e-7, seed: int = 0
) -> Tuple[Point, float]:
    """Welzl's minimum enclosing circle (move-to-front, expected O(n)).

    Returns ``(center, radius)``. Empty input -> ``((0,0), 0)``. Deterministic:
    the shuffle uses a fixed ``seed``.
    """
    pts: List[Point] = [(float(x), float(y)) for x, y in points]
    if not pts:
        return ((0.0, 0.0), 0.0)
    if len(pts) == 1:
        return (pts[0], 0.0)

    random.Random(seed).shuffle(pts)
    center = pts[0]
    radius = 0.0
    for i in range(1, len(pts)):
        if _in_circle(pts[i], center, radius, eps):
            continue
        # pts[i] is on the boundary of the MEC of pts[0..i]
        center, radius = pts[i], 0.0
        for j in range(i):
            if _in_circle(pts[j], center, radius, eps):
                continue
            center, radius = circle_from_two(pts[i], pts[j])
            for k in range(j):
                if _in_circle(pts[k], center, radius, eps):
                    continue
                cc = circumcircle(pts[i], pts[j], pts[k])
                if cc is not None:
                    center, radius = cc
    return (center, radius)
