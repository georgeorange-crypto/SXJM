"""Geometry primitives: points, angles, disc/arena/wedge membership.

Angle convention (DESIGN.md §1 note, 附录2(1)): ``svd_deg`` is the angle of the
"observation point -> source" vector measured CCW from the +x (east) axis, in
[0, 360). ``bearing_deg`` follows exactly that convention.
"""

from __future__ import annotations

from math import atan2, cos, degrees, hypot, radians, sin, sqrt
from typing import List, Tuple

Point = Tuple[float, float]

EPS = 1e-9

ARENA_RADIUS = 1800.0


# --- vector ops --------------------------------------------------------------


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def scale(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    """z-component of a x b."""
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Point) -> float:
    return hypot(a[0], a[1])


def dist(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def dist2(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


# --- angles ------------------------------------------------------------------


def deg2rad(d: float) -> float:
    return radians(d)


def rad2deg(r: float) -> float:
    return degrees(r)


def norm_deg(d: float) -> float:
    """Normalise an angle in degrees to [0, 360)."""
    r = d % 360.0
    return r + 360.0 if r < 0.0 else r


def bearing_deg(origin: Point, target: Point) -> float:
    """Angle (deg, CCW from +x, in [0,360)) of the ``origin -> target`` vector.

    Undefined when ``origin == target``; returns 0.0 there by convention.
    """
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return norm_deg(degrees(atan2(dy, dx)))


def angle_sep_deg(a: float, b: float) -> float:
    """Smallest absolute separation between two angles (deg), in [0, 180]."""
    d = abs(norm_deg(a) - norm_deg(b)) % 360.0
    return d if d <= 180.0 else 360.0 - d


# --- membership --------------------------------------------------------------


def in_disc(p: Point, center: Point, radius: float, eps: float = 0.0) -> bool:
    """True iff ``p`` lies in the closed disc ``B(center, radius)`` (± eps)."""
    return dist(p, center) <= radius + eps


def in_arena(p: Point, radius: float = ARENA_RADIUS, eps: float = 0.0) -> bool:
    """True iff ``p`` lies in the arena disc ``D(0, radius)`` (± eps)."""
    return hypot(p[0], p[1]) <= radius + eps


def point_in_wedge(
    apex: Point,
    center_deg: float,
    half_deg: float,
    q: Point,
    eps_deg: float = 0.0,
) -> bool:
    """True iff the bearing ``apex -> q`` is within ``half_deg (+ eps_deg)`` of
    ``center_deg`` (boundary inclusive — DESIGN.md §1 "含边界").

    The apex itself is treated as inside (bearing undefined there).
    """
    if q[0] == apex[0] and q[1] == apex[1]:
        return True
    return angle_sep_deg(bearing_deg(apex, q), center_deg) <= half_deg + eps_deg


# --- circle-circle intersection ---------------------------------------------


def circle_circle_intersections(
    c1: Point, r1: float, c2: Point, r2: float, eps: float = EPS
) -> List[Point]:
    """Intersection points of two circles.

    Returns [] (disjoint, one-inside-other, or coincident), [p] (tangent), or
    [p, q] (two crossings).
    """
    d = dist(c1, c2)
    if d <= eps:
        return []  # concentric (coincident circles => infinite; report none)
    if d > r1 + r2 + eps:
        return []  # too far apart
    if d < abs(r1 - r2) - eps:
        return []  # one strictly inside the other
    # a = distance from c1 to the radical line along the center line
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    ux = (c2[0] - c1[0]) / d
    uy = (c2[1] - c1[1]) / d
    mx = c1[0] + a * ux
    my = c1[1] + a * uy
    if h2 <= eps:  # tangent (single point)
        return [(mx, my)]
    h = sqrt(h2)
    # perpendicular direction (-uy, ux)
    return [
        (mx - h * uy, my + h * ux),
        (mx + h * uy, my - h * ux),
    ]
