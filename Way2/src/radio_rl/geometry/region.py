"""Convex feasible-region geometry — the primitives the state estimator runs on.

Ported from the reference estimator and wrapped in a small :class:`Region`.
Angle convention matches the problem statement (east = 0 deg, CCW positive,
range [0, 360)). A direction-finding reading ``svd`` with half-error ``delta``
constrains the true source to a wedge (two half-planes) from the detection
point; intersecting several wedges (plus the receive-radius disc and the arena
disc) yields the convex feasible region, whose minimum enclosing circle tells us
whether the source is localized well enough to clear.

This module is torch-free and imports nothing from ``env`` or ``models``
(architecture principles A/B): the estimator sees geometry, not the simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

_EPS = 1e-9


# --------------------------------------------------------------------------
# angles & vectors
# --------------------------------------------------------------------------
def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def norm_deg(a: float) -> float:
    a = math.fmod(a, 360.0)
    if a < 0:
        a += 360.0
    if a >= 360.0:
        a -= 360.0
    return a


def dir_vec(theta_deg: float) -> tuple[float, float]:
    r = deg2rad(theta_deg)
    return (math.cos(r), math.sin(r))


def cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


# --------------------------------------------------------------------------
# half-planes
# --------------------------------------------------------------------------
@dataclass
class Halfplane:
    """Half-plane { (x, y) : nx*x + ny*y <= c }."""

    nx: float
    ny: float
    c: float

    def contains(self, x: float, y: float, eps: float = _EPS) -> bool:
        return self.nx * x + self.ny * y <= self.c + eps


def wedge_halfplanes(sx: float, sy: float, svd_deg: float,
                     delta_deg: float = 1.0) -> list[Halfplane]:
    """The direction-finding wedge from (sx, sy) with reading ``svd_deg`` and
    half-error ``delta_deg``, as the intersection of two half-planes."""
    lo = norm_deg(svd_deg - delta_deg)
    hi = norm_deg(svd_deg + delta_deg)
    ux_lo, uy_lo = dir_vec(lo)
    ux_hi, uy_hi = dir_vec(hi)
    hp_lo = Halfplane(uy_lo, -ux_lo, uy_lo * sx - ux_lo * sy)
    hp_hi = Halfplane(-uy_hi, ux_hi, -uy_hi * sx + ux_hi * sy)
    return [hp_lo, hp_hi]


def _clip_polygon(poly: list[tuple[float, float]], hp: Halfplane,
                  eps: float = _EPS) -> list[tuple[float, float]]:
    """Sutherland-Hodgman: clip a convex polygon by one half-plane."""
    if not poly:
        return poly
    out: list[tuple[float, float]] = []
    n = len(poly)
    for i in range(n):
        cx, cy = poly[i]
        nx_, ny_ = poly[(i + 1) % n]
        c_in = (hp.nx * cx + hp.ny * cy) <= hp.c + eps
        n_in = (hp.nx * nx_ + hp.ny * ny_) <= hp.c + eps
        if c_in:
            out.append((cx, cy))
        if c_in != n_in:
            d_c = hp.c - (hp.nx * cx + hp.ny * cy)
            d_n = hp.c - (hp.nx * nx_ + hp.ny * ny_)
            denom = d_c - d_n
            if abs(denom) > 1e-18:
                t = d_c / denom
                out.append((cx + t * (nx_ - cx), cy + t * (ny_ - cy)))
    return out


def _dedup(poly: list[tuple[float, float]], eps: float = 1e-7) -> list[tuple[float, float]]:
    if not poly:
        return poly
    out = [poly[0]]
    for p in poly[1:]:
        if abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def disc_polygon(cx: float, cy: float, radius: float, m: int = 128
                 ) -> list[tuple[float, float]]:
    """CCW regular m-gon inscribing... approximating the disc of given centre/radius."""
    return [
        (cx + radius * math.cos(2 * math.pi * k / m),
         cy + radius * math.sin(2 * math.pi * k / m))
        for k in range(m)
    ]


# --------------------------------------------------------------------------
# convex hull & measurements on a polygon
# --------------------------------------------------------------------------
def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def build(seq):
        h: list[tuple[float, float]] = []
        for p in seq:
            while len(h) >= 2 and cross(h[-1][0] - h[-2][0], h[-1][1] - h[-2][1],
                                        p[0] - h[-2][0], p[1] - h[-2][1]) <= 0:
                h.pop()
            h.append(p)
        return h

    lower = build(pts)
    upper = build(list(reversed(pts)))
    return lower[:-1] + upper[:-1]


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def polygon_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    if not poly:
        return (0.0, 0.0)
    if len(poly) < 3:
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        return (cx, cy)
    a = 0.0
    cx = 0.0
    cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        crs = x1 * y2 - x2 * y1
        a += crs
        cx += (x1 + x2) * crs
        cy += (y1 + y2) * crs
    if abs(a) < 1e-18:
        mx = sum(p[0] for p in poly) / n
        my = sum(p[1] for p in poly) / n
        return (mx, my)
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a))


def polygon_diameter(poly: list[tuple[float, float]]
                     ) -> tuple[float, tuple[float, float], tuple[float, float]]:
    if not poly:
        return 0.0, (0.0, 0.0), (0.0, 0.0)
    if len(poly) == 1:
        return 0.0, poly[0], poly[0]
    best = -1.0
    ba, bb = poly[0], poly[0]
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(poly[i][0] - poly[j][0], poly[i][1] - poly[j][1])
            if d > best:
                best, ba, bb = d, poly[i], poly[j]
    return best, ba, bb


def min_enclosing_circle(points: list[tuple[float, float]]
                         ) -> tuple[tuple[float, float], float]:
    """Welzl minimum enclosing circle; returns (centre, radius)."""
    pts = list(points)
    if not pts:
        return (0.0, 0.0), 0.0
    random.Random(12345).shuffle(pts)

    def circle_two(p, q):
        return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0), math.hypot(p[0] - q[0], p[1] - q[1]) / 2.0

    def circle_three(p, q, s):
        ax, ay = p
        bx, by = q
        cx_, cy_ = s
        d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        if abs(d) < 1e-18:
            return None
        ux = ((ax * ax + ay * ay) * (by - cy_) + (bx * bx + by * by) * (cy_ - ay)
              + (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx_ - bx) + (bx * bx + by * by) * (ax - cx_)
              + (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d
        return (ux, uy), math.hypot(ax - ux, ay - uy)

    def in_circle(c, r, p, eps=1e-9):
        return math.hypot(p[0] - c[0], p[1] - c[1]) <= r + eps

    c = (0.0, 0.0)
    r = 0.0
    for i, p in enumerate(pts):
        if in_circle(c, r, p):
            continue
        c, r = p, 0.0
        for j in range(i):
            q = pts[j]
            if in_circle(c, r, q):
                continue
            c, r = circle_two(p, q)
            for k in range(j):
                s = pts[k]
                if in_circle(c, r, s):
                    continue
                cc = circle_three(p, q, s)
                if cc is not None:
                    c, r = cc
    return c, r


def diameter_circle_covers(poly, a, b, eps: float = 1e-6):
    """Whether the circle with diameter AB covers the polygon (Thales test)."""
    violators = []
    for (px, py) in poly:
        d = dot(px - a[0], py - a[1], px - b[0], py - b[1])
        scale = max(1.0, (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        if d > eps * scale:
            violators.append((px, py))
    return (len(violators) == 0), violators


# --------------------------------------------------------------------------
# Region: a convex feasible region as a clipped polygon
# --------------------------------------------------------------------------
@dataclass
class Region:
    """A convex feasible region for one source, stored as a CCW polygon.

    Starts as the arena disc and is narrowed by clipping with bearing wedges and
    range discs. All coordinates are metres.
    """

    poly: list[tuple[float, float]] = field(default_factory=list)

    @staticmethod
    def arena(radius: float = 1800.0, m: int = 128) -> "Region":
        return Region(disc_polygon(0.0, 0.0, radius, m))

    @staticmethod
    def box(half: float) -> "Region":
        return Region([(-half, -half), (half, -half), (half, half), (-half, half)])

    def is_empty(self) -> bool:
        return len(self.poly) < 3 or self.area() <= _EPS

    def copy(self) -> "Region":
        return Region(list(self.poly))

    def clip_halfplane(self, hp: Halfplane) -> "Region":
        self.poly = _dedup(_clip_polygon(self.poly, hp))
        return self

    def clip_wedge(self, sx: float, sy: float, svd_deg: float,
                   delta_deg: float = 1.0) -> "Region":
        for hp in wedge_halfplanes(sx, sy, svd_deg, delta_deg):
            self.poly = _clip_polygon(self.poly, hp)
            if not self.poly:
                break
        self.poly = _dedup(self.poly)
        return self

    def clip_inside_disc(self, cx: float, cy: float, radius: float,
                         m: int = 128) -> "Region":
        """Keep only the part inside the given disc (m-gon approximation)."""
        for k in range(m):
            a = 2 * math.pi * k / m
            nx_, ny_ = math.cos(a), math.sin(a)
            # inside: n . P <= n . centre + R
            hp = Halfplane(nx_, ny_, nx_ * cx + ny_ * cy + radius)
            self.poly = _clip_polygon(self.poly, hp)
            if not self.poly:
                break
        self.poly = _dedup(self.poly)
        return self

    def area(self) -> float:
        return polygon_area(self.poly)

    def centroid(self) -> tuple[float, float]:
        return polygon_centroid(self.poly)

    def diameter(self) -> float:
        return polygon_diameter(self.poly)[0]

    def min_enclosing_circle(self) -> tuple[tuple[float, float], float]:
        return min_enclosing_circle(self.poly)
