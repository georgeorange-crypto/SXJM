"""Convex-region geometry: halfplanes, wedge clipping, hull, diameter.

Way4 represents a channel's positive feasible set ``F_c`` as a convex polygon
that is a SUPERSET of the true feasible set (DESIGN.md §4/§5): the circumscribed
arena polygon clipped by the ``±(1°+ε)`` bearing wedges. Superset is the sound
choice for the clear guarantee — an outer bound only makes ``MEC(F_c).r`` larger,
never smaller, so ``MEC.r ≤ threshold`` still implies the true source is within
the clear radius (DESIGN.md §3.3, §6.5 CLEAR certificate).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import List, Sequence, Tuple

from .primitives import EPS, Point, deg2rad, dist, norm_deg


@dataclass(frozen=True)
class Halfplane:
    """The closed halfplane ``a*x + b*y <= c``."""

    a: float
    b: float
    c: float

    def value(self, p: Point) -> float:
        return self.a * p[0] + self.b * p[1] - self.c

    def contains(self, p: Point, eps: float = EPS) -> bool:
        return self.value(p) <= eps


def wedge_halfplanes(apex: Point, center_deg: float, half_deg: float) -> Tuple[Halfplane, Halfplane]:
    """The two halfplanes whose intersection is the wedge of half-width
    ``half_deg`` about bearing ``center_deg`` with vertex ``apex``.

    Valid for ``0 < half_deg < 90`` (a set-membership bearing wedge is ~1°, far
    inside that range). Each halfplane passes through ``apex``.
    """
    if not (0.0 < half_deg < 90.0):
        raise ValueError(f"wedge_halfplanes needs 0 < half_deg < 90, got {half_deg}")
    px, py = apex
    hi = deg2rad(norm_deg(center_deg + half_deg))  # upper boundary ray angle
    lo = deg2rad(norm_deg(center_deg - half_deg))  # lower boundary ray angle
    # Upper ray: interior is on the -sin(hi)*x + cos(hi)*y <= (that at apex) side.
    a1, b1 = -sin(hi), cos(hi)
    # Lower ray: interior is on the sin(lo)*x - cos(lo)*y <= (that at apex) side.
    a2, b2 = sin(lo), -cos(lo)
    return (
        Halfplane(a1, b1, a1 * px + b1 * py),
        Halfplane(a2, b2, a2 * px + b2 * py),
    )


def arena_polygon(radius: float = 1800.0, n_sides: int = 96) -> List[Point]:
    """A regular polygon CIRCUMSCRIBED about ``D(0, radius)`` (edges tangent to
    the circle) — hence a superset of the arena disc, suitable as the initial
    ``F_c`` outer bound. ``n_sides=96`` overshoots the true disc by ~0.05%.
    """
    r_out = radius / cos(pi / n_sides)  # vertex radius so inradius == radius
    verts: List[Point] = []
    for k in range(n_sides):
        ang = 2.0 * pi * (k + 0.5) / n_sides  # offset so edges (not verts) face axes
        verts.append((r_out * cos(ang), r_out * sin(ang)))
    return verts


def disc_superset_halfplanes(
    center: Point, radius: float, n_sides: int = 48
) -> List[Halfplane]:
    """Tangent halfplanes whose intersection is a regular ``n_sides``-gon
    CIRCUMSCRIBED about ``B(center, radius)`` — a superset of the disc.

    Clipping ``F_c`` by these keeps it a convex superset of the true feasible set
    while soundly folding in a range constraint ``B(S, R)`` (DESIGN.md §5): the
    true source lies in the disc, hence inside any circumscribing polygon, so it
    is never clipped away.
    """
    cx, cy = center
    hps: List[Halfplane] = []
    for k in range(n_sides):
        ang = 2.0 * pi * k / n_sides
        a, b = cos(ang), sin(ang)  # outward normal of the tangent edge
        hps.append(Halfplane(a, b, a * cx + b * cy + radius))
    return hps


def clip_polygon(poly: Sequence[Point], hp: Halfplane, eps: float = EPS) -> List[Point]:
    """Sutherland-Hodgman clip of a convex polygon by a halfplane (keep the
    ``a*x+b*y <= c`` side). Returns [] if the polygon is fully clipped away.
    """
    if not poly:
        return []
    out: List[Point] = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        cur_val = hp.value(cur)
        nxt_val = hp.value(nxt)
        cur_in = cur_val <= eps
        nxt_in = nxt_val <= eps
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            denom = cur_val - nxt_val
            if abs(denom) > 0.0:
                t = cur_val / denom
                out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
    return out


def halfplane_intersection(
    halfplanes: Sequence[Halfplane],
    initial: Sequence[Point],
) -> List[Point]:
    """Intersect ``initial`` (a convex polygon, e.g. ``arena_polygon()``) with all
    halfplanes by successive clipping. Returns the resulting convex polygon
    (possibly empty)."""
    poly = list(initial)
    for hp in halfplanes:
        poly = clip_polygon(poly, hp)
        if not poly:
            return []
    return poly


# --- hull / metrics ----------------------------------------------------------


def convex_hull(points: Sequence[Point]) -> List[Point]:
    """Andrew's monotone-chain hull, CCW, without repeating the first point.

    Degenerate inputs (<=2 unique points, or all collinear) return the sorted
    unique points.
    """
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def half(seq: Sequence[Point]) -> List[Point]:
        chain: List[Point] = []
        for p in seq:
            while len(chain) >= 2:
                o, a = chain[-2], chain[-1]
                if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) <= 0:
                    chain.pop()
                else:
                    break
            chain.append(p)
        return chain

    lower = half(pts)
    upper = half(list(reversed(pts)))
    return lower[:-1] + upper[:-1]


def polygon_area(poly: Sequence[Point]) -> float:
    """Absolute area via the shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def polygon_centroid(poly: Sequence[Point]) -> Point:
    """Area centroid; falls back to the vertex mean for degenerate polygons."""
    n = len(poly)
    if n == 0:
        return (0.0, 0.0)
    if n < 3:
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    a2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cr = x1 * y2 - x2 * y1
        a2 += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    if abs(a2) < 1e-12:
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    return (cx / (3.0 * a2), cy / (3.0 * a2))


def polygon_diameter(poly: Sequence[Point]) -> float:
    """Maximum pairwise vertex distance (brute force; polygons here are small)."""
    n = len(poly)
    if n < 2:
        return 0.0
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(poly[i], poly[j])
            if d > best:
                best = d
    return best
