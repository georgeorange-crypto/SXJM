"""P4 directionally-robust geometric certificates.

Unlike P3 disc covering, a directional source is certifiably discoverable at
``p`` only when the nearby scan points surround it.  This module keeps that
mathematical layer independent from the runtime certificate manager:

    p in conv({s: |s-p| <= r})

In two dimensions Caratheodory's theorem reduces the local certificate to a
triangle.  The sampler is deliberately a *counterexample finder*, not a proof
of coverage; ``certified`` is true only when every supplied sample passes.
Callers should use adaptive refinement or an externally supplied triangulation
for a formal domain certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, pi
from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _dist(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def point_in_triangle(p: Point, triangle: Sequence[Point], eps: float = 1e-9) -> bool:
    """Boundary-inclusive point-in-triangle test (orientation independent)."""
    if len(triangle) != 3:
        raise ValueError("triangle must contain exactly three points")
    signs = [_cross(triangle[i], triangle[(i + 1) % 3], p) for i in range(3)]
    return max(signs) <= eps or min(signs) >= -eps


def safe_triangle(triangle: Sequence[Point], radius: float = 1000.0,
                  eps: float = 1e-9) -> bool:
    """Whether all three detector separations are <= radius, within tolerance.

    ``eps`` is a numerical boundary tolerance so an analytically exact
    1000 m edge is not rejected because of decimal representation.
    """
    if len(triangle) != 3:
        return False
    return all(_dist(triangle[i], triangle[(i + 1) % 3]) <= radius + eps
               for i in range(3))


def nearby_convex_hull(point: Point, detectors: Sequence[Point], radius: float = 1000.0,
                       eps: float = 1e-9) -> List[Point]:
    """Return the convex hull of detectors within the guaranteed radius."""
    nearby = sorted({(float(x), float(y)) for x, y in detectors
                     if _dist(point, (x, y)) <= radius + eps})
    if len(nearby) <= 1:
        return nearby
    lower: List[Point] = []
    for q in nearby:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], q) <= eps:
            lower.pop()
        lower.append(q)
    upper: List[Point] = []
    for q in reversed(nearby):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], q) <= eps:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def _escape_direction(point: Point, nearby: Sequence[Point]) -> Optional[float]:
    """Return a blind-half-plane normal in degrees, or None if surrounded."""
    if not nearby:
        return 0.0
    angles = sorted((atan2(y - point[1], x - point[0]) + 2 * pi) % (2 * pi)
                    for x, y in nearby)
    gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * pi)
            for i in range(len(angles))]
    i = max(range(len(gaps)), key=gaps.__getitem__)
    if gaps[i] <= pi + 1e-9:
        return None
    # The midpoint of the largest gap is the outward direction.
    return (degrees(angles[i] + gaps[i] / 2.0) + 180.0) % 360.0


@dataclass(frozen=True)
class DirectionalCounterexample:
    point: Point
    nearby_detectors: Tuple[Point, ...]
    escape_direction_deg: float
    reason: str


@dataclass(frozen=True)
class DirectionalCertificateResult:
    certified: bool
    checked_points: int
    counterexample: Optional[DirectionalCounterexample] = None


def check_point(point: Point, detectors: Sequence[Point], radius: float = 1000.0,
                eps: float = 1e-9) -> Optional[DirectionalCounterexample]:
    """Check the local convex-hull condition and return an actionable witness."""
    nearby = tuple(nearby_convex_hull(point, detectors, radius, eps))
    escape = _escape_direction(point, nearby)
    if escape is not None:
        reason = "nearby detectors do not surround point in a closed half-plane"
        return DirectionalCounterexample(point, nearby, escape, reason)
    return None


def verify_samples(detectors: Sequence[Point], samples: Iterable[Point],
                   radius: float = 1000.0, eps: float = 1e-9) -> DirectionalCertificateResult:
    """Verify sampled domain points; first failing point is returned as a witness."""
    count = 0
    for point in samples:
        count += 1
        witness = check_point(point, detectors, radius, eps)
        if witness is not None:
            return DirectionalCertificateResult(False, count, witness)
    return DirectionalCertificateResult(True, count)


def disk_grid_samples(arena_radius: float = 1800.0, spacing: float = 25.0) -> Iterable[Point]:
    """Deterministic square-grid samples clipped to the target disk."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    n = int(arena_radius / spacing)
    for ix in range(-n, n + 1):
        for iy in range(-n, n + 1):
            p = (ix * spacing, iy * spacing)
            if hypot(*p) <= arena_radius + 1e-9:
                yield p
