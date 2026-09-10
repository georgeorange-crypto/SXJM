"""Information-gain heuristics over the belief geometry.

Pure functions used to rank *where to scan next* so a detected-but-not-localized
source gets triangulated fast. They are heuristics (never affect soundness): the
candidate generator proposes scan poses and these score them. A new bearing from
point P cuts the feasible region perpendicular to the line P->source, so it
shrinks the region's long axis most when that line is perpendicular to the axis
and P is close (the wedge's cross-range half-width grows with distance).
"""

from __future__ import annotations

import math

from .region import Region, polygon_centroid, polygon_diameter


def _ang_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def bearing_spread_deg(bearings: list[tuple[float, float, float]]) -> float:
    """Largest pairwise angular gap (deg, in [0, 180]) among measured bearings.

    Near 0 means all readings are collinear (poor triangulation); near 90 means
    a strong crossing baseline."""
    svds = [b[2] for b in bearings]
    if len(svds) < 2:
        return 0.0
    best = 0.0
    for i in range(len(svds)):
        for j in range(i + 1, len(svds)):
            best = max(best, min(_ang_diff_deg(svds[i], svds[j]),
                                 180.0 - _ang_diff_deg(svds[i], svds[j])))
    return best


def _diameter_axis(region: Region) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return (centroid, unit-axis, diameter) of the region's long axis."""
    if region.is_empty():
        return (0.0, 0.0), (1.0, 0.0), 0.0
    diam, a, b = polygon_diameter(region.poly)
    cx, cy = polygon_centroid(region.poly)
    ux, uy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(ux, uy)
    if n < 1e-9:
        return (cx, cy), (1.0, 0.0), 0.0
    return (cx, cy), (ux / n, uy / n), diam


def triangulation_quality(region: Region, px: float, py: float) -> float:
    """In [0, 1]: how well a bearing taken from (px, py) crosses the region's
    long axis. 1 means the line to the region is perpendicular to that axis."""
    (cx, cy), (ux, uy), diam = _diameter_axis(region)
    if diam <= 1e-6:
        return 0.0
    bx, by = cx - px, cy - py
    n = math.hypot(bx, by)
    if n < 1e-6:
        return 0.0
    bx, by = bx / n, by / n
    # |sin(angle between bearing and axis)| == |cross product| of unit vectors.
    return abs(bx * uy - by * ux)


def expected_region_shrink(region: Region, px: float, py: float,
                           delta_deg: float = 1.0) -> float:
    """Rough metres of long-axis reduction from a new bearing at (px, py).

    The wedge's cross-range half-width at the region is ~ D * tan(delta); only
    the component aligned with the long axis (the triangulation quality) shrinks
    the diameter. Returns >= 0, for ranking only."""
    (cx, cy), _, diam = _diameter_axis(region)
    if diam <= 1e-6:
        return 0.0
    d = math.hypot(cx - px, cy - py)
    residual = 2.0 * d * math.tan(math.radians(delta_deg))
    q = triangulation_quality(region, px, py)
    return max(0.0, q * (diam - residual))
