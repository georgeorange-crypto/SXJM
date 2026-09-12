"""Conservative mission lower bounds with explicit anti-double-counting rules.

These are certified geometric/service bounds for the supplied task snapshot,
not full-information oracle proxies.  The mission bound combines mandatory
service time with the maximum of mutually alternative travel requirements;
it therefore does not claim that coverage, localisation and clear routing are
simultaneously disjoint distances.
"""
from __future__ import annotations

from math import hypot
from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]


def _d(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def open_mst_lower_bound(start: Point, points: Sequence[Point]) -> float:
    """Lower bound for an open route from ``start`` visiting every point."""
    pts = [tuple(map(float, p)) for p in points]
    if not pts:
        return 0.0
    connected = [False] * len(pts)
    best = [float("inf")] * len(pts)
    best[0] = 0.0
    mst = 0.0
    for _ in pts:
        i = min((j for j, used in enumerate(connected) if not used), key=lambda j: best[j])
        connected[i] = True
        mst += best[i]
        for j in range(len(pts)):
            if not connected[j]:
                best[j] = min(best[j], _d(pts[i], pts[j]))
    return mst + min(_d(start, p) for p in pts)


def search_lower_bound(start: Point, coverage_points: Sequence[Point], speed: float) -> float:
    return open_mst_lower_bound(start, coverage_points) / max(float(speed), 1e-9)


def localization_lower_bound(
    start: Point, source_points: Sequence[Point], min_measurements: int, speed: float
) -> float:
    if not source_points:
        return 0.0
    return open_mst_lower_bound(start, source_points) / max(float(speed), 1e-9) + 5.0 * max(0, int(min_measurements))


def route_lower_bound(start: Point, clear_centers: Sequence[Point], speed: float) -> float:
    return open_mst_lower_bound(start, clear_centers) / max(float(speed), 1e-9)


def service_lower_bound(
    n_measure_min: int, n_sources: int, n_switch_min: int,
    measure_s: float = 5.0, clear_s: float = 5.0, switch_s: float = 1.0,
) -> float:
    return (
        max(0, int(n_measure_min)) * float(measure_s)
        + max(0, int(n_sources)) * float(clear_s)
        + max(0, int(n_switch_min)) * float(switch_s)
    )


def mission_lower_bound(
    *, start: Point, coverage_points: Sequence[Point], source_points: Sequence[Point],
    clear_centers: Sequence[Point], speed: float, n_measure_min: int,
    n_sources: int, n_switch_min: int,
) -> float:
    """Certified snapshot bound; travel alternatives use ``max`` to avoid double count."""
    service = service_lower_bound(n_measure_min, n_sources, n_switch_min)
    travel = max(
        search_lower_bound(start, coverage_points, speed),
        localization_lower_bound(start, source_points, 0, speed),
        route_lower_bound(start, clear_centers, speed),
    )
    return service + travel


def certified_gap(elapsed_s: float, lower_bound_s: float) -> float:
    if lower_bound_s <= 0.0:
        return 0.0 if elapsed_s <= 0.0 else float("inf")
    return (float(elapsed_s) - float(lower_bound_s)) / float(lower_bound_s)
