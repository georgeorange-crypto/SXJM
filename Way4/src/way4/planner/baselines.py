"""Deterministic planning baselines used for fair Way4 comparisons."""

from dataclasses import dataclass
from typing import List, Tuple

from sxjm_core.geometry import dist

Point = Tuple[float, float]


@dataclass(frozen=True)
class CoverageGreedyResult:
    waypoints: List[Point]
    covered_gain: float
    travel_distance: float


def coverage_greedy(certificate, channels, start=(0.0, 0.0), max_points=None):
    """Choose backbone points by marginal coverage gain per travel cost.

    This baseline has no learned scores, NBV, lifecycle preference or future-cost
    lookahead.  It is therefore a reproducible exploration-only comparator.  The
    certificate remains authoritative; this function only reads planner coverage
    gains and never certifies absence.
    """
    pending = [int(c) for c in channels]
    anchors = list(certificate.anchors)
    chosen: List[Point] = []
    pos = (float(start[0]), float(start[1]))
    total_gain = 0.0
    total_dist = 0.0
    limit = len(anchors) if max_points is None else int(max_points)
    for _ in range(limit):
        best = None
        for a in anchors:
            if a in chosen:
                continue
            gain = sum(float(certificate.coverage_gain(c, a)) for c in pending)
            cost = max(1.0, dist(pos, a))
            key = (gain / cost, gain, -cost, tuple(a))
            if best is None or key > best[0]:
                best = (key, (float(a[0]), float(a[1])), gain)
        if best is None or best[2] <= 0.0:
            break
        _key, point, gain = best
        chosen.append(point)
        total_gain += gain
        total_dist += dist(pos, point)
        pos = point
    return CoverageGreedyResult(chosen, total_gain, total_dist)
