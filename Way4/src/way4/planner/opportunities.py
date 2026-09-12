"""Route-aware opportunity primitives (P0/P1).

These helpers are deliberately pure: they only rank already geometry-validated
candidates and never alter belief or certificate state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class RemainingTask:
    task_type: str
    spatial_region: Tuple[Point, ...] = ()
    channel: int | None = None
    service_time: float = 0.0
    deadline: float | None = None
    information_value: float = 0.0
    mandatory: bool = True


def pareto_prune(items: Sequence, *, cost=lambda x: x.route_marginal,
                 gain=lambda x: x.refinement_gain) -> List:
    """Keep candidates not strictly dominated by route cost and information gain."""
    kept = []
    for i, item in enumerate(items):
        ci, gi = float(cost(item)), float(gain(item))
        dominated = any(
            float(cost(other)) <= ci and float(gain(other)) >= gi
            and (float(cost(other)) < ci or float(gain(other)) > gi)
            for j, other in enumerate(items) if i != j
        )
        if not dominated:
            kept.append(item)
    return kept


def insertion_cost(route: Sequence[Point], point: Point) -> float:
    """Minimum Euclidean insertion distance delta, including tail insertion."""
    import math
    if not route:
        return 0.0
    if len(route) == 1:
        return math.dist(route[0], point)
    best = math.inf
    for a, b in zip(route, route[1:]):
        best = min(best, math.dist(a, point) + math.dist(point, b) - math.dist(a, b))
    best = min(best, math.dist(route[-1], point))
    return max(0.0, best)
