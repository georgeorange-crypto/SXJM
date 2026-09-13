"""Offline P3 circular-cover backbone optimisation (Q01--Q02)."""
from dataclasses import dataclass
from math import cos, pi, sin
from typing import Sequence


@dataclass(frozen=True)
class P3BackboneResult:
    points: tuple[tuple[float, float], ...]
    n_points: int
    route_length_m: float
    overlap_area_proxy: float
    outside_area_proxy: float
    objective: float


def center_ring_baseline(*, arena_radius: float = 1800.0,
                         detection_radius: float = 1000.0,
                         n_ring: int = 8, ring_radius: float = 1150.0):
    """V6-style center + ring baseline."""
    return [(0.0, 0.0)] + [(float(ring_radius * cos(2 * pi * i / n_ring)),
                            float(ring_radius * sin(2 * pi * i / n_ring)))
                           for i in range(n_ring)]


def optimize_p3_backbone(candidate_points: Sequence[tuple[float, float]], *,
                         arena_radius: float = 1800.0,
                         detection_radius: float = 1000.0,
                         grid_step: float = 180.0,
                         weights=(1.0, 0.001, 0.001, 0.01)) -> P3BackboneResult:
    """Greedy near-optimal cover over a deterministic arena grid."""
    if arena_radius <= 0 or detection_radius <= 0 or grid_step <= 0:
        raise ValueError("radii and grid_step must be positive")
    w1, w2, w3, w4 = (float(x) for x in weights)
    grid = []
    n = int(arena_radius / grid_step) + 1
    for ix in range(-n, n + 1):
        for iy in range(-n, n + 1):
            p = (ix * grid_step, iy * grid_step)
            if p[0] * p[0] + p[1] * p[1] <= arena_radius * arena_radius:
                grid.append(p)
    pts = [(float(x), float(y)) for x, y in candidate_points]
    uncovered = set(range(len(grid)))
    selected = []
    while uncovered:
        choices = []
        for p in pts:
            cover = {i for i in uncovered if (grid[i][0] - p[0]) ** 2 +
                     (grid[i][1] - p[1]) ** 2 <= detection_radius ** 2}
            if cover:
                choices.append((len(cover), p, cover))
        if not choices:
            raise ValueError("candidate_points cannot cover the supplied arena grid")
        _, point, cover = max(choices, key=lambda x: (x[0], -x[1][0] ** 2 - x[1][1] ** 2, x[1]))
        selected.append(point)
        uncovered.difference_update(cover)
    route_length = sum(((selected[i][0] - selected[i - 1][0]) ** 2 +
                        (selected[i][1] - selected[i - 1][1]) ** 2) ** 0.5
                       for i in range(1, len(selected)))
    overlap = sum(max(0, sum((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 <= detection_radius ** 2
                              for q in grid) - 1) for p in selected)
    outside = sum(max(0.0, detection_radius ** 2 -
                      max(0.0, arena_radius - (p[0] ** 2 + p[1] ** 2) ** 0.5) ** 2)
                  for p in selected)
    objective = w1 * len(selected) + w2 * route_length + w3 * overlap + w4 * outside
    return P3BackboneResult(tuple(selected), len(selected), route_length,
                            float(overlap), float(outside), float(objective))
