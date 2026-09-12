"""RouteEstimator — clear-route length for clearable points and uncertain
neighbourhoods (DESIGN.md §9, §10 J_route, M6).

Two target kinds:
  * **Clearable** (``MEC.r ≤ thr``, treated as fixed points): exact **Held–Karp**
    open TSP, memoised and re-solved ONLY when the set changes (§9). The live
    robot start is decoupled from the expensive combinatorial part: the cached,
    start-free minimum path over the SET plus the cheap nearest-entry leg gives
    ``estimate_clear_travel`` for the future-cost hot loop; ``optimal_clear_order``
    gives the exact from-start order for actual execution.
  * **Uncertain** (neighbourhood ``F_c``): **TSPN** with approach distance
    ``d_N(x, F_c) = max(0, ‖x−m_c‖ − r_c)`` and an optional expected localisation
    cost ``L_c`` (offline-fit, cost-only — never touches the hard belief, §9).

Beyond the exact budget it falls back to nearest-neighbour + 2-opt (Way3 style).
All lengths are in metres; callers divide by speed for time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, inf
from typing import Dict, List, Optional, Sequence, Tuple

from .tsp import (
    brute_force_open,
    held_karp_min_path,
    held_karp_open,
    nearest_neighbor_open,
    two_opt_open,
)
from ..toolbox import generalized_tsp

Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class RoutePlan:
    order: List[Point]
    length: float
    exact: bool
    localization_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.length + self.localization_cost


@dataclass
class Neighborhood:
    """A TSPN target: approach anywhere within ``radius`` of ``center`` (§9)."""

    center: Point
    radius: float = 0.0
    localization_cost: float = 0.0


def guaranteed_clear_neighborhood(
    feasible: Neighborhood, clear_radius: float
) -> Optional[Neighborhood]:
    """Return the guaranteed-clear set ``K`` for a disk feasible set.

    For ``F = B(center, r_F)``, the worst-case source distance from a robot
    position ``y`` is ``||y-center|| + r_F``.  Therefore
    ``K = B(center, max(0, R_kill-r_F))``.  A negative residual means that
    the feasible set is wider than the clear radius and has no guaranteed
    point; returning ``None`` keeps that safety condition explicit.

    This is the exact convex-disk special case of the general intersection
    ``K = intersection_x B(x, R_kill)``.  Polygon/half-plane belief sets must
    provide their boundary vertices to a future generalisation.
    """
    residual = float(clear_radius) - float(feasible.radius)
    if residual < -1e-12:
        return None
    return Neighborhood(
        center=tuple(feasible.center),
        radius=max(0.0, residual),
        localization_cost=float(feasible.localization_cost),
    )


def guaranteed_clear_point(vertices: Sequence[Point], clear_radius: float):
    """Return ``(MEC center, MEC radius)`` when finite feasible vertices admit K.

    For a polygon/multipolygon outer boundary, every feasible point is covered
    whenever the maximum distance to the returned MEC center is <= ``clear_radius``.
    This is a conservative finite-vertex certificate: an empty/degenerate input or
    a radius failure returns ``None`` rather than inventing a waypoint.
    """
    pts = [(float(x), float(y)) for x, y in vertices]
    if not pts:
        return None
    from sxjm_core.geometry import min_enclosing_circle
    center, radius = min_enclosing_circle(pts)
    if radius > float(clear_radius) + 1e-9:
        return None
    return (center, float(radius))


@dataclass(frozen=True)
class LocalizationCostModel:
    """Offline-fit expected localisation cost ``L̂ = a0 + a1·r_MEC + a2·diam +
    a3/max(sinγ, ε)`` (§9). COST ESTIMATE ONLY — never mutates belief (禁止: it is
    not a measurement). Defaults are benign (all-zero) until fit from offline_sim."""

    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    eps: float = 1e-3

    def estimate(self, r_mec: float, diameter: float, sin_gamma: float) -> float:
        denom = max(abs(sin_gamma), self.eps)
        return self.a0 + self.a1 * r_mec + self.a2 * diameter + self.a3 / denom


class RouteEstimator:
    """Caches the exact tour over the clearable set; re-solves only on set change."""

    def __init__(self, speed: float = 5.0, exact_max_n: int = 13,
                 strategy: str = "tspn") -> None:
        self.speed = float(speed)
        self.exact_max_n = int(exact_max_n)
        if strategy not in ("tspn", "gtsp", "nearest"):
            raise ValueError("strategy must be 'tspn', 'gtsp' or 'nearest'")
        self.strategy = strategy
        self._set_tour_cache: Dict[frozenset, Tuple[List[int], float]] = {}
        self.solves = 0   # cache-miss solve count (for the memoisation test)

    # -- cost-matrix builders ----------------------------------------------

    @staticmethod
    def _point_matrices(start: Point, pts: Sequence[Point]):
        start_cost = [_dist(start, p) for p in pts]
        cost = [[_dist(a, b) for b in pts] for a in pts]
        return start_cost, cost

    @staticmethod
    def _neighborhood_matrices(start: Point, neigh: Sequence[Neighborhood]):
        # asymmetric: arriving at j only needs to get within r_j of its centre
        start_cost = [max(0.0, _dist(start, nb.center) - nb.radius) for nb in neigh]
        cost = [
            [max(0.0, _dist(a.center, b.center) - b.radius) for b in neigh]
            for a in neigh
        ]
        return start_cost, cost

    def _solve(self, start_cost, cost) -> Tuple[List[int], float, bool]:
        n = len(start_cost)
        if self.strategy == "nearest":
            order, length = nearest_neighbor_open(start_cost, cost)
            return order, length, False
        if n <= self.exact_max_n:
            order, length = held_karp_open(start_cost, cost)
            return order, length, True
        seed, _ = nearest_neighbor_open(start_cost, cost)
        order, length = two_opt_open(seed, start_cost, cost)
        return order, length, False

    # -- clearable fixed points --------------------------------------------

    def optimal_clear_order(self, start: Point, targets: Sequence[Point]) -> RoutePlan:
        """Exact (or 2-opt fallback) open route from ``start`` through all clear
        points, in metres. Used at execution time — NOT in the future-cost loop."""
        pts = [(float(x), float(y)) for x, y in targets]
        if not pts:
            return RoutePlan([], 0.0, True)
        start_cost, cost = self._point_matrices(start, pts)
        idx, length, exact = self._solve(start_cost, cost)
        return RoutePlan([pts[i] for i in idx], length, exact)

    def optimal_neighborhood_order(self, start: Point,
                                   neighborhoods: Sequence[Neighborhood]) -> RoutePlan:
        """Open online-TSPN route over conservative feasible neighborhoods."""
        neigh = list(neighborhoods)
        if not neigh:
            return RoutePlan([], 0.0, True)
        if self.strategy == "gtsp":
            route, length = self._gtsp_route(start, neigh)
            return RoutePlan(route, length, True,
                             sum(n.localization_cost for n in neigh))
        start_cost, cost = self._neighborhood_matrices(start, neigh)
        idx, length, exact = self._solve(start_cost, cost)
        return RoutePlan([neigh[i].center for i in idx], length, exact,
                         sum(n.localization_cost for n in neigh))

    def _set_tour(self, targets: Sequence[Point]) -> Tuple[List[int], float]:
        """Cached start-free minimum path over the target SET (order/length keyed by
        the set of rounded coords). Re-solved only when the set changes (§9)."""
        pts = [(float(x), float(y)) for x, y in targets]
        key = frozenset((round(x, 3), round(y, 3)) for x, y in pts)
        cached = self._set_tour_cache.get(key)
        if cached is not None:
            return cached
        self.solves += 1
        if len(pts) <= 1:
            result: Tuple[List[int], float] = (list(range(len(pts))), 0.0)
        else:
            cost = [[_dist(a, b) for b in pts] for a in pts]
            if len(pts) <= self.exact_max_n:
                result = held_karp_min_path(cost)
            else:
                sc = [0.0] * len(pts)
                seed, _ = nearest_neighbor_open(sc, cost)
                result = two_opt_open(seed, sc, cost)
        self._set_tour_cache[key] = result
        return result

    def set_tour_length(self, targets: Sequence[Point]) -> float:
        """Intra-set travel length (metres), cached and start-independent."""
        return self._set_tour(targets)[1]

    def estimate_clear_travel(self, start: Point, targets: Sequence[Point]) -> float:
        """Cheap J_route proxy: nearest-entry leg + cached set tour (§10). Avoids
        re-solving Held–Karp for the moving start on every candidate."""
        pts = [(float(x), float(y)) for x, y in targets]
        if not pts:
            return 0.0
        nearest = min(_dist(start, p) for p in pts)
        return nearest + self.set_tour_length(pts)

    @staticmethod
    def insertion_delta(route: Sequence[Point], point: Point) -> float:
        """Marginal length (metres) of folding ``point`` into an existing OPEN
        ``route`` at its cheapest position (§10 #5): the classic insertion delta
        ``ΔL = d(a,c)+d(c,b)−d(a,b)`` minimised over adjacent pairs ``(a,b)``, or a
        plain append ``d(last,c)`` at the tail. ``route[0]`` is the fixed start (the
        robot pos), so nothing is inserted ahead of it. This is the route-aware cost
        of adding one CLEAR/visit task to the unified pool — near-zero when ``point``
        already lies on the way, which is what lets a co-located clear ride along a
        bundle instead of being charged a fresh detour. Always ``>= 0`` (triangle
        inequality); empty route -> 0."""
        pts = [(float(x), float(y)) for x, y in route]
        c = (float(point[0]), float(point[1]))
        if not pts:
            return 0.0
        best = _dist(pts[-1], c)                       # append after the last node
        for a, b in zip(pts, pts[1:]):
            best = min(best, _dist(a, c) + _dist(c, b) - _dist(a, b))
        return max(0.0, best)

    @staticmethod
    def clear_detour_value(
        future_route_saving: float, detour_delta: float, offset: float = 5.0
    ) -> float:
        """Value of making a guaranteed clear detour (``V_clear``)."""
        denominator = max(0.0, float(detour_delta)) + max(0.0, float(offset))
        return float(future_route_saving) / denominator if denominator > 0.0 else inf

    # -- uncertain neighbourhoods (TSPN) -----------------------------------

    def tspn_route(
        self, start: Point, neighborhoods: Sequence[Neighborhood]
    ) -> RoutePlan:
        """Open TSPN over neighbourhoods using the ``d_N`` approach metric; adds the
        (order-independent) expected localisation cost of each target."""
        neigh = list(neighborhoods)
        if not neigh:
            return RoutePlan([], 0.0, True)
        if self.strategy == "gtsp":
            route, length = self._gtsp_route(start, neigh)
            return RoutePlan(route, length, True,
                             localization_cost=sum(nb.localization_cost for nb in neigh))
        start_cost, cost = self._neighborhood_matrices(start, neigh)
        idx, length, exact = self._solve(start_cost, cost)
        loc = sum(nb.localization_cost for nb in neigh)
        return RoutePlan([neigh[i].center for i in idx], length, exact, localization_cost=loc)

    @staticmethod
    def _gtsp_route(start: Point, neighborhoods: Sequence[Neighborhood]):
        """Discretized TSPN: each neighborhood becomes one candidate cluster."""
        from math import cos, pi, sin
        candidates = []
        for nb in neighborhoods:
            r = max(0.0, float(nb.radius))
            points = [tuple(nb.center)]
            if r > 0.0:
                points.extend((nb.center[0] + r * cos(2*pi*k/8),
                               nb.center[1] + r * sin(2*pi*k/8)) for k in range(8))
            candidates.append(points)
        return generalized_tsp(candidates, start, _dist, max_exact=8)

    def guaranteed_clear_route(
        self,
        start: Point,
        feasible: Sequence[Neighborhood],
        clear_radius: float,
    ) -> Optional[RoutePlan]:
        """Route through guaranteed-clear sets, dropping no uncertain target.

        If any ``K_i`` is empty, no sound guaranteed-clear route exists and
        ``None`` is returned; callers must then keep sensing/localising.
        """
        clear_sets = [guaranteed_clear_neighborhood(n, clear_radius) for n in feasible]
        if any(n is None for n in clear_sets):
            return None
        return self.tspn_route(start, [n for n in clear_sets if n is not None])

    # -- reference (M6 cross-check) ----------------------------------------

    def brute_force_clear_order(self, start: Point, targets: Sequence[Point]) -> RoutePlan:
        pts = [(float(x), float(y)) for x, y in targets]
        if not pts:
            return RoutePlan([], 0.0, True)
        start_cost, cost = self._point_matrices(start, pts)
        idx, length = brute_force_open(start_cost, cost)
        return RoutePlan([pts[i] for i in idx], length, True)
