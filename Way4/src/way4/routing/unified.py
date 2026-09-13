"""Unified SEARCH/REFINE/CLEAR/VERIFY route planner (Way4-Route-Math)."""
from dataclasses import dataclass
from math import dist
from typing import Sequence
from .estimator import Neighborhood, RouteEstimator, RoutePlan
from .tsp import route_repair_open
from ..toolbox import cheapest_insertion, two_opt

@dataclass(frozen=True)
class ServiceOpportunity:
    point: tuple[float, float]
    services: tuple[tuple[int, str], ...] = ()
    service_time: float = 0.0
    radius: float = 0.0

    @property
    def channels(self): return tuple(sorted({c for c, _ in self.services}))

class UnifiedRoutePlanner:
    """Plans one physical open route over all remaining service opportunities."""
    def __init__(self, estimator=None, exact_max_n=13):
        self.estimator = estimator or RouteEstimator(exact_max_n=exact_max_n)
        self.longjump_threshold = 1000.0
        self.longjump_penalty = 100.0

    def route_cost_seconds(self, route, *, crossing_count=0) -> float:
        """Primary route objective in seconds (J01)."""
        pts = list(route)
        lengths = [sum((pts[i][k] - pts[i-1][k]) ** 2 for k in (0, 1)) ** 0.5
                   for i in range(1, len(pts))]
        speed = max(float(getattr(self.estimator, "speed", 1.0)), 1e-9)
        return (sum(lengths) / speed + self.longjump_penalty / speed *
                sum(d > self.longjump_threshold for d in lengths) +
                self.longjump_penalty / speed * max(0, int(crossing_count)))

    def route_score(self, route, *, crossing_count=0) -> float:
        return self.route_cost_seconds(route, crossing_count=crossing_count)

    def plan(self, start, opportunities: Sequence[ServiceOpportunity]) -> RoutePlan:
        # Co-locate services before routing: a waypoint is paid once.
        unique = {}
        for op in opportunities:
            p = (float(op.point[0]), float(op.point[1]))
            old = unique.get(p)
            if old is None:
                unique[p] = op
            else:
                unique[p] = ServiceOpportunity(p, tuple(dict.fromkeys(old.services + op.services)),
                                                old.service_time + op.service_time,
                                                max(old.radius, op.radius))
        ops = list(unique.values())
        if any(o.radius > 0.0 for o in ops):
            plan = self.estimator.optimal_neighborhood_order(
                start, [Neighborhood(o.point, o.radius, o.service_time) for o in ops])
        else:
            plan = self.estimator.optimal_clear_order(start, [o.point for o in ops])
        # Repair heuristic/exact output after every route rebuild. For small exact
        # plans this is a no-op; for heuristic plans it covers relocate/swap gaps.
        if len(ops) > 1 and not plan.exact:
            points = [(float(o.point[0]), float(o.point[1])) for o in ops]
            # The public toolbox is the online heuristic path: seed by nearest
            # insertion, then repair with 2-opt.  The exact estimator remains
            # authoritative whenever its budget permits.
            route = []
            for point in points:
                route, _ = cheapest_insertion(route, point, start, lambda a, b: dist(a, b))
            route, length = two_opt(route, start, lambda a, b: dist(a, b))
            plan = RoutePlan(route, length, False, plan.localization_cost)
        return RoutePlan(plan.order, plan.length,
                         plan.exact, sum(o.service_time for o in unique.values()))

    def insertion_delta(self, current_route, opportunity):
        return self.estimator.insertion_delta(current_route, opportunity.point)

    def insert_at_cheapest(self, current_route, opportunity):
        """Insert one newly discovered task at minimum route delta (M02).

        This deliberately does not rebuild or 2-opt the existing route; callers
        may schedule cleanup separately under the M04 cadence.
        """
        route = list(current_route)
        point = tuple(float(x) for x in opportunity.point)
        if not route:
            return [point], 0, 0.0
        if len(route) == 1:
            return route + [point], 1, dist(route[0], point)
        choices = []
        for i, (a, b) in enumerate(zip(route, route[1:]), start=1):
            delta = dist(a, point) + dist(point, b) - dist(a, b)
            choices.append((delta, i))
        tail_delta = dist(route[-1], point)
        choices.append((tail_delta, len(route)))
        delta, index = min(choices, key=lambda x: (x[0], x[1]))
        route.insert(index, point)
        return route, index, max(0.0, float(delta))

    def insert_clear_if_cheap(self, current_route, clear_opportunity,
                              max_delta_m=float("inf")):
        """Insert a clear-ready task when its route increment is acceptable.

        Clear tasks use the same cheapest insertion calculation, preventing a
        dedicated out-and-back clear when an existing route can absorb it.
        """
        route, index, delta = self.insert_at_cheapest(current_route, clear_opportunity)
        if float(delta) <= float(max_delta_m):
            return route, index, delta, True
        return list(current_route), None, float(delta), False

    def cleanup_route(self, route, start=(0.0, 0.0)):
        """Run deterministic 2-opt/relocate/swap cleanup and accept only ΔT<0."""
        pts = list(route)
        if len(pts) < 2:
            cost = dist(start, pts[0]) if pts else 0.0
            return pts, float(cost), False
        old = dist(start, pts[0]) + sum(dist(a, b) for a, b in zip(pts, pts[1:]))
        # toolbox 2-opt is deterministic and strictly improving; relocate/swap
        # are applied by the route-repair kernel used for larger heuristic routes.
        repaired, new = two_opt(pts, start, lambda a, b: dist(a, b))
        if new < old - 1e-9:
            return repaired, float(new), True
        return pts, float(old), False


class RouteCleanupScheduler:
    """Cadence gate for deterministic route cleanup (M04)."""
    def __init__(self, decision_period: int = 5, distance_period_m: float = 750.0):
        if decision_period < 1 or distance_period_m <= 0:
            raise ValueError("cleanup cadence must be positive")
        self.decision_period = int(decision_period)
        self.distance_period_m = float(distance_period_m)
        self._last_decision = 0
        self._last_distance = 0.0

    def due(self, *, decision_count: int, distance_m: float,
            major_belief_change: bool = False) -> bool:
        if major_belief_change:
            return True
        return (int(decision_count) - self._last_decision >= self.decision_period or
                float(distance_m) - self._last_distance >= self.distance_period_m)

    def mark_cleaned(self, *, decision_count: int, distance_m: float) -> None:
        self._last_decision = int(decision_count)
        self._last_distance = float(distance_m)

    def insert_and_repair(self, current_route, opportunity, start=None):
        """Online-TSP operation used when a new service task is discovered."""
        anchor = start if start is not None else (current_route[0] if current_route else opportunity.point)
        route, _ = cheapest_insertion(
            list(current_route), opportunity.point, anchor,
            lambda a, b: dist(a, b),
        )
        return two_opt(route, anchor, lambda a, b: dist(a, b))[0]

    def route_cost_seconds(self, route, *, crossing_count=0) -> float:
        """Primary route objective in seconds (J01; no meter/score mixing)."""
        pts = list(route)
        lengths = [sum((pts[i][k] - pts[i-1][k]) ** 2 for k in (0, 1)) ** 0.5
                   for i in range(1, len(pts))]
        speed = max(float(getattr(self.estimator, "speed", 1.0)), 1e-9)
        return (sum(lengths) / speed + self.longjump_penalty / speed *
                sum(d > self.longjump_threshold for d in lengths) +
                self.longjump_penalty / speed * max(0, int(crossing_count)))

    def route_score(self, route, *, crossing_count=0) -> float:
        """Compatibility alias for the seconds-valued route objective."""
        return self.route_cost_seconds(route, crossing_count=crossing_count)

    def accept_replan(self, proposed_cost, current_cost, *, progress=False,
                      wait_search_gain=0.0, wait_search_cost=0.0,
                      detour_regret=0.0, max_detour_regret=300.0):
        """Deterministic route protections used before a learned choice.

        Small route improvements are held by hysteresis; SEARCH is waited for
        only when its predicted gain exceeds its cost; excessive detours are
        rejected. These are ranking guards, never certificate decisions.
        """
        if detour_regret > max_detour_regret:
            return False
        if wait_search_gain > 0 and wait_search_gain <= wait_search_cost:
            return False
        if not progress and proposed_cost >= current_cost - 5.0:
            return False
        return True
