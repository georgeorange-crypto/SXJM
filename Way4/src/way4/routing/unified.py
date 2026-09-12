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

    def insert_and_repair(self, current_route, opportunity, start=None):
        """Online-TSP operation used when a new service task is discovered."""
        anchor = start if start is not None else (current_route[0] if current_route else opportunity.point)
        route, _ = cheapest_insertion(
            list(current_route), opportunity.point, anchor,
            lambda a, b: dist(a, b),
        )
        return two_opt(route, anchor, lambda a, b: dist(a, b))[0]

    def route_score(self, route, *, crossing_count=0) -> float:
        """Primary route objective: length plus explicit long-jump/crossing costs."""
        pts = list(route)
        lengths = [sum((pts[i][k] - pts[i-1][k]) ** 2 for k in (0, 1)) ** 0.5
                   for i in range(1, len(pts))]
        return (sum(lengths) + self.longjump_penalty *
                sum(d > self.longjump_threshold for d in lengths) +
                self.longjump_penalty * max(0, int(crossing_count)))

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
