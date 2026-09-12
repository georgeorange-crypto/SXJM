"""Unified SEARCH/REFINE/CLEAR/VERIFY route planner (Way4-Route-Math)."""
from dataclasses import dataclass
from typing import Sequence
from .estimator import RouteEstimator, RoutePlan

@dataclass(frozen=True)
class ServiceOpportunity:
    point: tuple[float, float]
    services: tuple[tuple[int, str], ...] = ()
    service_time: float = 0.0

    @property
    def channels(self): return tuple(sorted({c for c, _ in self.services}))

class UnifiedRoutePlanner:
    """Plans one physical open route over all remaining service opportunities."""
    def __init__(self, estimator=None, exact_max_n=13):
        self.estimator = estimator or RouteEstimator(exact_max_n=exact_max_n)

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
                                                old.service_time + op.service_time)
        plan = self.estimator.optimal_clear_order(start, list(unique))
        return RoutePlan(plan.order, plan.length,
                         plan.exact, sum(o.service_time for o in unique.values()))

    def insertion_delta(self, current_route, opportunity):
        return self.estimator.insertion_delta(current_route, opportunity.point)

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
