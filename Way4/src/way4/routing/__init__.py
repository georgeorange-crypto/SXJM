"""Routing — exact Held–Karp open TSP + TSPN + 2-opt fallback (DESIGN.md §9)."""

from .estimator import (
    LocalizationCostModel,
    Neighborhood,
    RouteEstimator,
    RoutePlan,
    guaranteed_clear_point,
)
from .tsp import (
    brute_force_open,
    held_karp_min_path,
    held_karp_open,
    nearest_neighbor_open,
    two_opt_open,
)
from .unified import ServiceOpportunity, UnifiedRoutePlanner

__all__ = [
    "LocalizationCostModel",
    "Neighborhood",
    "RouteEstimator",
    "RoutePlan",
    "guaranteed_clear_point",
    "brute_force_open",
    "held_karp_min_path",
    "held_karp_open",
    "nearest_neighbor_open",
    "two_opt_open",
    "ServiceOpportunity",
    "UnifiedRoutePlanner",
]
