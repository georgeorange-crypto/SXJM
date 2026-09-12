"""Algorithm Toolbox: small, deterministic kernels used by Way4.

The toolbox is deliberately dependency-free.  It exposes safe mathematical
building blocks; callers still own execution and certificate decisions.
"""

from .routing import cheapest_insertion, rolling_subset_dp, two_opt
from .coverage import greedy_set_cover, maximum_coverage
from .information import entropy, information_gain, value_of_information
from .probability import bayesian_update, gaussian_mixture, kde, Particle, ParticleFilter
from .graph import dijkstra, astar, d_star_lite, lpa_star
from .optimization import expected_cost, cvar, robust_min, branch_and_bound
from .clustering import kmeans, dbscan
from .advanced import generalized_tsp, mst_lower_bound, one_tree_lower_bound, oracle_route, route_metrics
from .planning import beam_search, mpc, HyperHeuristic, shield, bounded_residual

__all__ = [
    "cheapest_insertion", "rolling_subset_dp", "two_opt",
    "greedy_set_cover", "maximum_coverage",
    "entropy", "information_gain", "value_of_information",
    "bayesian_update", "gaussian_mixture", "kde", "Particle", "ParticleFilter",
    "dijkstra", "astar", "d_star_lite", "lpa_star",
    "expected_cost", "cvar", "robust_min", "branch_and_bound",
    "kmeans", "dbscan",
    "generalized_tsp", "mst_lower_bound", "one_tree_lower_bound", "oracle_route", "route_metrics",
    "beam_search", "mpc", "HyperHeuristic", "shield", "bounded_residual",
]
