"""Small robust/stochastic optimization helpers."""
from __future__ import annotations
from math import inf
from typing import Callable, Iterable, Sequence

def expected_cost(costs: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(costs) != len(probabilities): raise ValueError("length mismatch")
    return sum(float(c)*float(p) for c, p in zip(costs, probabilities))

def cvar(costs: Sequence[float], probabilities: Sequence[float] | None = None, alpha: float = .95) -> float:
    if not 0 < alpha < 1: raise ValueError("alpha must be in (0,1)")
    pairs = sorted(zip(costs, probabilities or [1/len(costs)]*len(costs)), reverse=True)
    need = 1-alpha; total = 0.0; mass = 0.0
    for c, p in pairs:
        take = min(float(p), need-mass); total += float(c)*take; mass += take
        if mass >= need-1e-12: break
    return total / max(mass, 1e-12)

def robust_min(options: Iterable, cost: Callable, tie_break: Callable | None = None):
    return min(options, key=lambda x: (max(cost(x)) if isinstance(cost(x), (list, tuple)) else cost(x),
                                       tie_break(x) if tie_break else 0))

def branch_and_bound(items, lower_bound, complete_cost, feasible=lambda _: True):
    best = [inf, None]
    def visit(prefix, remaining):
        if not feasible(prefix): return
        if not remaining:
            v = complete_cost(prefix)
            if v < best[0]: best[:] = [v, list(prefix)]
            return
        if lower_bound(prefix, remaining) >= best[0]: return
        for x in remaining: visit(prefix+[x], [y for y in remaining if y != x])
    visit([], list(items)); return best[1], best[0]
