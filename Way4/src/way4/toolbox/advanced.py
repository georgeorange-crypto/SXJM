"""GTSP/TSPN candidates, bounds and route evaluation."""
from __future__ import annotations
from itertools import product, permutations
from math import hypot, inf
from typing import Sequence, Callable

def generalized_tsp(candidates: Sequence[Sequence], start, distance: Callable, max_exact: int = 8):
    """Choose one service point per neighborhood and the shortest open route."""
    groups = [list(g) for g in candidates]
    if not groups: return [], 0.0
    if len(groups) <= max_exact:
        best = (inf, None)
        for choice in product(*groups):
            for order in permutations(range(len(choice))):
                route = [choice[i] for i in order]
                val = distance(start, route[0]) + sum(distance(a,b) for a,b in zip(route, route[1:]))
                if val < best[0]: best = (val, route)
        return best[1], float(best[0])
    route = []; remaining = list(groups); cur = start
    while remaining:
        group = min(remaining, key=lambda g: min(distance(cur, p) for p in g))
        point = min(group, key=lambda p: distance(cur, p)); route.append(point); cur = point; remaining.remove(group)
    return route, float(distance(start, route[0]) + sum(distance(a,b) for a,b in zip(route, route[1:])))

def mst_lower_bound(points: Sequence, distance: Callable) -> float:
    if not points: return 0.0
    used = {0}; total = 0.0
    while len(used) < len(points):
        val, idx = min((distance(points[i], points[j]), j) for i in used for j in range(len(points)) if j not in used)
        total += val; used.add(idx)
    return float(total)

def one_tree_lower_bound(points: Sequence, distance: Callable) -> float:
    if len(points) < 2: return 0.0
    root = 0; others = list(range(1, len(points)))
    mst = mst_lower_bound([points[i] for i in others], distance)
    edges = sorted(distance(points[root], points[i]) for i in others)
    return float(mst + sum(edges[:2]))

def oracle_route(points: Sequence, start, distance: Callable):
    best = (inf, [])
    for order in permutations(points):
        value = distance(start, order[0]) + sum(distance(a,b) for a,b in zip(order, order[1:]))
        if value < best[0]: best = (value, list(order))
    return best[1], float(best[0])

def route_metrics(route: Sequence, start, distance: Callable):
    legs = [distance(start, route[0])] if route else []
    legs += [distance(a,b) for a,b in zip(route, route[1:])]
    return {"length": float(sum(legs)), "n_legs": len(legs), "n_longjump": sum(x > 1000 for x in legs)}
