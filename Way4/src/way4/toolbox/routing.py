"""Deterministic route kernels used by the online planner."""
from __future__ import annotations
from math import inf
from typing import Callable, Sequence, Tuple, List, TypeVar

T = TypeVar("T")

def _length(route, start, distance):
    if not route:
        return 0.0
    total = distance(start, route[0])
    total += sum(distance(a, b) for a, b in zip(route, route[1:]))
    return float(total)

def cheapest_insertion(route: Sequence[T], item: T, start: T,
                       distance: Callable[[T, T], float]) -> Tuple[List[T], float]:
    """Insert ``item`` at the minimum open-route cost increase.

    The route is not mutated.  For an empty route the result is ``[item]``.
    """
    base = list(route)
    if not base:
        return [item], float(distance(start, item))
    best_route, best_delta = None, inf
    for i in range(len(base) + 1):
        cand = base[:i] + [item] + base[i:]
        delta = _length(cand, start, distance) - _length(base, start, distance)
        if delta < best_delta - 1e-12:
            best_route, best_delta = cand, delta
    return best_route, float(_length(best_route, start, distance))

def two_opt(route: Sequence[T], start: T, distance: Callable[[T, T], float],
            max_passes: int = 40) -> Tuple[List[T], float]:
    """Open-route 2-opt with full rescoring (supports asymmetric costs)."""
    best = list(route)
    value = _length(best, start, distance)
    for _ in range(max(0, int(max_passes))):
        improved = False
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                score = _length(cand, start, distance)
                if score < value - 1e-9:
                    best, value, improved = cand, score, True
        if not improved:
            break
    return best, float(value)

def rolling_subset_dp(items: Sequence[T], start: T, distance: Callable[[T, T], float],
                      k: int = 8) -> Tuple[List[T], float]:
    """Exact open Held--Karp route over the first ``k`` supplied items."""
    selected = list(items)[:max(0, int(k))]
    n = len(selected)
    if not n:
        return [], 0.0
    dp = {(1 << i, i): (float(distance(start, selected[i])), -1) for i in range(n)}
    for mask in range(1 << n):
        for last in range(n):
            cur = dp.get((mask, last))
            if cur is None:
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                key = (mask | (1 << nxt), nxt)
                val = cur[0] + distance(selected[last], selected[nxt])
                if key not in dp or val < dp[key][0]:
                    dp[key] = (val, last)
    full = (1 << n) - 1
    last = min(range(n), key=lambda i: dp[(full, i)][0])
    order, mask = [], full
    while last >= 0:
        order.append(last)
        prev = dp[(mask, last)][1]
        mask ^= 1 << last
        last = prev
    order.reverse()
    return [selected[i] for i in order], float(dp[(full, order[-1])][0])
