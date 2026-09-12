"""Exact & heuristic open-path TSP kernels (DESIGN.md §9, M6).

*Open* TSP = a Hamiltonian PATH (visit every node once, no return to start) — the
right model for "drive from where I am and clear all remaining sources", where the
robot does not come home. Three kernels, all over a general (possibly asymmetric)
cost matrix so the same code serves fixed clear points and TSPN neighbourhoods:

  * ``held_karp_open``     — exact, anchored at ``start`` (O(2^n·n²)); the planner
    memoises it and re-solves ONLY when the clearable set changes (§9 工程约束).
  * ``held_karp_min_path`` — exact, both endpoints free: the minimum path over a
    SET, independent of the live start (the cached, start-free part of J_route).
  * ``brute_force_open``   — reference by permutation, for the M6 cross-check.
  * ``nearest_neighbor_open`` + ``two_opt_open`` — the Way3-style fallback for n
    beyond the exact budget.
"""

from __future__ import annotations

from itertools import permutations
from math import inf
from typing import List, Sequence, Tuple

Matrix = Sequence[Sequence[float]]


def _path_length(order: Sequence[int], start_cost: Sequence[float], cost: Matrix) -> float:
    if not order:
        return 0.0
    total = start_cost[order[0]]
    for i in range(len(order) - 1):
        total += cost[order[i]][order[i + 1]]
    return total


def held_karp_open(start_cost: Sequence[float], cost: Matrix) -> Tuple[List[int], float]:
    """Exact minimum open path from ``start`` (encoded by ``start_cost[j]`` = the
    cost start→node j) visiting every node once. Returns (order, length)."""
    n = len(start_cost)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], float(start_cost[0])

    size = 1 << n
    dp = [[inf] * n for _ in range(size)]
    par = [[-1] * n for _ in range(size)]
    for j in range(n):
        dp[1 << j][j] = float(start_cost[j])

    for mask in range(size):
        row = dp[mask]
        for j in range(n):
            cur = row[j]
            if cur == inf or not (mask >> j) & 1:
                continue
            cj = cost[j]
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                nc = cur + cj[k]
                if nc < dp[nm][k]:
                    dp[nm][k] = nc
                    par[nm][k] = j

    full = size - 1
    best, end = inf, 0
    for j in range(n):
        if dp[full][j] < best:
            best, end = dp[full][j], j

    order: List[int] = []
    mask, j = full, end
    while j != -1:
        order.append(j)
        pj = par[mask][j]
        mask ^= 1 << j
        j = pj
    order.reverse()
    return order, float(best)


def held_karp_min_path(cost: Matrix) -> Tuple[List[int], float]:
    """Exact minimum Hamiltonian path with BOTH endpoints free (start anywhere,
    end anywhere) — the intra-set travel of a target set, independent of the live
    robot start. Returns (order, length)."""
    n = len(cost)
    if n <= 1:
        return list(range(n)), 0.0
    size = 1 << n
    dp = [[inf] * n for _ in range(size)]
    par = [[-1] * n for _ in range(size)]
    for j in range(n):
        dp[1 << j][j] = 0.0

    for mask in range(size):
        row = dp[mask]
        for j in range(n):
            cur = row[j]
            if cur == inf or not (mask >> j) & 1:
                continue
            cj = cost[j]
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                nc = cur + cj[k]
                if nc < dp[nm][k]:
                    dp[nm][k] = nc
                    par[nm][k] = j

    full = size - 1
    best, end = inf, 0
    for j in range(n):
        if dp[full][j] < best:
            best, end = dp[full][j], j

    order: List[int] = []
    mask, j = full, end
    while j != -1:
        order.append(j)
        pj = par[mask][j]
        mask ^= 1 << j
        j = pj
    order.reverse()
    return order, float(best)


def brute_force_open(start_cost: Sequence[float], cost: Matrix) -> Tuple[List[int], float]:
    """Reference solver by full permutation (n small only — the M6 oracle)."""
    n = len(start_cost)
    if n == 0:
        return [], 0.0
    best, best_perm = inf, None
    for perm in permutations(range(n)):
        c = _path_length(perm, start_cost, cost)
        if c < best:
            best, best_perm = c, list(perm)
    return best_perm, float(best)


def nearest_neighbor_open(start_cost: Sequence[float], cost: Matrix) -> Tuple[List[int], float]:
    """Greedy nearest-neighbour open path — the 2-opt seed / large-n fallback."""
    n = len(start_cost)
    if n == 0:
        return [], 0.0
    cur = min(range(n), key=lambda j: start_cost[j])
    order = [cur]
    unvisited = set(range(n))
    unvisited.discard(cur)
    while unvisited:
        nxt = min(unvisited, key=lambda k: cost[cur][k])
        order.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return order, _path_length(order, start_cost, cost)


def two_opt_open(
    order: Sequence[int],
    start_cost: Sequence[float],
    cost: Matrix,
    max_passes: int = 60,
) -> Tuple[List[int], float]:
    """2-opt local search on an open path (segment reversals; full re-score, so it
    is valid for the mildly asymmetric TSPN metric too). Way3's fallback for n
    beyond the exact budget."""
    best = list(order)
    best_len = _path_length(best, start_cost, cost)
    n = len(best)
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            for k in range(i + 1, n):
                cand = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                cl = _path_length(cand, start_cost, cost)
                if cl < best_len - 1e-9:
                    best, best_len = cand, cl
                    improved = True
        if not improved:
            break
    return best, best_len


def route_repair_open(order: Sequence[int], start_cost: Sequence[float],
                      cost: Matrix, max_passes: int = 20) -> Tuple[List[int], float]:
    """Apply 2-opt, relocate and swap until a local open-route minimum.

    The operators are deterministic and cost-only; they never alter the task set,
    so callers can use this as a safe geometric repair after discovery/clear events.
    """
    best, best_len = two_opt_open(order, start_cost, cost)
    n = len(best)
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = list(best); item = cand.pop(i); cand.insert(j, item)
                value = _path_length(cand, start_cost, cost)
                if value < best_len - 1e-9:
                    best, best_len, improved = cand, value, True
        for i in range(n):
            for j in range(i + 1, n):
                cand = list(best); cand[i], cand[j] = cand[j], cand[i]
                value = _path_length(cand, start_cost, cost)
                if value < best_len - 1e-9:
                    best, best_len, improved = cand, value, True
        if not improved:
            break
    return best, best_len
