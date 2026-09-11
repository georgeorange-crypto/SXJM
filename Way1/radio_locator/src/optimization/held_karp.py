"""
Held-Karp 精确开放式 TSP（起点固定、无需返回），O(n²·2ⁿ)。

用于给定一批必到点（如各频道的清除点/覆盖点）时，求最短访问顺序。
开放式：从固定 start 出发，访问所有点各一次，不回到起点。n<=16 时可精确求解。
返回 (最短总距离, 访问顺序[含 start 为首])。
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..domain.types import Vec2


def _dist_matrix(nodes: Sequence[Vec2]) -> List[List[float]]:
    n = len(nodes)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i][j] = math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
    return D


def held_karp_open(nodes: Sequence[Vec2], start_index: int = 0) -> Tuple[float, List[int]]:
    """
    nodes[start_index] 为起点。返回最短开放路径 (距离, 索引序列)。
    n<=16 使用；更大规模请改用 2-opt。
    """
    n = len(nodes)
    if n == 0:
        return 0.0, []
    if n == 1:
        return 0.0, [start_index]
    D = _dist_matrix(nodes)
    others = [i for i in range(n) if i != start_index]
    m = len(others)
    idx = {node: b for b, node in enumerate(others)}  # bit position

    FULL = (1 << m) - 1
    INF = float("inf")
    # dp[mask][j] = 从 start 出发访问 mask 中的点、最后停在 others[j] 的最短距离
    dp = [[INF] * m for _ in range(1 << m)]
    par = [[-1] * m for _ in range(1 << m)]
    for b, node in enumerate(others):
        dp[1 << b][b] = D[start_index][node]

    for mask in range(1 << m):
        for j in range(m):
            if dp[mask][j] == INF or not (mask & (1 << j)):
                continue
            base = dp[mask][j]
            nj_node = others[j]
            for b in range(m):
                if mask & (1 << b):
                    continue
                nmask = mask | (1 << b)
                cand = base + D[nj_node][others[b]]
                if cand < dp[nmask][b]:
                    dp[nmask][b] = cand
                    par[nmask][b] = j

    best = INF
    best_j = -1
    for j in range(m):
        if dp[FULL][j] < best:
            best = dp[FULL][j]
            best_j = j

    # 回溯
    order_rev: List[int] = []
    mask, j = FULL, best_j
    while j != -1:
        order_rev.append(others[j])
        pj = par[mask][j]
        mask ^= (1 << j)
        j = pj
    order = [start_index] + list(reversed(order_rev))
    return best, order


def path_length(nodes: Sequence[Vec2], order: Sequence[int]) -> float:
    total = 0.0
    for a, b in zip(order, order[1:]):
        total += math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
    return total
