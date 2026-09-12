"""
2-opt 局部搜索：当必到点数 n > Held-Karp 上限（16）时的近似 open-TSP 求解。

保持起点固定，反复反转子段以消除路径交叉，直到无改进。配合最近邻初始解，
通常能达到 MST 下界的 1.05~1.15 倍。用于 Q3/Q4 大规模覆盖点排程。
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..domain.types import Vec2
from .held_karp import path_length


def nearest_neighbor_order(nodes: Sequence[Vec2], start_index: int = 0) -> List[int]:
    n = len(nodes)
    if n <= 1:
        return list(range(n))
    unvisited = set(range(n))
    unvisited.discard(start_index)
    order = [start_index]
    cur = start_index
    while unvisited:
        nxt = min(unvisited, key=lambda j: math.hypot(
            nodes[cur][0] - nodes[j][0], nodes[cur][1] - nodes[j][1]))
        order.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return order


def two_opt(nodes: Sequence[Vec2], order: List[int] = None,
            start_index: int = 0, max_iter: int = 10000) -> Tuple[float, List[int]]:
    """
    开放式 2-opt。start 固定为 order[0]，只反转 order[1:] 内部子段。
    返回 (路径长, 顺序)。
    """
    n = len(nodes)
    if n <= 2:
        o = order or list(range(n))
        return path_length(nodes, o), o
    if order is None:
        order = nearest_neighbor_order(nodes, start_index)

    def d(a: int, b: int) -> float:
        return math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])

    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        # i 从 1 开始（保 start 固定）；开放路径末点可动
        for i in range(1, n - 1):
            a = order[i - 1]
            b = order[i]
            for k in range(i + 1, n):
                c = order[k]
                dnode = order[k + 1] if k + 1 < n else None
                # 反转 order[i..k]：断边 (a,b) 与 (c,dnode)，接 (a,c) 与 (b,dnode)
                if dnode is None:
                    # 末段反转：只影响 (a,b) → (a,c)
                    delta = d(a, c) - d(a, b)
                else:
                    delta = (d(a, c) + d(b, dnode)) - (d(a, b) + d(c, dnode))
                if delta < -1e-9:
                    order[i:k + 1] = reversed(order[i:k + 1])
                    improved = True
                    b = order[i]
            it += 1
    return path_length(nodes, order), order


def solve_route(nodes: Sequence[Vec2], start_index: int = 0,
                held_karp_max: int = 16) -> Tuple[float, List[int]]:
    """
    统一入口：n<=held_karp_max 用精确 Held-Karp，否则最近邻 + 2-opt。
    """
    n = len(nodes)
    if n <= 1:
        return 0.0, list(range(n))
    if n <= held_karp_max:
        from .held_karp import held_karp_open
        return held_karp_open(nodes, start_index)
    order = nearest_neighbor_order(nodes, start_index)
    return two_opt(nodes, order, start_index)
