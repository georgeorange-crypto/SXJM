"""
MST 下界与 Prim 实现：为路径规划提供【可采纳下界】。

开放式 TSP 从 start 出发的最短路长 >= 以 start 为根的最小生成树权重（MST 是任何
生成树的下界，而一条哈密顿路径本身就是一棵生成树）。用于：
  - 分支限界/剪枝的下界；
  - 评估 2-opt 解的质量（解 / 下界 的比值）；
  - receding-horizon 中对“剩余任务代价”的乐观估计（admissible heuristic）。
"""
from __future__ import annotations

import math
from typing import List, Sequence

from ..domain.types import Vec2


def mst_weight(nodes: Sequence[Vec2]) -> float:
    """Prim 求完全图欧氏 MST 总权重。"""
    n = len(nodes)
    if n <= 1:
        return 0.0
    INF = float("inf")
    in_tree = [False] * n
    dist = [INF] * n
    dist[0] = 0.0
    total = 0.0
    for _ in range(n):
        u = -1
        best = INF
        for i in range(n):
            if not in_tree[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1:
            break
        in_tree[u] = True
        total += best
        ux, uy = nodes[u]
        for v in range(n):
            if not in_tree[v]:
                d = math.hypot(ux - nodes[v][0], uy - nodes[v][1])
                if d < dist[v]:
                    dist[v] = d
    return total


def open_tsp_lower_bound(nodes: Sequence[Vec2]) -> float:
    """开放式 TSP 下界：MST 权重（含 start，因 start 也是待连接节点）。"""
    return mst_weight(nodes)
