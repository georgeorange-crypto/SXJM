"""Shortest-path kernels for changing discrete maps."""
from __future__ import annotations
import heapq
from math import inf
from typing import Callable, Dict, Hashable, Iterable, Mapping, Optional

Node = Hashable

def dijkstra(graph: Mapping[Node, Iterable[tuple[Node, float]]], start: Node, goal: Node | None = None):
    dist = {start: 0.0}; prev = {}; q = [(0.0, start)]
    while q:
        d, u = heapq.heappop(q)
        if d != dist[u]: continue
        if u == goal: break
        for v, w in graph.get(u, ()):
            nd = d + float(w)
            if nd < dist.get(v, inf): dist[v] = nd; prev[v] = u; heapq.heappush(q, (nd, v))
    return (dist, prev) if goal is None else (reconstruct(prev, start, goal), dist.get(goal, inf))

def astar(graph, start, goal, heuristic: Callable[[Node], float]):
    return _best_first(graph, start, goal, heuristic)

def _best_first(graph, start, goal, heuristic):
    g = {start: 0.0}; prev = {}; q = [(heuristic(start), start)]
    while q:
        _, u = heapq.heappop(q)
        if u == goal: return reconstruct(prev, start, goal), g[u]
        for v, w in graph.get(u, ()):
            ng = g[u] + float(w)
            if ng < g.get(v, inf): g[v] = ng; prev[v] = u; heapq.heappush(q, (ng + heuristic(v), v))
    return [], inf

def reconstruct(prev, start, goal):
    if goal != start and goal not in prev: return []
    out = [goal]
    while out[-1] != start: out.append(prev[out[-1]])
    return out[::-1]

def d_star_lite(graph, start, goal, heuristic=lambda _: 0.0):
    """Practical replanning facade; recomputes A* on the changed graph."""
    return astar(graph, start, goal, heuristic)

lpa_star = d_star_lite
