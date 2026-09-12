"""Deterministic spatial clustering for task batching."""
from __future__ import annotations
from math import hypot
from typing import Sequence, List

def kmeans(points: Sequence[tuple[float,float]], k: int, iterations: int = 30):
    if not points or k <= 0: return []
    centers = [tuple(map(float, p)) for p in points[:k]]
    for _ in range(iterations):
        groups = [[] for _ in centers]
        for p in points: groups[min(range(len(centers)), key=lambda i: hypot(p[0]-centers[i][0], p[1]-centers[i][1]))].append(p)
        new = [tuple(map(float, (sum(p[0] for p in g)/len(g), sum(p[1] for p in g)/len(g)))) if g else centers[i] for i,g in enumerate(groups)]
        if new == centers: break
        centers = new
    return centers, groups

def dbscan(points: Sequence[tuple[float,float]], eps: float, min_samples: int = 2):
    labels = [-1]*len(points); cid = 0
    if eps <= 0: return labels
    # Spatial hashing avoids the O(n^2) all-pairs scan when the candidate pool
    # contains many repeated/nearby waypoints.  We inspect the 3x3 neighboring
    # buckets, then retain the exact Euclidean distance predicate.
    cell = float(eps)
    buckets = {}
    for i, (x, y) in enumerate(points):
        key = (int(x // cell), int(y // cell))
        buckets.setdefault(key, []).append(i)
    def near(i):
        x, y = points[i]; bx, by = int(x // cell), int(y // cell)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((bx + dx, by + dy), ()):
                    q = points[j]
                    if hypot(x-q[0], y-q[1]) <= eps:
                        out.append(j)
        return out
    for i in range(len(points)):
        if labels[i] != -1: continue
        ns = near(i)
        if len(ns) < min_samples: continue
        labels[i] = cid; queue = list(ns)
        while queue:
            j = queue.pop()
            if labels[j] == -1: labels[j] = cid
            n2 = near(j)
            if len(n2) >= min_samples: queue.extend(x for x in n2 if labels[x] == -1)
        cid += 1
    return labels
