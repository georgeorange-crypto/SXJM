"""Greedy coverage kernels; certificates remain in ``way4.certificate``."""
from __future__ import annotations
from typing import Iterable, Sequence, Set, Tuple, List

def greedy_set_cover(universe: Iterable, sets: Sequence[Iterable], max_sets=None) -> List[int]:
    remaining = set(universe); chosen = []
    limit = len(sets) if max_sets is None else max(0, int(max_sets))
    while remaining and len(chosen) < limit:
        best = max((i for i in range(len(sets)) if i not in chosen),
                   key=lambda i: len(remaining.intersection(set(sets[i]))), default=None)
        if best is None or not remaining.intersection(set(sets[best])):
            break
        chosen.append(best); remaining.difference_update(set(sets[best]))
    return chosen

def maximum_coverage(universe: Iterable, sets: Sequence[Iterable], budget: int) -> Tuple[List[int], Set]:
    chosen = greedy_set_cover(universe, sets, budget)
    covered = set().union(*(set(sets[i]) for i in chosen)) if chosen else set()
    return chosen, covered
