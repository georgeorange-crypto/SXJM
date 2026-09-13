"""Deterministic small-task endgame solver for final unresolved work."""
from dataclasses import dataclass
from math import hypot
from typing import Sequence


@dataclass(frozen=True)
class EndgamePlan:
    order: tuple
    route_length: float
    used_exact: bool


def solve_endgame(start, points: Sequence[tuple[float, float]], *, exact_limit: int = 10) -> EndgamePlan:
    """Solve a small open route exactly with Held-Karp dynamic programming."""
    pts = tuple((float(x), float(y)) for x, y in points)
    if exact_limit < 1:
        raise ValueError("exact_limit must be positive")
    origin = (float(start[0]), float(start[1]))
    if not pts:
        return EndgamePlan((), 0.0, True)
    n = len(pts)
    if n > exact_limit:
        remaining = list(range(n)); current = origin; order = []; length = 0.0
        while remaining:
            i = min(remaining, key=lambda j: (hypot(pts[j][0]-current[0], pts[j][1]-current[1]), j))
            length += hypot(pts[i][0]-current[0], pts[i][1]-current[1])
            order.append(pts[i]); current = pts[i]; remaining.remove(i)
        return EndgamePlan(tuple(order), length, False)
    dp = {(1 << i, i): (hypot(pts[i][0]-origin[0], pts[i][1]-origin[1]), (i,)) for i in range(n)}
    for size in range(2, n + 1):
        for mask in range(1 << n):
            if mask.bit_count() != size:
                continue
            for last in range(n):
                if mask & (1 << last):
                    prior = mask ^ (1 << last)
                    options = [(dp[(prior, j)][0] + hypot(pts[j][0]-pts[last][0], pts[j][1]-pts[last][1]),
                                dp[(prior, j)][1] + (last,)) for j in range(n) if prior & (1 << j)]
                    dp[(mask, last)] = min(options, key=lambda item: (item[0], item[1]))
    length, indices = min((dp[((1 << n)-1, i)] for i in range(n)), key=lambda item: (item[0], item[1]))
    return EndgamePlan(tuple(pts[i] for i in indices), length, True)


class EndgameController:
    """Select the exact-route first candidate below an unresolved threshold."""
    def __init__(self, unresolved_threshold: int = 3, exact_limit: int = 10):
        if unresolved_threshold < 0 or exact_limit < 1:
            raise ValueError("endgame thresholds must be valid")
        self.unresolved_threshold = int(unresolved_threshold)
        self.exact_limit = int(exact_limit)

    def select(self, start, candidates, unresolved_count: int):
        if int(unresolved_count) > self.unresolved_threshold:
            return None
        options = [c for c in candidates if getattr(c, "action_type", None).value != "EXIT"]
        if not options:
            return None
        plan = solve_endgame(start, [c.target for c in options], exact_limit=self.exact_limit)
        first = plan.order[0]
        return min((c for c in options if tuple(c.target) == tuple(first)),
                   key=lambda c: (float(c.expected_time), repr(c.target)))
