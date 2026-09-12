"""Route diagnostic metrics (P0 #9 — route decomposition; DESIGN.md §15 spirit).

A pure, **planner-agnostic** decomposition of one *realised* episode route into the
waste signatures the space-centric routing upgrade targets. It is computed from the
executed primitive sequence only: it reads nothing from the planner and drives
nothing back into it — no belief, no certificate, no gate depends on it, so it can
never affect correctness or the full-clear verdict (禁止10). Its sole purpose is to
let the legacy (action-centric) planner and the future ``spatial`` planner be
compared field-for-field, so the P0 #10 ablation can quantify the routing waste
*before* and *after* the upgrade.

Because the two planners use different macro vocabularies, the decomposition must not
key off ``MacroActionType``. Instead:

* A **stop** is a maximal run of consecutive primitives within ``stop_tol`` of one
  another (the 5 m ``near`` scale). A batch scan at one waypoint — plus any
  opportunistic clear taken there (same target) — is therefore ONE stop. For the
  legacy planner every same-stop primitive shares an exact target, so ``stop_tol``
  is a no-op there; it only matters for a future planner that places services in a
  small neighbourhood.
* Each primitive carries a **role** assigned by the measured channel's belief status
  *at the macro's start* (planner-agnostic — both planners measure channels that have
  a belief status):

      MEASURE on DETECTED / LOCALIZED -> 'localize'   (narrowing a known source)
      MEASURE on UNKNOWN              -> 'coverage'    (exploration / absence scan)
      CLEAR                           -> 'clear'

Metrics (all travel is measured as the leg *into* a stop, counted once per stop):

  * ``services_per_stop``       — bundling: total services / number of stops.
  * ``shared_stop_ratio``       — fraction of stops doing >= 2 services.
  * ``pure_refine_travel``      — travel into stops that ONLY localize (single-purpose
                                  localization legs the action-centric planner cannot
                                  bundle with anything else).
  * ``certificate_only_travel`` — travel into stops that ONLY do coverage.
  * ``revisit_distance``        — travel into a stop within ``revisit_tol`` of a
                                  strictly-earlier stop (backtracking waste).
  * ``clear_insertion_delta``   — sum over DEDICATED clear stops of the detour
                                  ``dL = d(prev,c) + d(c,next) - d(prev,next)``.
                                  An opportunistic clear shares its scan's waypoint,
                                  so its stop is not clear-only -> contributes 0; a
                                  standalone blind-clear routed as its own leg (legacy
                                  CLEAR macro at the MEC centre) contributes its full
                                  detour. A route optimiser that inserts clears
                                  on the way (the P0 #5 goal) drives this toward 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple
from math import isfinite

Point = Tuple[float, float]

# role constants (kept as plain strings so this module stays decoupled from belief)
ROLE_LOCALIZE = "localize"
ROLE_COVERAGE = "coverage"
ROLE_CLEAR = "clear"


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@dataclass(frozen=True)
class RouteEvent:
    """One realised primitive on the route, tagged for decomposition. Built by the
    pipeline from the executed ``PrimitiveResult`` sequence; ``role`` is assigned from
    the measured channel's belief status at the macro's start (see module docstring)."""

    loc: Point
    kind: str                 # "MEASURE" | "CLEAR"
    channel: int
    role: str                 # ROLE_LOCALIZE | ROLE_COVERAGE | ROLE_CLEAR
    cleared: bool = False     # a CLEAR that neutralised a source


@dataclass
class RouteMetrics:
    """The six P0 #9 route-decomposition metrics, plus two context counts."""

    services_per_stop: float = 0.0
    pure_refine_travel: float = 0.0
    certificate_only_travel: float = 0.0
    revisit_distance: float = 0.0
    shared_stop_ratio: float = 0.0
    clear_insertion_delta: float = 0.0
    # context (not in the P0 list, but needed to read the six above honestly)
    n_stops: int = 0
    n_services: int = 0
    total_distance: float = 0.0
    clear_distance: float = 0.0
    n_longjump: int = 0
    n_crossing: int = 0
    backtrack_m: float = 0.0
    repeated_edge_m: float = 0.0
    unnecessary_return_m: float = 0.0


@dataclass(frozen=True)
class EfficiencyMetrics:
    """Canonical episode efficiency ratios (AE01--AE07)."""
    wasted_time_ratio: float = 0.0
    backtrack_ratio: float = 0.0
    repeated_edge_ratio: float = 0.0
    unnecessary_return_ratio: float = 0.0
    scan_efficiency_s_per_useful_observation: float = 0.0
    lower_bound_ratio: Optional[float] = None
    route_efficiency: Optional[float] = None


def compute_efficiency_metrics(*, total_time_s, no_progress_time_s=0.0,
                               move_distance_m=0.0, backtrack_m=0.0,
                               repeated_edge_m=0.0, unnecessary_return_m=0.0,
                               scan_time_s=0.0, useful_observations=0,
                               lower_bound_s=None, route_lower_bound_m=None):
    """Compute ratios with explicit zero-denominator semantics."""
    vals = (total_time_s, no_progress_time_s, move_distance_m, backtrack_m,
            repeated_edge_m, unnecessary_return_m, scan_time_s)
    if not all(isfinite(float(v)) for v in vals) or any(float(v) < 0 for v in vals):
        raise ValueError('efficiency inputs must be finite and nonnegative')
    t, move = float(total_time_s), float(move_distance_m)
    useful = int(useful_observations)
    if useful < 0: raise ValueError('useful observations must be nonnegative')
    lb = None if lower_bound_s is None else float(lower_bound_s)
    route_lb = None if route_lower_bound_m is None else float(route_lower_bound_m)
    if lb is not None and (not isfinite(lb) or lb <= 0): raise ValueError('lower bound must be positive')
    if route_lb is not None and (not isfinite(route_lb) or route_lb <= 0): raise ValueError('route lower bound must be positive')
    return EfficiencyMetrics(
        wasted_time_ratio=float(no_progress_time_s) / t if t else 0.0,
        backtrack_ratio=float(backtrack_m) / move if move else 0.0,
        repeated_edge_ratio=float(repeated_edge_m) / move if move else 0.0,
        unnecessary_return_ratio=float(unnecessary_return_m) / move if move else 0.0,
        scan_efficiency_s_per_useful_observation=float(scan_time_s) / useful if useful else 0.0,
        lower_bound_ratio=t / lb if lb else None,
        route_efficiency=move / route_lb if route_lb else None,
    )


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    def o(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    e = 1e-9
    x, y, z, w = o(a,b,c), o(a,b,d), o(c,d,a), o(c,d,b)
    return ((x > e and y < -e) or (x < -e and y > e)) and ((z > e and w < -e) or (z < -e and w > e))


@dataclass
class _Stop:
    anchor: Point
    roles: Set[str]
    n_services: int
    has_clear: bool
    leg_in: float             # distance from the previous stop anchor (or start_pos)


def compute_route_metrics(
    events: Sequence[RouteEvent],
    start_pos: Point,
    stop_tol: float = 5.0,
    revisit_tol: float = 50.0,
) -> RouteMetrics:
    """Decompose a realised route into the P0 #9 diagnostics. Pure function — no
    side effects, no dependence on the planner or belief. ``start_pos`` is the robot's
    pose at episode start (the leg into the first stop is measured from there)."""
    if not events:
        return RouteMetrics()

    start = (float(start_pos[0]), float(start_pos[1]))

    # 1. cluster consecutive primitives into stops (fixed-anchor within stop_tol).
    stops: List[_Stop] = []
    prev_anchor = start
    cur: Optional[_Stop] = None
    for ev in events:
        loc = (float(ev.loc[0]), float(ev.loc[1]))
        if cur is not None and _dist(loc, cur.anchor) <= stop_tol:
            cur.roles.add(ev.role)
            cur.n_services += 1
            if ev.kind == "CLEAR":
                cur.has_clear = True
        else:
            if cur is not None:
                stops.append(cur)
                prev_anchor = cur.anchor
            cur = _Stop(
                anchor=loc,
                roles={ev.role},
                n_services=1,
                has_clear=(ev.kind == "CLEAR"),
                leg_in=_dist(prev_anchor, loc),
            )
    if cur is not None:
        stops.append(cur)

    n_stops = len(stops)
    n_services = sum(s.n_services for s in stops)

    # 2/3. bundling + single-purpose travel.
    shared = sum(1 for s in stops if s.n_services >= 2)
    pure_refine = sum(s.leg_in for s in stops if s.roles == {ROLE_LOCALIZE})
    cert_only = sum(s.leg_in for s in stops if s.roles == {ROLE_COVERAGE})

    # 4. revisit: a stop whose anchor lands within revisit_tol of a strictly-earlier
    #    stop anchor — i.e. the route came back to a neighbourhood it already left.
    revisit = 0.0
    for i, s in enumerate(stops):
        for j in range(i):
            if _dist(s.anchor, stops[j].anchor) <= revisit_tol:
                revisit += s.leg_in
                break

    # 5. clear insertion delta: only for DEDICATED clear stops (opportunistic clears
    #    share a scan's waypoint, so their stop is not clear-only -> 0 by construction).
    clear_delta = 0.0
    for i, s in enumerate(stops):
        if not (s.has_clear and s.roles == {ROLE_CLEAR}):
            continue
        prev = stops[i - 1].anchor if i > 0 else start
        if i + 1 < n_stops:
            nxt = stops[i + 1].anchor
            clear_delta += _dist(prev, s.anchor) + _dist(s.anchor, nxt) - _dist(prev, nxt)
        else:
            clear_delta += _dist(prev, s.anchor)   # terminal clear: one-way detour

    points = [start] + [s.anchor for s in stops]
    legs = list(zip(points, points[1:]))
    lengths = [_dist(a, b) for a, b in legs]
    backtrack = 0.0
    repeated = 0.0
    for i, ((a, b), length) in enumerate(zip(legs, lengths)):
        if length == 0.0:
            continue
        if i:
            (pa, pb) = legs[i - 1]
            vx, vy = b[0] - a[0], b[1] - a[1]
            wx, wy = pb[0] - pa[0], pb[1] - pa[1]
            denom = length * _dist(pa, pb)
            if denom and (vx * wx + vy * wy) / denom < -0.5:
                backtrack += length
        for j in range(i):
            old_a, old_b = legs[j]
            if ((_dist(a, old_b) <= stop_tol and _dist(b, old_a) <= stop_tol)
                    or (_dist(a, old_a) <= stop_tol and _dist(b, old_b) <= stop_tol)):
                repeated += length
                break
    unnecessary_return = revisit
    crossings = sum(_segments_cross(*legs[i], *legs[j])
                    for i in range(len(legs)) for j in range(i + 2, len(legs)))

    return RouteMetrics(
        services_per_stop=(n_services / n_stops) if n_stops else 0.0,
        pure_refine_travel=pure_refine,
        certificate_only_travel=cert_only,
        revisit_distance=revisit,
        shared_stop_ratio=(shared / n_stops) if n_stops else 0.0,
        clear_insertion_delta=clear_delta,
        n_stops=n_stops,
        n_services=n_services,
        total_distance=sum(lengths),
        clear_distance=sum(s.leg_in for s in stops if s.has_clear),
        n_longjump=sum(d > 1000.0 for d in lengths),
        n_crossing=int(crossings),
        backtrack_m=backtrack,
        repeated_edge_m=repeated,
        unnecessary_return_m=unnecessary_return,
    )
