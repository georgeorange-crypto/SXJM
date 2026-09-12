"""Way3 homing controller — the sound directional localizer (DESIGN.md §6.10, §11).

Ported faithfully ("Way3 homing 原样移植") from Way3's ``Hunter._home_and_clear``
family. Way4's omni ``MinimaxNBV`` assumes an *omnidirectional* source: to shrink a
channel's belief it drives the robot to viewpoints that would tighten an omni disc.
For a *directional* source (a 180° arc whose direction is unknown) those viewpoints
routinely land in the source's blind half-plane and return NO_SIGNAL — and 禁止5
forbids using a directional NO_SIGNAL to subtract a range disc (a source in range
but facing away is also silent). So the belief's ``F_c`` never shrinks and the
channel livelocks: the ``no_progress_stall`` seen on P4 seeds 2000/2002/2003, each
of which clears all-but-one source then spins.

Way3's homing is the DESIGN-prescribed remedy and is **sound for both problems**
(Way3 runs this exact routine on P3 at 100%): it never relies on negative info. From
a single bearing it builds a baseline by stepping **perpendicular** to the bearing
(overshoot-immune — perpendicular motion cannot cross the source into its blind
back), then orbits the estimate taking viewpoints toward the **circular mean of the
directions signal was actually received** (guaranteed inside the 180° arc), shrinking
the localization region until a blind clear at its centre is guaranteed to hit
(region MEC radius ≤ clear margin < 20 m).

The controller drives the env through the ``MacroExecutor`` folding path
(``step_measure``/``step_clear``), so every measurement and clear updates belief +
certificate exactly as a planned macro would (Invariants B/D preserved — it marks
nothing and certifies nothing itself; clears are real env hits). It keeps a local
Way3-style fix (bearing-ray intersections) for its *decisions* only, which is what
preserves Way3's proven geometry; Way4's ``F_c`` belief is updated in parallel and
owns the certificate/resolved bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from sxjm_core.geometry import bearing_deg, min_enclosing_circle, norm_deg
from way4.certificate.fallback import ray_sweep_points, triangular_clear_sweep_points

Point = Tuple[float, float]

# ±1° svd error half-width used to build the localization region (附件2 §2.3).
_SVD_HALF = 1.0


# --------------------------------------------------------------------------- #
# pure geometry, ported from Way3's jammerhunt.geometry (stdlib-only, testable)
# --------------------------------------------------------------------------- #
def _unit(bearing: float) -> Point:
    """Bearing (deg, CCW from +x) -> unit direction vector."""
    r = math.radians(bearing)
    return math.cos(r), math.sin(r)


def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute separation between two bearings, in [0, 180]."""
    d = abs(norm_deg(a) - norm_deg(b))
    return d if d <= 180.0 else 360.0 - d


def _crossing_angle(b1: float, b2: float) -> float:
    """Crossing angle of two bearing lines, in (0, 90]; ~90° is the best fix."""
    d = _ang_diff(b1, b2)
    return min(d, 180.0 - d)


def _ray_intersect(p1: Point, b1: float, p2: Point, b2: float) -> Optional[Point]:
    """Intersection of the lines through ``p1``/``p2`` at bearings ``b1``/``b2``.
    Returns None when near-parallel."""
    u1x, u1y = _unit(b1)
    u2x, u2y = _unit(b2)
    denom = u1x * u2y - u1y * u2x
    if abs(denom) < 1e-12:
        return None
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    t1 = (dx * u2y - dy * u2x) / denom
    return p1[0] + t1 * u1x, p1[1] + t1 * u1y


def _least_squares_fix(obs: List[Tuple[Point, float]]) -> Optional[Point]:
    """Least-squares crossing of >=2 bearing lines (min sum of squared perp dists).
    Returns None if singular (all lines parallel)."""
    Sxx = Sxy = Syy = bx = by = 0.0
    n = 0
    for (px, py), bdeg in obs:
        c, s = _unit(bdeg)
        ax, ay, d = s, -c, s * px - c * py
        Sxx += ax * ax
        Sxy += ax * ay
        Syy += ay * ay
        bx += ax * d
        by += ay * d
        n += 1
    if n < 2:
        return None
    det = Sxx * Syy - Sxy * Sxy
    if abs(det) < 1e-9:
        return None
    return (Syy * bx - Sxy * by) / det, (Sxx * by - Sxy * bx) / det


def _best_pair_fix(obs: List[Tuple[Point, float]]) -> Optional[Point]:
    """Intersection of the widest-crossing observation pair; else least-squares."""
    best = None
    best_ang = -1.0
    m = len(obs)
    for i in range(m):
        for j in range(i + 1, m):
            (p1, b1), (p2, b2) = obs[i], obs[j]
            ang = _crossing_angle(b1, b2)
            if ang > best_ang:
                pt = _ray_intersect(p1, b1, p2, b2)
                if pt is not None:
                    best_ang, best = ang, pt
    return best if best is not None else _least_squares_fix(obs)


def _localization_region(
    p1: Point, b1: float, p2: Point, b2: float, half: float = _SVD_HALF
) -> Optional[List[Point]]:
    """The convex quad where the two ±half° wedges cross (真源必在其中)."""
    corners: List[Point] = []
    for d1 in (-half, half):
        for d2 in (-half, half):
            pt = _ray_intersect(p1, b1 + d1, p2, b2 + d2)
            if pt is None:
                return None
            corners.append(pt)
    return corners


def _region_center_radius(
    p1: Point, b1: float, p2: Point, b2: float, half: float = _SVD_HALF
) -> Optional[Tuple[Point, float]]:
    corners = _localization_region(p1, b1, p2, b2, half)
    if corners is None:
        return None
    return min_enclosing_circle(corners)


def _best_region(
    obs: List[Tuple[Point, float]], half: float = _SVD_HALF
) -> Optional[Tuple[Point, float]]:
    """Over all observation pairs, the smallest localization-region MEC (centre, r)."""
    best = None
    best_r = math.inf
    m = len(obs)
    for i in range(m):
        for j in range(i + 1, m):
            (p1, b1), (p2, b2) = obs[i], obs[j]
            if _crossing_angle(b1, b2) < 2.0:      # too parallel: skip
                continue
            cr = _region_center_radius(p1, b1, p2, b2, half)
            if cr is not None and cr[1] < best_r:
                best_r, best = cr[1], cr
    return best


@dataclass
class _Fix:
    """Way3's ``ChannelState`` fix state: observations + current point estimate.

    ``obs`` are (measurement point, svd bearing) pairs; ``est``/``region_r`` are the
    best crossing estimate and its localization-region MEC radius (inf when a single
    or near-parallel set gives no reliable crossing)."""

    obs: List[Tuple[Point, float]] = field(default_factory=list)
    est: Optional[Point] = None
    region_r: float = math.inf

    def refresh(self) -> None:
        if len(self.obs) >= 2:
            br = _best_region(self.obs)
            if br is not None:
                self.est, self.region_r = br
                return
            est = _best_pair_fix(self.obs)
            self.est = est if est is not None else _least_squares_fix(self.obs)
            self.region_r = math.inf
        elif len(self.obs) == 1:
            self.est, self.region_r = None, math.inf


class HomingController:
    """Way3-faithful homing for a detected (possibly directional) source.

    Drives the env through ``executor.step_measure``/``step_clear`` so belief +
    certificate fold identically to a planned macro. State is threaded call-scoped
    (mirroring Way3's ``world``): ``home_and_clear`` sets ``_state``/``_budget``/
    ``_finished`` and returns the advanced state."""

    def __init__(
        self,
        executor,
        belief,
        *,
        clear_margin_m: float = 14.0,
        prune_radius_m: float = 30.0,
        orbit_radius_m: float = 85.0,
        orbit_delta_deg: float = 42.0,
        reliable_region_m: float = 300.0,
        baseline_fwd_m: float = 420.0,
        baseline_perp_m: float = 320.0,
        max_homing_iters: int = 8,
    ) -> None:
        self.exec = executor
        self.belief = belief
        self.clear_margin_m = float(clear_margin_m)
        self.prune_radius_m = float(prune_radius_m)
        self.orbit_radius_m = float(orbit_radius_m)
        self.orbit_delta_deg = float(orbit_delta_deg)
        self.reliable_region_m = float(reliable_region_m)
        self.baseline_fwd_m = float(baseline_fwd_m)
        self.baseline_perp_m = float(baseline_perp_m)
        self.max_homing_iters = int(max_homing_iters)
        # call-scoped mutable state (set per home_and_clear)
        self._state = None
        self._budget = float("inf")
        self._finished = False

    # -- public -------------------------------------------------------------

    def home_and_clear(self, channel: int, state, time_budget_s: float = float("inf")):
        """Home channel ``channel`` from ``state`` and blind-clear it.

        Returns ``(state, cleared, finished)``: the advanced robot state, whether the
        source was neutralised, and whether the env signalled finish (deadline)."""
        self._state = state
        self._budget = float(time_budget_s)
        self._finished = False
        fix = _Fix(obs=[(b.point, b.svd_deg) for b in self.belief[channel].bearings])
        fix.refresh()
        cleared = self._home_and_clear(channel, fix)
        return self._state, cleared, self._finished

    # -- Way3 homing loop (ported verbatim in structure) --------------------

    def _home_and_clear(self, c: int, fix: _Fix) -> bool:
        R = self.orbit_radius_m
        for _ in range(self.max_homing_iters):
            if self._finished:
                return False
            fix.refresh()
            reliable = fix.est is not None and fix.region_r <= self.reliable_region_m
            # 1) tight enough -> blind-clear at the region centre.
            if reliable and fix.region_r <= self.clear_margin_m:
                if self._try_clear(c, fix.est):
                    return True
            # 2) no est / unreliable (near-parallel -> far phantom crossing) ->
            #    build a perpendicular baseline from the single bearing.
            if not reliable:
                if not fix.obs:
                    return False
                if self._establish_baseline(c, fix):
                    return True
                fix.refresh()
                if not (fix.est is not None and fix.region_r <= self.reliable_region_m):
                    self._forward_step(c, fix)      # close in, rebuild next round
                continue
            # 3) reliable but not tight -> orbit the est, shrinking R each round.
            if self._orbit_triangulate(c, fix, R):
                return True
            R = max(35.0, R * 0.6)
        # fallback: clear only if a tight-enough estimate exists (no phantom clears).
        fix.refresh()
        if fix.est is not None and fix.region_r <= self.prune_radius_m:
            return self._try_clear(c, fix.est)
        # P4 deterministic rescue: after the normal orbit has stalled, sweep a
        # narrow zig-zag around the last *observed* bearing.  This is deliberately
        # a sensing fallback, not a certificate: only a new bearing can update
        # the fix, and clearing remains guarded by the MEC threshold below.
        if self._ray_sweep_rescue(c, fix):
            return True
        if fix.est is not None and fix.region_r <= 100.0:
            if self._triangular_clear_rescue(c, fix):
                return True
        return False

    def _ray_sweep_rescue(self, c: int, fix: _Fix) -> bool:
        """Probe a last reliable bearing with a bounded transverse sweep.

        The sweep is useful for a directional source whose later orbit points
        fall in its blind half-plane.  It never subtracts a range disc from a
        directional NO_SIGNAL and never clears from geometry of the path alone.
        """
        if not fix.obs:
            return False
        origin, bearing = fix.obs[-1]
        length = min(1500.0, max(0.0, self.reliable_region_m * 2.0))
        for point in ray_sweep_points(origin, bearing, length, step_m=15.0):
            if self._finished:
                return False
            result = self._measure_into(c, fix, point)
            if result == "cleared":
                return True
            fix.refresh()
            if fix.est is not None and fix.region_r <= self.clear_margin_m:
                if self._try_clear(c, fix.est):
                    return True
        return False

    def _triangular_clear_rescue(self, c: int, fix: _Fix) -> bool:
        """Attempt real clears on a 33 m local triangular sweep."""
        for point in triangular_clear_sweep_points(fix.est, radius_m=50.0, spacing_m=33.0):
            if self._finished:
                return False
            if self._try_clear(c, point):
                return True
        return False

    def _establish_baseline(self, c: int, fix: _Fix) -> bool:
        """From one bearing (p,b): take a *perpendicular* offset to get a second,
        crossable bearing without risking overshoot past the source into its blind
        back (禁止/seed4068-ch15 lesson). Perpendicular-first, then forward+perp only
        when hugging the r_eff edge. Returns whether cleared on the spot (near hit)."""
        p, b = fix.obs[-1]
        ux, uy = _unit(b)
        nx, ny = -uy, ux                            # perpendicular unit
        L = self.baseline_perp_m
        plans = [(0.0, L), (0.0, 0.62 * L), (0.5 * self.baseline_fwd_m, L)]
        for fwd, off in plans:
            cands = [
                (p[0] + fwd * ux + sg * off * nx, p[1] + fwd * uy + sg * off * ny)
                for sg in (1.0, -1.0)
            ]
            cands.sort(key=lambda q: q[0] * q[0] + q[1] * q[1])   # nearer centre first
            for q in cands:
                if self._finished:
                    return False
                r = self._measure_into(c, fix, q)
                if r == "cleared":
                    return True
                if r == "obs":
                    fix.refresh()
                    if fix.est is not None and fix.region_r <= self.clear_margin_m:
                        if self._try_clear(c, fix.est):
                            return True
                    if fix.est is not None and fix.region_r <= self.reliable_region_m:
                        return False              # reliable crossing -> hand to orbit
        return False

    def _orbit_triangulate(self, c: int, fix: _Fix, R: float) -> bool:
        """Two points around ``est`` toward the circular mean of the directions signal
        was received (guaranteed in-arc; seed815-ch17 lesson — never toward the robot's
        current heading, which may sit in the blind zone after the tour)."""
        est = fix.est
        if fix.obs:
            cx = sum(math.cos(math.radians(bearing_deg(est, p))) for p, _ in fix.obs)
            cy = sum(math.sin(math.radians(bearing_deg(est, p))) for p, _ in fix.obs)
            alpha = (
                math.degrees(math.atan2(cy, cx))
                if (cx or cy)
                else bearing_deg(est, self._state.pos)
            )
        else:
            alpha = bearing_deg(est, self._state.pos)
        for delta in (self.orbit_delta_deg, -self.orbit_delta_deg):
            if self._finished:
                return False
            a = math.radians(alpha + delta)
            wp = (est[0] + R * math.cos(a), est[1] + R * math.sin(a))
            r = self._measure_into(c, fix, wp)
            if r == "cleared":
                return True
            fix.refresh()
            if fix.est is not None and fix.region_r <= self.clear_margin_m:
                if self._try_clear(c, fix.est):
                    return True
        return False

    def _forward_step(self, c: int, fix: _Fix) -> None:
        """Step toward the source along the latest bearing (shrinks range) to set up a
        better baseline next round."""
        if not fix.obs:
            return
        p, b = fix.obs[-1]
        ux, uy = _unit(b)
        self._measure_into(
            c, fix, (p[0] + self.baseline_fwd_m * ux, p[1] + self.baseline_fwd_m * uy)
        )

    # -- env stepping (folds through the executor) --------------------------

    def _measure_into(self, c: int, fix: _Fix, pt: Point) -> Optional[str]:
        """Measure c at pt: near -> clear on the spot ('cleared'); bearing -> record
        ('obs'); no_signal / deadline -> None."""
        if self._finished:
            return None
        self._state, obs, finished = self.exec.step_measure(self._state, pt, c, self._budget)
        if finished:
            self._finished = True
        if obs is None:
            return None
        if obs.is_near:
            return "cleared" if self._try_clear(c, self._state.pos) else None
        if obs.is_bearing:
            fix.obs.append((pt, float(obs.svd_deg)))
            fix.refresh()
            return "obs"
        return None                                  # no_signal

    def _try_clear(self, c: int, target: Point) -> bool:
        """Blind-clear c at ``target``; folds mark_cleared on a real env hit."""
        if self._finished:
            return False
        self._state, hit, finished = self.exec.step_clear(self._state, target, c, self._budget)
        if finished:
            self._finished = True
        return bool(hit)
