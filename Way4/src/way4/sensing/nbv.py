"""Minimax next-best-view for bearing refinement (DESIGN.md §8, M5).

Way3 (and Bishop) fix a 90° triangulation angle. Way4 instead follows Tokekar's
*robust* active localization: pick the next viewpoint ``q`` that minimises the
WORST-CASE size of the still-feasible region over a small set of representative
source hypotheses drawn from the current ``F_c``:

    U(q)   = max_{p_j ∈ P} diam( F_c ∩ W(q, θ_j, ±1°) ∩ B(q, 1500) )   θ_j = bearing(q→p_j)
    NBV(q) = U(q) + λ_t · T(q)                                          (T = travel+measure time)
    q*     = argmin_q NBV(q)

Two soundness points that keep this a faithful *heuristic* (never a certificate):

  * ``F_c`` and every clip use the same convex SUPERSET geometry as the belief
    layer, so the region we score is an outer bound — we never over-claim shrink.
  * **Out-of-range guard (guaranteed range).** A candidate ``q`` so far that a
    hypothesis ``p_j`` sits beyond the *guaranteed* detection radius ``R_lo=1000``
    (the R_eff LOWER bound, §6.1) is treated as NO_SIGNAL — no bearing sliver, no
    shrink — and scored as the *full current diameter* (no information). Using the
    lower bound, not the optimistic 1500 upper bound, is the crux: a viewpoint only
    "counts on" a shrink where a detection is CERTAIN. Scoring far viewpoints on the
    1500 upper bound is exactly what lured the robot to a standoff whose true R_eff
    (e.g. 1022) returned only NO_SIGNAL — a zero-progress re-scan that, being a
    no-move, then won on cost forever (the seed-1017 REFINE livelock). The range-disc
    *clip* still uses the sound 1500 superset, matching the belief layer's update.
  * **Get-in-range standpoints + homing.** The region centroid and MEC centre are in
    the pool: standing at the MEC centre puts the true source (∈ F_c) within
    ``mec_radius`` — a GUARANTEED detection whenever ``mec_radius ≤ R_lo``. When the
    region is too large for any single guaranteed-detection viewpoint (all score the
    full diameter), a small approach term breaks the tie toward the viewpoint that
    minimises the worst-case distance to the region, homing in until in range.
  * **No free re-scan.** Viewpoints coinciding with a point already scanned for the
    channel (its bearings + NO_SIGNAL disc centres) are dropped: re-measuring there
    is deterministic and yields no new bearing information (Invariant-safe: this only
    prunes the *proposal* pool, it never marks or certifies).

This module only proposes REFINE viewpoints for a DETECTED channel; it marks
nothing and clears nothing (禁止6/7). Robot viewpoints are NOT clamped to the
arena (源/机器人域分离, §7): an outside standoff is often the best crossing point
for a near-boundary source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sin
from typing import List, Optional, Sequence, Tuple

from sxjm_core.geometry import (
    Point,
    angle_sep_deg,
    bearing_deg,
    disc_superset_halfplanes,
    dist,
    halfplane_intersection,
    polygon_centroid,
    polygon_diameter,
    wedge_halfplanes,
)

# Type alias only; the belief channel is duck-typed (F_c / bearings / mec_*).


@dataclass
class NBVResult:
    """The chosen refinement viewpoint and why."""

    point: Point
    worst_case_diameter: float          # U(q*) — the minimax objective at the winner
    current_diameter: float             # diam(F_c) before the measurement
    score: float                        # U(q*) + λ_t T(q*)
    hypotheses: List[Point] = field(default_factory=list)
    n_candidates: int = 0
    improved: bool = False              # worst_case_diameter < current_diameter

    @property
    def expected_shrink(self) -> float:
        """A non-negative worst-case diameter reduction the viewpoint guarantees."""
        return max(0.0, self.current_diameter - self.worst_case_diameter)


class MinimaxNBV:
    """Robust NBV planner (Tokekar minimax) over an analytic candidate pool."""

    def __init__(
        self,
        range_radius: float = 1500.0,
        detect_lower_bound: float = 1000.0,
        wedge_half_deg: float = 1.0,
        eps_num_deg: float = 1e-6,
        speed: float = 5.0,
        measure_s: float = 5.0,
        lambda_t: float = 0.0,
        approach_weight: float = 1e-2,
        exclude_scanned_eps: float = 1.0,
        n_hypotheses: int = 8,
        ring_radii: Sequence[float] = (400.0, 700.0, 1000.0, 1300.0),
        n_ring_angles: int = 16,
        perp_standoffs: Sequence[float] = (400.0, 800.0, 1200.0),
        disc_sides: int = 16,
        max_robot_coord: float = 2.0e6,   # robot domain bound (题面: |coord| ≤ 2e6)
    ) -> None:
        self.range_radius = float(range_radius)
        self.detect_lower_bound = float(detect_lower_bound)
        self.wedge_half = float(wedge_half_deg) + float(eps_num_deg)
        self.speed = float(speed)
        self.measure_s = float(measure_s)
        self.lambda_t = float(lambda_t)
        self.approach_weight = float(approach_weight)
        self.exclude_scanned_eps = float(exclude_scanned_eps)
        self.n_hypotheses = int(n_hypotheses)
        self.ring_radii = tuple(float(r) for r in ring_radii)
        self.n_ring_angles = int(n_ring_angles)
        self.perp_standoffs = tuple(float(s) for s in perp_standoffs)
        self.disc_sides = int(disc_sides)
        self.max_robot_coord = float(max_robot_coord)

    # -- representative hypotheses P (≤ n_hypotheses) ----------------------

    def representative_hypotheses(self, poly: Sequence[Point]) -> List[Point]:
        """A small, spread cover of ``F_c`` (§8): centroid + the diameter-realising
        pair (long-axis endpoints) + evenly-subsampled vertices, capped at
        ``n_hypotheses``. Duplicates removed while preserving priority order."""
        pts = [(float(x), float(y)) for x, y in poly]
        if not pts:
            return []
        chosen: List[Point] = [polygon_centroid(pts)]
        a, b = _diameter_pair(pts)
        if a is not None:
            chosen.append(a)
            chosen.append(b)
        # subsample the remaining vertices evenly around the boundary
        budget = self.n_hypotheses - len(chosen)
        if budget > 0 and pts:
            step = max(1, len(pts) // budget)
            chosen.extend(pts[::step][:budget])
        # dedup, keep order, cap
        out: List[Point] = []
        seen = set()
        for p in chosen:
            key = (round(p[0], 6), round(p[1], 6))
            if key not in seen:
                seen.add(key)
                out.append(p)
            if len(out) >= self.n_hypotheses:
                break
        return out

    # -- candidate viewpoints ----------------------------------------------

    def candidate_viewpoints(self, channel, robot_pos: Point) -> List[Point]:
        """Analytic pool: a ring about the region centroid (Tokekar sampling) plus
        perpendicular-baseline standoffs off each existing bearing (Bishop 90° /
        Way3 vertical baseline). Deduplicated; coords bounded to the robot domain."""
        poly = channel.F_c or []
        center = polygon_centroid(poly) if poly else (channel.mec_center or robot_pos)
        cands: List[Point] = []

        # 0) get-in-range standpoints: the region centroid and MEC centre. Standing at
        #    the MEC centre, the source (∈ F_c) is within mec_radius — a guaranteed
        #    detection when mec_radius ≤ R_lo, i.e. the cheapest way back into range for
        #    a large, poorly-crossed region (avoids the far-standoff NO_SIGNAL trap).
        cands.append((center[0], center[1]))
        mc = getattr(channel, "mec_center", None)
        if mc is not None:
            cands.append((float(mc[0]), float(mc[1])))

        # 1) ring about the region centre (broad Tokekar coverage)
        for r in self.ring_radii:
            for k in range(self.n_ring_angles):
                ang = radians(360.0 * k / self.n_ring_angles)
                cands.append((center[0] + r * cos(ang), center[1] + r * sin(ang)))

        # 2) perpendicular baselines off each existing bearing (best crossing)
        for bobs in getattr(channel, "bearings", []) or []:
            los = radians(bobs.svd_deg)          # first line-of-sight direction
            px, py = -sin(los), cos(los)          # unit ⟂ to the LOS
            for s in self.perp_standoffs:
                cands.append((center[0] + px * s, center[1] + py * s))
                cands.append((center[0] - px * s, center[1] - py * s))

        # 3) shape-adaptive stand-offs.  For a thin feasible set, a viewpoint
        # displaced across the principal axis is more informative than another
        # point along its long direction.  For round sets use the ordinary ring
        # pool above; this branch only changes the candidate set, never the
        # sound outer belief or the certificate.
        axis = getattr(channel, "principal_axis", None)
        kappa = float(getattr(channel, "kappa", 1.0))
        if axis is not None and kappa >= 8.0:
            ax, ay = float(axis[0]), float(axis[1])
            nx, ny = -ay, ax
            for s in self.perp_standoffs:
                cands.append((center[0] + nx * s, center[1] + ny * s))
                cands.append((center[0] - nx * s, center[1] - ny * s))

        # bound to the robot domain, dedup
        out: List[Point] = []
        seen = set()
        for (x, y) in cands:
            x = max(-self.max_robot_coord, min(self.max_robot_coord, x))
            y = max(-self.max_robot_coord, min(self.max_robot_coord, y))
            key = (round(x, 3), round(y, 3))
            if key not in seen:
                seen.add(key)
                out.append((x, y))
        return out

    # -- objective ----------------------------------------------------------

    def post_measurement_diameter(
        self, poly: Sequence[Point], q: Point, hypothesis: Point, current_diameter: float
    ) -> float:
        """Diameter of ``F_c`` after a bearing from ``q`` toward ``hypothesis``.

        Guaranteed in range (``dist ≤ R_lo``) → clip by the ±1° wedge and the sound
        range-disc superset (radius 1500, matching belief). Beyond the guaranteed
        radius → treated as a possible NO_SIGNAL, no bearing sliver, so the diameter is
        unchanged (the robust guard — see module docstring)."""
        if dist(q, hypothesis) > self.detect_lower_bound:
            return current_diameter
        theta = bearing_deg(q, hypothesis)
        hps = list(wedge_halfplanes(q, theta, self.wedge_half))
        hps.extend(disc_superset_halfplanes(q, self.range_radius, n_sides=self.disc_sides))
        region = halfplane_intersection(hps, poly)
        return polygon_diameter(region) if region else 0.0

    def worst_case_diameter(
        self, poly: Sequence[Point], q: Point, hypotheses: Sequence[Point], current_diameter: float
    ) -> float:
        """``U(q)`` — the minimax objective (max over hypotheses)."""
        worst = 0.0
        for p in hypotheses:
            d = self.post_measurement_diameter(poly, q, p, current_diameter)
            if d > worst:
                worst = d
        return worst

    def travel_time(self, robot_pos: Point, q: Point) -> float:
        return dist(robot_pos, q) / self.speed + self.measure_s

    # -- top-level choice ---------------------------------------------------

    def choose(self, channel, robot_pos: Point) -> Optional[NBVResult]:
        """Pick the robust next viewpoint for a DETECTED channel. Returns ``None``
        when there is nothing to refine (no polygon / already a point) or every
        candidate coincides with an already-scanned point."""
        poly = channel.F_c
        if not poly or len(poly) < 3:
            return None
        current = polygon_diameter(poly)
        effective_sampler = getattr(channel, "sample_effective_hypotheses", None)
        hyps = effective_sampler(limit=self.n_hypotheses, spacing=60.0) if effective_sampler else []
        # Keep the analytic polygon representative fallback for legacy duck-typed
        # beliefs; native ChannelBelief uses effective hypotheses above.
        if not hyps:
            hyps = self.representative_hypotheses(poly)
        if not hyps:
            return None
        cands = self.candidate_viewpoints(channel, robot_pos)
        if not cands:
            return None

        # No free re-scan: a viewpoint at a point already scanned for this channel
        # (a prior bearing point or a NO_SIGNAL disc centre) re-measures the same
        # deterministic outcome and yields no new bearing — drop it, so a stalled
        # region cannot re-pick the same zero-progress standoff for free (§8).
        scanned: List[Point] = [
            (float(b.point[0]), float(b.point[1])) for b in getattr(channel, "bearings", []) or []
        ]
        scanned.extend(
            (float(d.center[0]), float(d.center[1]))
            for d in getattr(channel, "negative_discs", []) or []
        )
        eps = self.exclude_scanned_eps

        best: Optional[NBVResult] = None
        best_key = None
        for q in cands:
            if scanned and any(dist(q, s) <= eps for s in scanned):
                continue
            u = self.worst_case_diameter(poly, q, hyps, current)
            # Approach term: worst-case distance from q to the region. It only breaks
            # ties in u — when no viewpoint can guarantee a shrink (a region too large
            # for the guaranteed radius, so every u == current), it homes the robot to
            # the point covering the region best, i.e. back into detection range.
            reach = max(dist(q, p) for p in hyps)
            score = u + self.lambda_t * self.travel_time(robot_pos, q) + self.approach_weight * reach
            key = (round(score, 6), round(u, 6))
            if best is None or key < best_key:
                best_key = key
                best = NBVResult(
                    point=q,
                    worst_case_diameter=u,
                    current_diameter=current,
                    score=score,
                    hypotheses=list(hyps),
                    n_candidates=len(cands),
                    improved=u < current - 1e-9,
                )
        return best

    def completion_point(self, channel, robot_pos: Point, kill_radius: float = 20.0) -> Optional[Point]:
        """Return a candidate whose robust post-measurement diameter fits kill radius."""
        poly = channel.F_c
        if not poly:
            return None
        current = polygon_diameter(poly)
        hyps = channel.sample_effective_hypotheses(self.n_hypotheses, 60.0) \
            if hasattr(channel, "sample_effective_hypotheses") else self.representative_hypotheses(poly)
        best = None
        for q in self.candidate_viewpoints(channel, robot_pos):
            if any(dist(q, s) <= self.exclude_scanned_eps for s in
                   [(b.point[0], b.point[1]) for b in getattr(channel, "bearings", [])]):
                continue
            post = self.worst_case_diameter(poly, q, hyps, current)
            key = (post, self.travel_time(robot_pos, q))
            if best is None or key < best[0]:
                best = (key, q)
        return None if best is None or best[0][0] > 2.0 * kill_radius else best[1]

    def verification_point(self, channel, robot_pos: Point, coverage_gain=None) -> Optional[Point]:
        """Choose the candidate maximizing NO_SIGNAL coverage/certificate gain."""
        cands = self.candidate_viewpoints(channel, robot_pos)
        holes = getattr(channel, "negative_discs", [])
        if not cands:
            return None
        if coverage_gain is None:
            # A geometry-only fallback: prefer the point nearest the effective set.
            return min(cands, key=lambda q: min((dist(q, d.center) for d in holes), default=0.0))
        scored = [(float(coverage_gain(q)), self.travel_time(robot_pos, q), q) for q in cands]
        return max(scored, key=lambda x: (x[0], -x[1]))[2]

    def initialization_point(self, channel, robot_pos: Point) -> Optional[Point]:
        """Choose a point maximizing bearing crossing quality for INITIALIZED state."""
        cands = self.candidate_viewpoints(channel, robot_pos)
        bearings = getattr(channel, "bearings", []) or []
        if not cands or not bearings:
            return None
        scored = []
        for q in cands:
            if any(dist(q, b.point) <= self.exclude_scanned_eps for b in bearings):
                continue
            from math import sin, radians
            los = bearing_deg(q, channel.mec_center or q)
            crossing = max(abs(sin(radians(angle_sep_deg(b.svd_deg, los)))) for b in bearings)
            scored.append((crossing, -self.travel_time(robot_pos, q), q))
        return max(scored)[2] if scored else None


# --- helpers -----------------------------------------------------------------


def _diameter_pair(pts: Sequence[Point]) -> Tuple[Optional[Point], Optional[Point]]:
    """The vertex pair realising the polygon diameter (long-axis endpoints)."""
    n = len(pts)
    if n < 2:
        return None, None
    best = -1.0
    bi, bj = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(pts[i], pts[j])
            if d > best:
                best, bi, bj = d, i, j
    return pts[bi], pts[bj]
