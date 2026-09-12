"""Receding-horizon planner (DESIGN.md §10).

The rolling loop is *Observe → Update → Plan → Execute one*. This module is the
Plan step: given the macro candidates for the current belief, choose the one that
minimises immediate cost **plus** expected cost-to-go under the outcomes it could
produce:

    Q(a) = C(a) + f_o[ Ĵ(B' | a, o) ]
    a*   = argmin_a Q(a)

``f_o`` is **robust/minimax** by default (``max`` over a small representative
outcome set ``O``, ≤3 per action here), or **expected** (probability-weighted).
Costs are analytical proxies (``FutureCostEstimator``) and routing is memoised, so
a replan is cheap enough for the ≤20-min real-time budget (§10 实时预算).

V1 is ``horizon=1`` (route-aware one step). V2 is ``horizon=2`` with a beam: the
terminal ``Ĵ`` is refined by one greedy proxy step from the predicted view, over
the ``beam_width`` cheapest continuations. Deeper planning only sharpens the
*ranking* — the full-clear guarantee is the SafetyShield's job (§11), never this
proxy's, so an approximate look-ahead can cost time but never correctness.

The planner never marks absent and never clears without the guard (禁止6/7): it only
ranks the candidates the generator already vetted; execution and safety come after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from sxjm_core.geometry import dist

from ..belief import ChannelStatus
from ..core import MacroActionType, MacroCandidate
from .future_cost import CostView, DetectedRegion, FutureCostEstimator, build_cost_view

Point = Tuple[float, float]


@dataclass
class Outcome:
    label: str
    prob: float
    view: CostView


@dataclass
class CandidateEvaluation:
    candidate: MacroCandidate
    immediate_cost: float
    future_cost: float                       # aggregated Ĵ over outcomes
    q_value: float
    outcomes: List[Tuple[str, float]] = field(default_factory=list)   # (label, Ĵ)


@dataclass
class PlanResult:
    best: Optional[MacroCandidate]
    q_value: float
    evaluations: List[CandidateEvaluation]
    horizon: int
    outcome_mode: str


class OutcomePredictor:
    """Predicts a small set of representative post-action ``CostView``s (§10, ``O``
    limited to ≤3 outcomes). Perturbations are analytical: a NO_SIGNAL scan erases
    the holes it would cover; a detection turns an UNKNOWN channel into a nominal
    DETECTED region (or shrinks an existing one by the NBV expected shrink); a
    ``near`` makes the focus channel clearable on the spot. It clones only the cheap
    ``CostView`` — never the belief or the numpy coverage masks."""

    def __init__(
        self,
        detection_radius: float = 1000.0,
        range_radius: float = 1500.0,
        clear_threshold: float = 18.0,
        # outcome likelihoods: a DETECTED source (confirmed present) almost surely
        # responds; an UNKNOWN channel usually returns no signal.
        p_detected_channel=(("detect", 0.85), ("near", 0.10), ("no_signal", 0.05)),
        p_unknown_channel=(("no_signal", 0.70), ("detect", 0.25), ("near", 0.05)),
    ) -> None:
        self.detection_radius = float(detection_radius)
        self.range_radius = float(range_radius)
        self.clear_threshold = float(clear_threshold)
        self.p_detected_channel = p_detected_channel
        self.p_unknown_channel = p_unknown_channel

    def geometric_features(self, belief_region, point, bearing_deg, *, half_width=1.0):
        """Return geometry-derived post-bearing features without fixed outcome
        probabilities.  The caller supplies the current feasible region and may
        evaluate the returned wedges with the project's MEC implementation."""
        from math import radians, cos, sin
        q = (float(point[0]), float(point[1]))
        theta = radians(float(bearing_deg))
        return {
            "point": q,
            "bearing_deg": float(bearing_deg),
            "bearing_unit": (cos(theta), sin(theta)),
            "half_width_deg": float(half_width),
            "region_size_before": len(belief_region) if hasattr(belief_region, "__len__") else None,
        }

    def sampled_geometry_outcomes(self, region_points, point, bearings=(-1.0, 0.0, 1.0)):
        """Build outcome features from source samples, rather than fixed priors."""
        return [self.geometric_features(region_points, point, b) for b in bearings]

    def predict(self, candidate: MacroCandidate, belief, base: CostView) -> List[Outcome]:
        at = candidate.action_type
        if at == MacroActionType.EXIT:
            return [Outcome("exit", 1.0, base.copy())]
        if at == MacroActionType.CLEAR:
            return [Outcome("hit", 1.0, self._after_clear(candidate, base))]
        if at == MacroActionType.STOP:
            # P0 #2/#3 spatial bundle: a location-first visit that does a batch scan
            # AND any folded clears in one stop. Its effect on the remaining-work view
            # is the union of what its parts would do — nominally (the batch covers its
            # neighbourhood, the folded clears remove their targets). Deterministic like
            # CLEAR: this only ranks the bundle; correctness stays the guards' job.
            return [Outcome("stop", 1.0, self._after_stop(candidate, base))]
        return self._after_scan(candidate, belief, base)

    # -- STOP (spatial bundle) ---------------------------------------------

    def _after_stop(self, candidate: MacroCandidate, base: CostView) -> CostView:
        v = base.copy()
        q = (float(candidate.target[0]), float(candidate.target[1]))
        v.pos = q
        if candidate.scan_channels:
            # the batch covers the neighbourhood around q (as a NO_SIGNAL scan would)
            v.unknown_holes = [h for h in v.unknown_holes if dist(h, q) > self.detection_radius]
            if candidate.refinement_gain > 0 and v.detected:
                v.detected = self._shrink_one(v.detected, q, float(candidate.refinement_gain))
        # each folded clear removes its own clearable target (as CLEAR would)
        for tgt in getattr(candidate, "clear_targets", ()):
            v.clearable_targets = _remove_nearest(v.clearable_targets, tgt)
        return v

    # -- CLEAR --------------------------------------------------------------

    def _after_clear(self, candidate: MacroCandidate, base: CostView) -> CostView:
        v = base.copy()
        v.pos = (float(candidate.target[0]), float(candidate.target[1]))
        v.clearable_targets = _remove_nearest(v.clearable_targets, candidate.target)
        return v

    # -- scans (EXPLORE / VERIFY / REFINE / INITIALIZE / PURSUE) ------------

    def _after_scan(self, candidate: MacroCandidate, belief, base: CostView) -> List[Outcome]:
        q = (float(candidate.target[0]), float(candidate.target[1]))
        focus = self._focus_channel(candidate)
        focus_status = belief[focus].status if focus is not None else ChannelStatus.UNKNOWN
        table = (
            self.p_detected_channel
            if focus_status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED)
            else self.p_unknown_channel
        )
        outs: List[Outcome] = []
        for label, prob in table:
            if label == "no_signal":
                outs.append(Outcome(label, prob, self._view_no_signal(candidate, q, base)))
            elif label == "detect":
                outs.append(Outcome(label, prob, self._view_detect(candidate, focus, focus_status, q, base)))
            elif label == "near":
                outs.append(Outcome(label, prob, self._view_near(focus, focus_status, q, base)))
        return outs

    def _view_no_signal(self, candidate: MacroCandidate, q: Point, base: CostView) -> CostView:
        v = base.copy()
        v.pos = q
        # the batch covers the UNKNOWN band near q -> erase holes within range
        v.unknown_holes = [h for h in v.unknown_holes if dist(h, q) > self.detection_radius]
        return v

    def _view_detect(self, candidate, focus, focus_status, q: Point, base: CostView) -> CostView:
        v = base.copy()
        v.pos = q
        if focus_status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED):
            # refine: shrink the matching region by the NBV expected shrink
            shrink = float(candidate.refinement_gain)
            v.detected = self._shrink_one(v.detected, q, shrink)
        else:
            # first detection on an UNKNOWN channel: a nominal fresh region near q
            v.n_unknown = max(0, v.n_unknown - 1)
            v.detected = list(v.detected) + [
                DetectedRegion(q, self.range_radius / 2.0, self.range_radius, 0.3)
            ]
        return v

    def _view_near(self, focus, focus_status, q: Point, base: CostView) -> CostView:
        v = base.copy()
        v.pos = q
        v.clearable_targets = list(v.clearable_targets) + [q]
        if focus_status == ChannelStatus.UNKNOWN:
            v.n_unknown = max(0, v.n_unknown - 1)
        elif focus_status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED):
            v.detected = _remove_nearest_region(v.detected, q)
        return v

    def _shrink_one(self, regions: List[DetectedRegion], q: Point, shrink: float) -> List[DetectedRegion]:
        """Shrink the region nearest ``q`` by ``shrink`` (diameter); if it drops to
        clearable it will be picked up as a clear next step (kept as a small region
        here — the safety layer owns the clearable transition)."""
        if not regions:
            return regions
        out = list(regions)
        idx = min(range(len(out)), key=lambda i: dist(out[i].center, q))
        d = out[idx]
        new_diam = max(0.0, d.diameter - max(0.0, shrink))
        scale = (new_diam / d.diameter) if d.diameter > 1e-9 else 0.0
        out[idx] = DetectedRegion(d.center, d.r_mec * scale, new_diam, min(1.0, d.sin_gamma + 0.3))
        return out

    def _focus_channel(self, candidate: MacroCandidate) -> Optional[int]:
        if "refine_channel" in candidate.meta:
            return int(candidate.meta["refine_channel"])
        if candidate.clear_channel is not None:
            return int(candidate.clear_channel)
        if candidate.scan_channels:
            return int(candidate.scan_channels[0])
        return None


class RecedingHorizonPlanner:
    def __init__(
        self,
        future_cost: Optional[FutureCostEstimator] = None,
        predictor: Optional[OutcomePredictor] = None,
        horizon: int = 1,
        beam_width: int = 8,
        outcome_mode: str = "robust",     # "robust" (minimax) | "expected"
        speed: float = 5.0,
        measure_s: float = 5.0,
        clear_hit_s: float = 5.0,
        voi_weight: float = 0.0,
    ) -> None:
        self.fce = future_cost or FutureCostEstimator(
            speed=speed, measure_s=measure_s, clear_hit_s=clear_hit_s
        )
        self.predictor = predictor or OutcomePredictor(
            detection_radius=self.fce.detection_radius
        )
        self.horizon = int(horizon)
        self.beam_width = int(beam_width)
        self.outcome_mode = outcome_mode
        self.speed = float(speed)
        self.measure_s = float(measure_s)
        self.clear_hit_s = float(clear_hit_s)
        self.voi_weight = float(voi_weight)

    # -- public -------------------------------------------------------------

    def plan(self, belief, certificate, state, candidates: Sequence[MacroCandidate]) -> PlanResult:
        base = build_cost_view(belief, certificate, state)
        evals: List[CandidateEvaluation] = []
        for a in candidates:
            outs = self.predictor.predict(a, belief, base)
            jvals = [(o.label, o.prob, self._cost_to_go(o.view, self.horizon - 1)) for o in outs]
            jagg = self._aggregate(jvals)
            c = float(a.expected_time)
            evals.append(
                CandidateEvaluation(a, c, jagg, c + jagg, [(lbl, j) for lbl, _, j in jvals])
            )
        if evals and self.voi_weight > 0.0:
            # VOI is a secondary task-time signal. Safety remains unchanged:
            # candidates were generated and masked by the mathematical planner.
            best = min(
                evals,
                key=lambda e: e.q_value - self.voi_weight * self.value_of_information(
                    e.candidate, belief, certificate, state
                ),
            )
        else:
            best = min(evals, key=lambda e: e.q_value) if evals else None
        return PlanResult(
            best.candidate if best else None,
            best.q_value if best else 0.0,
            evals,
            self.horizon,
            self.outcome_mode,
        )

    def value_of_information(self, candidate: MacroCandidate, belief, certificate, state) -> float:
        """Expected task-time reduction per second spent on an option."""
        base = build_cost_view(belief, certificate, state)
        before = self.fce.estimate(base).total
        outcomes = self.predictor.predict(candidate, belief, base)
        if not outcomes or candidate.expected_time <= 0.0:
            return 0.0
        after = sum(o.prob * self._cost_to_go(o.view, max(0, self.horizon - 1))
                    for o in outcomes)
        return max(0.0, before - after) / float(candidate.expected_time)

    def sensing_trigger(self, candidate: MacroCandidate, belief, certificate, state,
                        threshold: float = 0.0) -> bool:
        """Decide whether a waypoint's sensing value pays its incremental cost.

        ``candidate.expected_time`` already includes route travel, detection and
        switching for the proposed batch.  This keeps the trigger on mission-time
        units and avoids the unsafe shortcut of measuring every route waypoint.
        """
        if candidate.action_type in (MacroActionType.CLEAR, MacroActionType.EXIT):
            return False
        return self.value_of_information(candidate, belief, certificate, state) > float(threshold)

    def should_replan(self, event, *, marginal_value_drop: float = 0.0,
                      new_candidate_better: bool = False) -> bool:
        """Event-driven global-replan policy.

        Strategic lifecycle/cardinality/certificate events force a replan;
        ordinary batch completion only does so when its marginal value collapsed
        or a materially better candidate appeared.
        """
        name = getattr(getattr(event, "event_type", event), "value", event)
        strategic = {
            "POSITIVE_DISCOVERY", "INITIALIZED", "LOCALIZED",
            "SOURCE_CLEARED", "CHANNEL_CERTIFIED_EMPTY", "CARDINALITY_CLOSURE",
            "COVERAGE_THRESHOLD", "TIME_BUDGET_WARNING",
        }
        if name in strategic:
            return True
        if name == "BATCH_COMPLETED":
            return float(marginal_value_drop) > 0.5 or bool(new_candidate_better)
        return False

    # -- aggregation over outcomes -----------------------------------------

    def _aggregate(self, jvals: Sequence[Tuple[str, float, float]]) -> float:
        if not jvals:
            return 0.0
        if self.outcome_mode == "expected":
            wsum = sum(p for _, p, _ in jvals)
            if wsum <= 0:
                return max(j for _, _, j in jvals)
            return sum(p * j for _, p, j in jvals) / wsum
        return max(j for _, _, j in jvals)      # robust / minimax (default)

    # -- cost-to-go (horizon recursion) ------------------------------------

    def _cost_to_go(self, view: CostView, depth: int) -> float:
        base = self.fce.estimate(view).total
        if depth <= 0:
            return base
        # V2: refine the terminal estimate by one greedy proxy step (beam-limited).
        proxies = self._proxy_steps(view)
        if not proxies:
            return base
        proxies.sort(key=lambda cv: cv[0])            # cheapest immediate first
        best2 = min(
            c2 + self._cost_to_go(v2, depth - 1) for c2, v2 in proxies[: self.beam_width]
        )
        return best2

    def _proxy_steps(self, view: CostView) -> List[Tuple[float, CostView]]:
        """Coarse next-step continuations from a predicted view — no belief, no NBV,
        no quadtree (§10 analytical only). Each removes the work it addresses so the
        rollout estimate is monotone."""
        out: List[Tuple[float, CostView]] = []

        # clear one already-clearable target
        for t in view.clearable_targets:
            v = view.copy()
            v.pos = t
            v.clearable_targets = _remove_nearest(v.clearable_targets, t)
            c = dist(view.pos, t) / self.speed + self.clear_hit_s
            out.append((c, v))

        # localise+clear one detected region (approach to its rim)
        for d in view.detected:
            v = view.copy()
            v.pos = d.center
            v.detected = _remove_nearest_region(v.detected, d.center)
            approach = max(0.0, dist(view.pos, d.center) - max(0.0, d.r_mec))
            c = approach / self.speed + self.measure_s + self.clear_hit_s
            out.append((c, v))

        # take one exploration stop toward the best-covering anchor
        if view.n_unknown > 0 and view.unknown_holes and view.anchors:
            anchor = self._best_cover_anchor(view)
            if anchor is not None:
                v = view.copy()
                v.pos = anchor
                v.unknown_holes = [h for h in v.unknown_holes if dist(h, anchor) > self.fce.detection_radius]
                c = dist(view.pos, anchor) / self.speed + self.measure_s
                out.append((c, v))
        return out

    def _best_cover_anchor(self, view: CostView) -> Optional[Point]:
        r2 = self.fce.detection_radius ** 2
        step = max(1, len(view.unknown_holes) // self.fce.max_cover_holes)
        pts = view.unknown_holes[::step]
        best, best_gain = None, 0
        for a in view.anchors:
            g = sum(1 for h in pts if (h[0] - a[0]) ** 2 + (h[1] - a[1]) ** 2 <= r2)
            if g > best_gain:
                best_gain, best = g, a
        return best


# --- helpers -----------------------------------------------------------------


def _remove_nearest(points: Sequence[Point], target: Point) -> List[Point]:
    pts = list(points)
    if not pts:
        return pts
    idx = min(range(len(pts)), key=lambda i: dist(pts[i], target))
    return pts[:idx] + pts[idx + 1:]


def _remove_nearest_region(regions: Sequence[DetectedRegion], target: Point) -> List[DetectedRegion]:
    regs = list(regions)
    if not regs:
        return regs
    idx = min(range(len(regs)), key=lambda i: dist(regs[i].center, target))
    return regs[:idx] + regs[idx + 1:]
