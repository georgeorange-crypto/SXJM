"""FutureCostEstimator — cheap analytical cost-to-go ``Ĵ(B)`` (DESIGN.md §10).

The receding-horizon planner needs to know, for a *predicted* belief ``B'``, how
much work is still expected before EXIT — cheaply, thousands of times per episode.
So ``Ĵ`` is a sum of four **analytical proxies**, never an exact rollout tree
(禁用整局精确树) and never the hard quadtree (§6.8):

    Ĵ(B) = J_route + J_localization + J_exploration + J_certificate

  * **J_route**        — visit-and-clear the already-clearable sources: cached
    Held–Karp set tour + nearest entry (§9 hot-loop proxy) + a clear dwell each.
  * **J_localization** — convert every DETECTED region to clearable: a TSPN approach
    over the region discs + the offline-fit ``L̂`` localisation cost + a clear each.
  * **J_certificate**  — finish hard coverage of the CURRENT actual holes: a greedy
    set-cover of the still-uncovered §6.6 grid cells with the remaining fallback
    anchors, then the travel to walk the chosen anchors. **(Conservatism note B)**
    the holes come ONLY from the in-arena coverage map, never the hard verifier's
    unresolved cells — those include out-of-arena shells that would lure the robot
    into covering nonexistent area.
  * **J_exploration**  — the sensing dwell to actually scan the UNKNOWN band at the
    chosen coverage stops (the measurement time J_certificate's travel omits).

Every term is in **seconds**. This is a ranking signal for the planner, not a
guarantee: the full-clear guarantee lives in the SafetyShield / EXIT guard (§11),
so an imperfect proxy can only cost time, never correctness. It marks nothing and
certifies nothing (Invariant B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import radians, sin
from typing import List, Optional, Sequence, Tuple

from sxjm_core.geometry import angle_sep_deg, dist

from ..belief import ChannelStatus
from ..routing import LocalizationCostModel, Neighborhood, RouteEstimator

Point = Tuple[float, float]


@dataclass
class DetectedRegion:
    """A DETECTED channel's region, reduced to what the localisation cost needs."""

    center: Point
    r_mec: float
    diameter: float
    sin_gamma: float = 0.5          # crossing-quality proxy; smaller => costlier to fix
    channel: int = 0


@dataclass
class FutureCost:
    """The decomposed cost-to-go (seconds)."""

    j_route: float = 0.0
    j_localization: float = 0.0
    j_exploration: float = 0.0
    j_certificate: float = 0.0
    total: float = 0.0
    n_cover_stops: int = 0
    measurement_predictions: dict = field(default_factory=dict)


@dataclass
class CostView:
    """The minimal snapshot ``Ĵ`` consumes. Decoupling the cost math from the live
    belief/certificate lets the receding-horizon planner perturb cheap *copies* for
    hypothetical outcomes without cloning numpy masks or re-running the quadtree."""

    pos: Point
    clearable_targets: List[Point] = field(default_factory=list)
    detected: List[DetectedRegion] = field(default_factory=list)
    unknown_holes: List[Point] = field(default_factory=list)   # §6.6 in-arena holes only
    n_unknown: int = 0
    anchors: List[Point] = field(default_factory=list)          # shared immutable template

    def copy(self) -> "CostView":
        return CostView(
            self.pos,
            list(self.clearable_targets),
            list(self.detected),
            list(self.unknown_holes),
            self.n_unknown,
            self.anchors,                # never mutated -> shared, not copied
        )


def _sin_gamma_proxy(channel_belief, default: float = 0.3) -> float:
    """Best existing bearing-crossing quality ``|sin(Δ)|`` for a DETECTED channel;
    a single bearing has no crossing yet, so it gets a poor default (harder to
    localise -> larger ``L̂``)."""
    bs = getattr(channel_belief, "bearings", []) or []
    if len(bs) < 2:
        return default
    best = 0.0
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            best = max(best, abs(sin(radians(angle_sep_deg(bs[i].svd_deg, bs[j].svd_deg)))))
    return max(best, default)


def build_cost_view(
    belief,
    certificate,
    state,
    default_sin_gamma: float = 0.3,
) -> CostView:
    """Snapshot the live belief/certificate into a ``CostView`` (called once per
    replan; the planner perturbs copies for the outcome lookahead)."""
    clearable: List[Point] = []
    for c in belief.clearable_channels():
        t = belief[c].clear_target
        if t is not None:
            clearable.append((float(t[0]), float(t[1])))

    detected: List[DetectedRegion] = []
    for c in range(1, belief.n_channels + 1):
        b = belief[c]
        if b.status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED) and b.mec_center is not None:
            detected.append(
                DetectedRegion(
                    (float(b.mec_center[0]), float(b.mec_center[1])),
                    float(b.mec_radius),
                    float(b.diameter),
                    _sin_gamma_proxy(b, default_sin_gamma),
                    channel=c,
                )
            )

    unknown = belief.unknown_channels()
    pending_present = belief.unobserved_present_channels()
    holes: List[Point] = []
    if certificate is not None and unknown:
        seen = set()
        for c in unknown:
            for h in certificate.remaining_holes(c):
                key = (round(h[0], 3), round(h[1], 3))
                if key not in seen:
                    seen.add(key)
                    holes.append((float(h[0]), float(h[1])))

    anchors = [(float(a[0]), float(a[1])) for a in certificate.anchors] if certificate is not None else []
    return CostView(state.pos, clearable, detected, holes,
                    len(unknown) + len(pending_present), anchors)


class FutureCostEstimator:
    def __init__(
        self,
        route: Optional[RouteEstimator] = None,
        loc_model: Optional[LocalizationCostModel] = None,
        speed: float = 5.0,
        measure_s: float = 5.0,
        clear_hit_s: float = 5.0,
        detection_radius: float = 1000.0,
        w_route: float = 1.0,
        w_localization: float = 1.0,
        w_exploration: float = 1.0,
        w_certificate: float = 1.0,
        max_cover_holes: int = 500,
        joint_route: bool = False,
    ) -> None:
        self.route = route or RouteEstimator(speed=speed)
        self.loc = loc_model or LocalizationCostModel()
        self.speed = float(speed)
        self.measure_s = float(measure_s)
        self.clear_hit_s = float(clear_hit_s)
        self.detection_radius = float(detection_radius)
        self.w_route = float(w_route)
        self.w_localization = float(w_localization)
        self.w_exploration = float(w_exploration)
        self.w_certificate = float(w_certificate)
        self.max_cover_holes = int(max_cover_holes)
        # P0 #4/#7 (planner=spatial): score remaining work as ONE unified route
        # instead of four independent additive sub-tours. Default OFF -> the legacy
        # additive estimate is byte-identical (full-clear provably unchanged).
        self.joint_route = bool(joint_route)

    # -- top level ----------------------------------------------------------

    def estimate(self, view: CostView) -> FutureCost:
        if self.joint_route:
            return self._estimate_joint(view)
        return self._estimate_additive(view)

    def _estimate_additive(self, view: CostView) -> FutureCost:
        j_route = self._j_route(view)
        j_loc = self._j_localization(view)
        j_cert, n_stops = self._j_certificate(view)
        j_expl = self._j_exploration(view, n_stops)
        total = (
            self.w_route * j_route
            + self.w_localization * j_loc
            + self.w_exploration * j_expl
            + self.w_certificate * j_cert
        )
        return FutureCost(j_route, j_loc, j_expl, j_cert, total, n_stops,
                          self._measurement_predictions(view))

    # -- joint route (P0 #4/#7, planner=spatial) ---------------------------

    def _estimate_joint(self, view: CostView) -> FutureCost:
        """One *unified* open route over every remaining task — clearables, DETECTED
        region approaches, and the certificate cover stops — instead of the three
        independent start-rooted tours the additive estimate sums. Co-located work
        then pays its travel ONCE (the additive sum re-pays the trip out from ``pos``
        for each sub-tour), so a spatially-bundled plan (a ``SpatialStop``) is finally
        scored as the saving it is rather than as three separate detours.

        A fixed clear/cover point is just a ``Neighborhood`` of radius 0, so the whole
        pool routes through the existing (tested) TSPN solver. Dwell terms
        (clear/measure/localisation) are order-independent and *identical* to the
        additive estimate — only the travel unifies.

        **Subadditive cap:** a single TSP over disjoint clusters can be longer than
        three start-rooted open tours, and Phase D must never score a plan as *worse*
        than legacy would. So the joint result is returned only when it does not
        exceed the additive one; otherwise the exact additive result is returned
        unchanged. Hence ``joint.total <= additive.total`` always. Correctness is
        never at stake here — like the additive proxy, this only *ranks*; the
        full-clear guarantee is the SafetyShield / EXIT guard's job (§11)."""
        add = self._estimate_additive(view)
        chosen = (
            self._greedy_cover(view.unknown_holes, view.anchors)
            if (view.n_unknown and view.unknown_holes)
            else []
        )
        neigh: List[Neighborhood] = [
            Neighborhood(t, 0.0, 0.0) for t in view.clearable_targets
        ]
        for d in view.detected:
            neigh.append(
                Neighborhood(
                    d.center,
                    max(0.0, d.r_mec),
                    self.loc.estimate(d.r_mec, d.diameter, d.sin_gamma),
                )
            )
        neigh.extend(Neighborhood(a, 0.0, 0.0) for a in chosen)
        if not neigh:
            return FutureCost(0.0, 0.0, 0.0, 0.0, 0.0, 0)

        plan = self.route.tspn_route(view.pos, neigh)
        travel_time = plan.length / self.speed
        # unified-route decomposition: all travel folds into j_route (#4); the dwell
        # split matches the additive estimate so the two totals are travel-comparable.
        j_route = travel_time + self.clear_hit_s * len(view.clearable_targets)
        j_loc = plan.localization_cost + self.clear_hit_s * len(view.detected)
        j_expl = self.measure_s * len(chosen)
        j_cert = 0.0
        total = (
            self.w_route * j_route
            + self.w_localization * j_loc
            + self.w_exploration * j_expl
            + self.w_certificate * j_cert
        )
        joint = FutureCost(j_route, j_loc, j_expl, j_cert, total, len(chosen))
        return joint if joint.total <= add.total else add

    # -- J_route ------------------------------------------------------------

    def _j_route(self, view: CostView) -> float:
        if not view.clearable_targets:
            return 0.0
        travel = self.route.estimate_clear_travel(view.pos, view.clearable_targets)
        return travel / self.speed + self.clear_hit_s * len(view.clearable_targets)

    # -- J_localization -----------------------------------------------------

    def _j_localization(self, view: CostView) -> float:
        if not view.detected:
            return 0.0
        neigh = [
            Neighborhood(
                d.center,
                radius=max(0.0, d.r_mec),
                localization_cost=self.loc.estimate(d.r_mec, d.diameter, d.sin_gamma),
            )
            for d in view.detected
        ]
        plan = self.route.tspn_route(view.pos, neigh)
        # approach travel + expected localisation work + one clear per region
        return plan.length / self.speed + plan.localization_cost + self.clear_hit_s * len(view.detected)

    @staticmethod
    def _measurement_predictions(view: CostView) -> dict:
        """Uncertainty-aware estimate of remaining measurements per DETECTED channel.

        This is deliberately a ranking proxy: larger MEC/diameter and poor
        crossing geometry require more refinement; it never writes belief or
        certificate state.
        """
        out = {}
        for idx, d in enumerate(view.detected):
            key = int(d.channel) if d.channel else idx
            geometry = (max(0.0, d.r_mec) / 10.0) + (max(0.0, d.diameter) / 20.0)
            crossing = 1.0 / max(abs(d.sin_gamma), 1e-3)
            out[key] = max(1, int(round(1.0 + geometry * crossing)))
        return out

    # -- J_certificate + stop count ----------------------------------------

    def _j_certificate(self, view: CostView) -> Tuple[float, int]:
        if view.n_unknown == 0 or not view.unknown_holes:
            return 0.0, 0
        chosen = self._greedy_cover(view.unknown_holes, view.anchors)
        if not chosen:
            return 0.0, 0
        travel = self.route.estimate_clear_travel(view.pos, chosen)
        return travel / self.speed, len(chosen)

    def _greedy_cover(self, holes: Sequence[Point], anchors: Sequence[Point]) -> List[Point]:
        """Greedy set-cover of the (subsampled) uncovered cells with the fallback
        anchors (§10). Each anchor covers cells within ``detection_radius``; pick the
        anchor covering the most still-uncovered cells until covered or no progress.
        The fixed backbone provably covers ``D_1800`` (Invariant C), so this always
        terminates covered when anchors remain."""
        if not anchors:
            return []
        # subsample the dense grid so the cover stays cheap in the hot loop
        step = max(1, len(holes) // self.max_cover_holes)
        pts = list(holes)[::step]
        r2 = self.detection_radius * self.detection_radius
        # per-anchor covered cell indices
        covers: List[set] = []
        for a in anchors:
            s = {
                i for i, h in enumerate(pts)
                if (h[0] - a[0]) ** 2 + (h[1] - a[1]) ** 2 <= r2
            }
            covers.append(s)
        uncovered = set(range(len(pts)))
        chosen: List[Point] = []
        used = [False] * len(anchors)
        while uncovered:
            best_i, best_gain = -1, 0
            for i, s in enumerate(covers):
                if used[i]:
                    continue
                g = len(s & uncovered)
                if g > best_gain:
                    best_gain, best_i = g, i
            if best_i < 0 or best_gain == 0:
                break                     # remaining holes uncoverable by anchors
            used[best_i] = True
            chosen.append(anchors[best_i])
            uncovered -= covers[best_i]
        return chosen

    # -- J_exploration ------------------------------------------------------

    def _j_exploration(self, view: CostView, n_stops: int) -> float:
        """Sensing dwell to scan the UNKNOWN band at the coverage stops (the measure
        time J_certificate's travel term omits). One representative batch dwell per
        stop — batch scan folds the whole UNKNOWN band into that one dwell (§7)."""
        if view.n_unknown == 0 or n_stops == 0:
            return 0.0
        return self.measure_s * n_stops
