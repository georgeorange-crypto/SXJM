"""Space-centric candidate generation (P0 #2/#3 — the ``planner=spatial`` lane).

The frozen ``CandidateGenerator`` is *action-centric*: one macro = one action at one
target (a clear here, a refine there, a coverage batch elsewhere). Its routing waste
(quantified by the Phase-A metrics) is that co-located services become separate stops
and a clear on the way is a separate detour.

``SpatialStopGenerator`` adds the space-centric object on top: it clusters the
generator's own co-located scan candidates into a single ``SpatialStop`` and folds any
nearby blind-clear point into that same stop (P0 #3 multi-service bundling — the mixed
MEASURE+CLEAR visit the action-centric candidate cannot express).

**HYBRID by design (why full-clear cannot regress).** ``generate`` returns *every*
legacy candidate unchanged **plus** the bundles — it never drops or merges away a
legacy option. So the spatial planner can always replicate the legacy plan exactly
(the same candidates are on the table), and the pipeline's frozen EXIT guard and
no-progress guard still backstop every tick. A bundle can only ever *add* an
alternative the planner may pick; it can never remove a coverage point, an anchor, a
refine, or a required clear. Making bundles reliably *win* (the route-aware
subadditive Ĵ) is Phase D (#7) — not here. Here they merely exist as first-class,
correctly-costed, ranked objects (#2) that bundle services (#3).

Belief / NBV / certificate / guards are untouched: this only reshapes candidates the
frozen generator already vetted. It certifies nothing and clears nothing (禁止6/7)."""

from __future__ import annotations

from typing import List, Optional, Sequence

from sxjm_core.geometry import Point, dist

from ..channels import SchedulerMode
from ..belief import ChannelStatus
from ..core import AnalyticalCostModel, MacroActionType, RobotState, SpatialStop
from .candidates import CandidateGenerator


class InformationRidge:
    """Shared spatial value field for multi-channel sensing.

    The ridge is deliberately a planner-only score: it aggregates per-channel
    marginal coverage/refinement value on common points, so overlapping tasks can
    produce one shared waypoint instead of one waypoint per channel.
    """

    def __init__(self, scheduler=None):
        self.scheduler = scheduler

    def score(self, point, belief, certificate) -> float:
        total = 0.0
        for c in range(1, belief.n_channels + 1):
            b = belief[c]
            if b.status in (ChannelStatus.CLEARED, ChannelStatus.ABSENT_CERTIFIED,
                            ChannelStatus.LOCALIZED):
                continue
            if b.status in (ChannelStatus.UNKNOWN, ChannelStatus.PRESENT_UNOBSERVED):
                total += float(certificate.coverage_gain(c, point))
            elif b.status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED):
                total += float(getattr(b, "readiness_score", lambda **_: 0.0)())
        return total

    def top(self, points, belief, certificate, k=8):
        scored = [(self.score(p, belief, certificate), p) for p in points]
        return [p for _s, p in sorted(scored, key=lambda x: (-x[0], x[1]))[:int(k)]]


class _Cluster:
    """A neighbourhood of co-located scan candidates + the clears that fall in it.

    The representative ``target`` is the first member's waypoint (the batch scan is
    charged from there); ``channels`` is the de-duplicated, order-preserving union of
    every member's scan batch; ``refine_gain`` is the largest member NBV shrink (a
    conservative stand-in until Phase D scores routes properly)."""

    __slots__ = ("target", "channels", "refine_gain", "n_members", "clears")

    def __init__(self, first) -> None:
        self.target: Point = (float(first.target[0]), float(first.target[1]))
        self.channels: List[int] = []
        self.refine_gain: float = 0.0
        self.n_members: int = 0
        self.clears: list = []
        self.add(first)

    def add(self, cand) -> None:
        self.n_members += 1
        for ch in cand.scan_channels:
            ci = int(ch)
            if ci not in self.channels:
                self.channels.append(ci)
        self.refine_gain = max(self.refine_gain, float(cand.refinement_gain))


class SpatialStopGenerator:
    """Wraps a frozen ``CandidateGenerator`` and augments its output with
    ``SpatialStop`` bundles (see module docstring — additive / non-destructive)."""

    def __init__(
        self,
        base: Optional[CandidateGenerator] = None,
        cost_model: Optional[AnalyticalCostModel] = None,
        cluster_radius: float = 1000.0,
    ) -> None:
        self.base = base if base is not None else CandidateGenerator(cost_model=cost_model)
        # keep timing consistent with the base generator when no explicit model given
        self.cost = cost_model if cost_model is not None else self.base.cost
        self.cluster_radius = float(cluster_radius)

    # -- public -------------------------------------------------------------

    def generate(
        self,
        belief,
        certificate,
        state: RobotState,
        scan_mode: SchedulerMode = SchedulerMode.EARLY,
    ) -> List:
        """Frozen legacy candidates, unchanged, **plus** any spatial bundles.

        The order (legacy first, bundles appended) is irrelevant to the planner, which
        ranks by ``q_value``; it only keeps the legacy list byte-identical to the
        ``planner=legacy`` path for anything that inspects it head-first."""
        cands = self.base.generate(belief, certificate, state, scan_mode=scan_mode)
        bundles = self._bundle(cands, state)
        return list(cands) + bundles

    # -- bundling -----------------------------------------------------------

    def _bundle(self, cands: Sequence, state: RobotState) -> List[SpatialStop]:
        scans = [c for c in cands if c.is_scan and c.scan_channels]
        clears = [c for c in cands if c.action_type == MacroActionType.CLEAR]
        if not scans:
            # No scan clusters to anchor a bundle. Standalone clears stay as legacy
            # CLEAR candidates (already in ``cands``); routing pure-clear sequences is
            # Phase D (unified remaining-task route), not here.
            return []
        clusters = self._cluster_scans(scans)
        self._attach_clears(clusters, clears)
        bundles: List[SpatialStop] = []
        for cl in clusters:
            # Emit a bundle only when it genuinely bundles: two-or-more co-located
            # scans merged into one visit, OR a clear folded into a scan stop. A lone
            # scan with no clear is exactly a legacy candidate — no new object needed.
            if cl.n_members >= 2 or cl.clears:
                bundles.append(self._make_stop(cl, state))
        return bundles

    def _cluster_scans(self, scans: Sequence) -> List[_Cluster]:
        """Greedy proximity clustering: a scan joins the first existing cluster whose
        representative is within ``cluster_radius``, else it seeds a new one. Cheap and
        order-stable; exact cluster shape does not affect correctness (bundles are only
        added options), only how aggressively co-located work is merged."""
        clusters: List[_Cluster] = []
        for c in scans:
            placed = False
            for cl in clusters:
                if dist((float(c.target[0]), float(c.target[1])), cl.target) <= self.cluster_radius:
                    cl.add(c)
                    placed = True
                    break
            if not placed:
                clusters.append(_Cluster(c))
        return clusters

    def _attach_clears(self, clusters: Sequence[_Cluster], clears: Sequence) -> None:
        """Fold each clear into the nearest cluster within ``cluster_radius`` (a
        clear-on-the-way). A clear with no cluster in range is left untouched — it
        survives as its own legacy CLEAR candidate in the passthrough list."""
        for cc in clears:
            if cc.clear_channel is None:
                continue
            tgt = (float(cc.target[0]), float(cc.target[1]))
            best: Optional[_Cluster] = None
            best_d = self.cluster_radius
            for cl in clusters:
                d = dist(tgt, cl.target)
                if d <= best_d:
                    best_d = d
                    best = cl
            if best is not None:
                best.clears.append(cc)

    def _make_stop(self, cl: _Cluster, state: RobotState) -> SpatialStop:
        channels = tuple(cl.channels)
        clear_channels = tuple(int(c.clear_channel) for c in cl.clears)
        clear_targets = tuple((float(c.target[0]), float(c.target[1])) for c in cl.clears)
        stop = SpatialStop(
            action_type=MacroActionType.STOP,
            target=cl.target,
            scan_channels=channels,
            clear_channels=clear_channels,
            clear_targets=clear_targets,
        )
        stop.refinement_gain = cl.refine_gain
        # certificate_gain is a diagnostic here (# distinct channels serviced); the
        # planner ranks by expected_time + future cost, so the honest chained duration
        # is what matters. Phase D replaces this with a route-aware subadditive Ĵ.
        stop.certificate_gain = float(len(channels))
        stop.expected_time = self._stop_time(state, stop)
        stop.meta["bundle_members"] = cl.n_members
        stop.meta["bundle_clears"] = len(clear_channels)
        return stop

    def _stop_time(self, state: RobotState, stop: SpatialStop) -> float:
        """Honest chained duration of the whole visit: the batch scan at ``target``
        (only the first measure pays the move, §7), then each folded clear from the
        running pose. Uses the same cost model as the legacy generator, so a bundle and
        its constituent legacy candidates are costed on identical arithmetic."""
        cur = state
        total = 0.0
        if stop.scan_channels:
            total += self.cost.batch_scan_time_s(cur, stop.target, stop.scan_channels)
            cur = RobotState(
                float(stop.target[0]), float(stop.target[1]), int(stop.scan_channels[-1]), 0
            )
        for tgt in stop.clear_targets:
            total += self.cost.clear_time_s(cur, tgt, hit=True)
            cur = RobotState(float(tgt[0]), float(tgt[1]), cur.channel, 0)
        return total
