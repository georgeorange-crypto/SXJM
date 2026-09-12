"""CertificateManager — per-channel coverage certificates (DESIGN.md §6, M3).

Answers, per channel, the existence question "could an un-cleared source still be
hiding in the arena?" via the three-source disjunction (§6.5):

    is_absent_certified(c) = arbitrary_disc_cover_complete(c)   # active NO_SIGNAL scans cover D_1800
                           ∨ legacy_backbone_complete(c)        # walked Way3's guaranteed template
                           ∨ absent_by_cardinality(c)           # 16 channels already proven PRESENT

Two-layer separation (§6.2), never mixed:
  * ``CoverageGainMap`` — fast 40 m grid, planner scoring only, NEVER certifies (Invariant B).
  * ``HardDiscCoverVerifier`` — conservative quadtree, the ONLY geometric certifier
    (sound; false negatives allowed, false positives never — §6.3).

Invariants enforced here:
  * A — only a real NO_SIGNAL observation appends a hard exclusion disc / scan point.
  * B — the heuristic map never drives ABSENT (only the three sources above).
  * C — the Way3 backbone always guarantees completion when the arbitrary verifier
        can't certify a zero-margin seam (note A).
  * D — cardinality counts only channels a real positive observation confirmed
        PRESENT (``_present`` is fed solely by ``obs.is_positive``); a mistaken
        PRESENT would, via the cardinality shortcut, wrongly certify a real channel
        absent — so purity is guarded at the single write site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

from .coverage_gain import CoverageGainMap
from .fallback import directional_fallback_anchors, omni_fallback_anchors
from .hard_disc_cover import HardDiscCoverVerifier
from .directional_certificate import DirectionalCounterexample, check_point
from ..belief import ChannelStatus
from ..core import Observation

if TYPE_CHECKING:  # avoid any import cost / cycle at runtime
    from ..belief import BeliefState

Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class CertificateSource(str, Enum):
    """Which disjunct certified a channel absent (§6.5)."""

    ARBITRARY_DISC_COVER = "ARBITRARY_DISC_COVER"   # active NO_SIGNAL scans cover D_1800
    LEGACY_BACKBONE = "LEGACY_BACKBONE"             # walked Way3's guaranteed template
    CARDINALITY = "CARDINALITY"                     # 16 channels already PRESENT (pigeonhole)


@dataclass
class OmniChannelCertificate:
    """Per-channel coverage state (DESIGN.md §6.4)."""

    channel: int
    negative_scan_points: List[Point] = field(default_factory=list)
    heuristic_coverage_ratio: float = 0.0
    coverage_debt: float = 1.0
    uncovered_area: float = 0.0
    hard_complete: bool = False                       # arbitrary-disc-cover verdict (sticky True)
    completion_source: Optional[CertificateSource] = None
    active_scan_count: int = 0                        # real NO_SIGNAL scans folded in
    fallback_anchor_count: int = 0                    # distinct Way3 anchors reached
    present: bool = False                             # a positive obs stopped certificate work
    visited_anchor_idx: Set[int] = field(default_factory=set)
    _hard_checked_count: int = -1                     # scan count at last nominal verify
    directional_local_certificate: bool = False       # local convex-hull check only
    directional_counterexample: Optional[DirectionalCounterexample] = None


@dataclass(frozen=True)
class CardinalityState:
    """Sound global 10..16 source-count bounds over the 20 channels."""

    present: int
    absent: int
    unknown: int
    q_min: int
    q_max: int


class CertificateManager:
    """Owns coverage certificates for all channels (§6.4)."""

    def __init__(
        self,
        n_channels: int = 20,
        problem: int = 3,                    # 3 = omni only; 4 = directional mix (§6.9)
        max_sources: int = 16,               # cardinality upper bound (line 27)
        arena_radius: float = 1800.0,
        detection_radius: float = 1000.0,    # guaranteed R_eff lower bound (§6.1)
        coverage_spacing: float = 40.0,      # §6.6
        min_cell_size: float = 10.0,         # §6.10 nominal
        rescue_cell_sizes: Sequence[float] = (5.0, 2.5),   # §6.10 rescue
        rescue_only_in_verification: bool = True,
        max_depth: int = 11,
        eps: float = 1e-7,
        anchor_reached_tol: float = 1.0,     # <= design margin, keeps backbone soundness
        eager_hard_verify: bool = True,
        eager_ratio_threshold: float = 0.995,  # cheap gate before running the quadtree (§6.8)
        speed: float = 5.0,
        scan_time_s: float = 5.0,
        fallback_anchors: Optional[List[Point]] = None,
        enable_no_signal: bool = True,
        enable_cardinality: bool = True,
    ) -> None:
        self.n_channels = n_channels
        self.problem = int(problem)
        self.max_sources = max_sources
        self.anchor_reached_tol = anchor_reached_tol
        self.eager_hard_verify = eager_hard_verify
        self.eager_ratio_threshold = eager_ratio_threshold
        self.rescue_only_in_verification = rescue_only_in_verification
        self.speed = speed
        self.scan_time_s = scan_time_s
        self.enable_no_signal = bool(enable_no_signal)
        self.enable_cardinality = bool(enable_cardinality)

        self.map = CoverageGainMap(arena_radius, coverage_spacing, detection_radius)
        self._verifier = HardDiscCoverVerifier(
            radius=detection_radius, arena_radius=arena_radius,
            min_cell_size=min_cell_size, max_depth=max_depth, eps=eps,
        )
        self._rescue_verifiers = [
            HardDiscCoverVerifier(
                radius=detection_radius, arena_radius=arena_radius,
                min_cell_size=s, max_depth=max_depth, eps=eps,
            )
            for s in rescue_cell_sizes
        ]
        # Backbone anchors ARE the guaranteed-completion template, and it differs by
        # problem (§6.9). P3: Way3's omni 1-cover (7 pts). P4: Way3's directional
        # 3-cover (~31 pts), whose 'verify_three_cover' angular-gap<180° theorem is the
        # ONLY sound hard absent-certificate for a possibly-directional source (禁止5:
        # an omni NO_SIGNAL disc says nothing about a source facing away).
        if fallback_anchors is not None:
            self.anchors = list(fallback_anchors)
        elif self.problem == 4:
            self.anchors = directional_fallback_anchors(src_radius=arena_radius)
        else:
            self.anchors = omni_fallback_anchors()
        self.certs: Dict[int, OmniChannelCertificate] = {
            c: OmniChannelCertificate(channel=c) for c in range(1, n_channels + 1)
        }
        # Invariant-D-pure sets: written ONLY from real positive observations / clears.
        self._present: Set[int] = set()
        self._cleared: Set[int] = set()

    # -- planner-facing residual responsibility (never a certificate) ----------

    def unverified_region(self, channel: int) -> List[Point]:
        return list(self.map.remaining_holes(int(channel)))

    def remaining_backbone_anchors(self, channel: int) -> List[Point]:
        cert = self.certs[int(channel)]
        return [p for i, p in enumerate(self.anchors) if i not in cert.visited_anchor_idx]

    # -- observation intake (§6.4) -----------------------------------------

    def record_observation(self, channel: int, point: Point, obs: Observation) -> None:
        """Fold one channel's scan outcome at ``point`` into its certificate.

        Positive  -> mark PRESENT, stop certificate computation (§6.4); historical
                     negative discs stay for spatial exclusion but no new coverage
                     accrues (a present channel needs no absence certificate).
        NO_SIGNAL -> append scan point + update the heuristic map + (only when the
                     cheap ratio gate says it's worth it) run the hard verifier."""
        cert = self.certs[channel]
        if obs.is_positive:
            cert.present = True
            self._present.add(channel)     # Invariant D: only real positive obs counts
            return
        if not obs.is_no_signal:
            return
        if not self.enable_no_signal:
            return
        if cert.present or channel in self._cleared:
            return

        p = (float(point[0]), float(point[1]))
        cert.negative_scan_points.append(p)
        cert.active_scan_count += 1
        if self.problem == 4:
            # P4 must not turn a directional NO_SIGNAL into an omni exclusion.
            # This local check is nevertheless useful: once the current point is
            # surrounded by nearby scan points, it records the new P4 geometry
            # without certifying the whole arena.
            points = cert.negative_scan_points
            probe = (sum(x for x, _ in points) / len(points),
                     sum(y for _, y in points) / len(points))
            cert.directional_counterexample = check_point(
                probe, points, radius=self._verifier.radius
            )
            cert.directional_local_certificate = (
                cert.directional_counterexample is None
                and len(cert.negative_scan_points) >= 3
            )
        self.map.add_no_signal(channel, p)
        cert.heuristic_coverage_ratio = self.map.coverage_ratio(channel)
        cert.coverage_debt = max(0.0, 1.0 - cert.heuristic_coverage_ratio)
        cert.uncovered_area = float(len(self.map.remaining_holes(channel))) * self.map.spacing ** 2

        # legacy backbone bookkeeping: which fixed anchors has this channel reached?
        for j, a in enumerate(self.anchors):
            if j not in cert.visited_anchor_idx and _dist(p, a) <= self.anchor_reached_tol:
                cert.visited_anchor_idx.add(j)
        cert.fallback_anchor_count = len(cert.visited_anchor_idx)

        # §6.8: only run the quadtree once heuristic coverage is essentially complete.
        if (
            self.eager_hard_verify
            and not cert.hard_complete
            and cert.heuristic_coverage_ratio >= self.eager_ratio_threshold
        ):
            self._arbitrary_disc_cover_complete(channel, force=False)

    def mark_cleared(self, channel: int) -> None:
        """A source on ``channel`` was cleared (deterministic hit). Stops its
        certificate work; still counts as a real PRESENT for cardinality."""
        self._cleared.add(channel)
        self._present.add(channel)
        self.certs[channel].present = True

    # -- the three certificate sources (§6.5) ------------------------------

    def _arbitrary_disc_cover_complete(self, channel: int, force: bool = False) -> bool:
        """Hard verifier over this channel's real NO_SIGNAL scan discs.

        ``force`` (EXIT / verification mode) always runs the quadtree and may
        escalate to the rescue cell sizes; otherwise the run is gated on the cheap
        heuristic-ratio threshold and skipped once a scan set has already failed.

        **P4 soundness gate (禁止5, §6.9).** The arbitrary omni disc-cover rests on
        "NO_SIGNAL@s ⟹ no source within B(s,1000)", which holds ONLY for an
        omnidirectional source. A *directional* source inside B(s,1000) facing away
        also returns NO_SIGNAL, so tiling the arena with omni discs can never certify
        a possibly-directional channel absent. On P4 this disjunct is therefore
        disabled outright — absence there comes only from the directional 3-cover
        backbone (``legacy_backbone_complete``, whose ``verify_three_cover`` angular-gap
        theorem IS sound for directional sources) or cardinality. This is the fix for
        the seed-2000/2003 regression (a true directional source wrongly ABSENT-ed)."""
        if self.problem == 4:
            return False
        cert = self.certs[channel]
        if cert.present:
            return False
        if cert.hard_complete:
            return True
        n = len(cert.negative_scan_points)
        if n == 0:
            return False
        if not force:
            if cert.heuristic_coverage_ratio < self.eager_ratio_threshold:
                return False
            if cert._hard_checked_count == n:
                return False   # already tried this exact scan set at nominal resolution

        ok = self._verifier.is_covered(cert.negative_scan_points)
        if not ok and force:
            # rescue: finer resolution, verification/EXIT only (§6.10)
            for v in self._rescue_verifiers:
                if v.is_covered(cert.negative_scan_points):
                    ok = True
                    break
        cert._hard_checked_count = n
        if ok:
            cert.hard_complete = True
            if cert.completion_source is None:
                cert.completion_source = CertificateSource.ARBITRARY_DISC_COVER
        return ok

    def legacy_backbone_complete(self, channel: int) -> bool:
        """Way3 guaranteed template: every fixed backbone anchor scanned NO_SIGNAL for
        this channel (Invariant C). The backbone (and the theorem certifying it) is
        problem-dependent (§6.9):

          * **P3** — Way3's omni 1-cover (``omni_scan_points``); the anchor discs cover
            D_1800 with margin (``verify_one_cover``, asserted by the quadtree in tests).
          * **P4** — Way3's directional 3-cover (``directional_scan_points``); soundness
            is the angular-gap theorem (``verify_three_cover``): from every potential
            source point, the anchors within R_lo=1000 span a max angular gap < 180°, so
            no 180° directional blind-spot can hide a source from all of them. This is
            the ONLY sound absent-certificate when a source may be directional (禁止5)."""
        cert = self.certs[channel]
        if cert.present:
            return False
        return len(self.anchors) > 0 and len(cert.visited_anchor_idx) == len(self.anchors)

    def absent_by_cardinality(self, channel: int) -> bool:
        """Pigeonhole (§6.5): once ``max_sources`` distinct channels are proven
        PRESENT, no un-found channel can hold a (17th) source. ``_present`` is
        Invariant-D-pure (real positive observations only)."""
        if self.certs[channel].present:
            return False
        return len(self._present) >= self.max_sources

    def cardinality_state(self) -> CardinalityState:
        """Return the complete remaining-source interval.

        ``present`` is deliberately sourced only from real positive observations
        or clears; ``absent`` is sourced only from hard certifications.  The
        interval is therefore safe for planning and future lifecycle propagation.
        """
        present = len(self._present)
        absent = sum(
            1 for cert in self.certs.values()
            if not cert.present and self.is_absent_certified(cert.channel, force=False)
        )
        unknown = max(0, self.n_channels - present - absent)
        return CardinalityState(
            present=present,
            absent=absent,
            unknown=unknown,
            q_min=max(0, 10 - present),
            q_max=min(unknown, max(0, 16 - present)),
        )

    # -- combined queries (§6.4) -------------------------------------------

    def hard_coverage_complete(self, channel: int, force: bool = False) -> bool:
        """Geometric hard coverage (arbitrary disc cover OR legacy backbone) —
        excludes the cardinality shortcut."""
        return self._arbitrary_disc_cover_complete(channel, force) or self.legacy_backbone_complete(channel)

    def is_absent_certified(self, channel: int, force: bool = False) -> bool:
        """Three-source disjunction (§6.5). Records the winning source. A PRESENT
        channel is never absent."""
        cert = self.certs[channel]
        if cert.present:
            return False
        if self._arbitrary_disc_cover_complete(channel, force):
            source = CertificateSource.ARBITRARY_DISC_COVER
        elif self.legacy_backbone_complete(channel):
            source = CertificateSource.LEGACY_BACKBONE
        elif self.enable_cardinality and self.absent_by_cardinality(channel):
            source = CertificateSource.CARDINALITY
        else:
            return False
        if cert.completion_source is None:
            cert.completion_source = source
        return True

    def apply_certifications(self, belief: "BeliefState", force: bool = True) -> List[int]:
        """Push ABSENT_CERTIFIED into belief for every UNKNOWN channel the three
        sources certify. The ONLY place a channel is marked absent (禁止6: the
        planner never does). Returns the channels newly certified."""
        newly: List[int] = []
        for c in range(1, self.n_channels + 1):
            b = belief[c]
            if b.status == ChannelStatus.UNKNOWN and self.is_absent_certified(c, force=force):
                if b.mark_absent_certified():
                    newly.append(c)
        return newly

    def apply_cardinality_presence(self, belief: "BeliefState") -> List[int]:
        """Propagate PRESENT_UNOBSERVED only when every remaining unknown is forced present.

        If ``q_min == unknown`` then the lower cardinality bound proves each
        remaining channel contains a source.  A weaker ``q_min > 0`` does not
        identify which channel and therefore performs no per-channel mutation.
        """
        if not self.enable_cardinality:
            return []
        state = self.cardinality_state()
        if state.unknown == 0 or state.q_min != state.unknown:
            return []
        newly: List[int] = []
        for c in range(1, self.n_channels + 1):
            if self.is_absent_certified(c, force=False):
                continue
            if belief[c].mark_present_unobserved():
                newly.append(c)
        return newly

    # -- planner-facing heuristics (never certify) -------------------------

    def heuristic_coverage_ratio(self, channel: int) -> float:
        return self.map.coverage_ratio(channel)

    def coverage_debt(self, channel: int) -> float:
        """Normalised remaining heuristic coverage debt (planner-only)."""
        return self.certs[channel].coverage_debt

    def uncovered_area(self, channel: int) -> float:
        return self.certs[channel].uncovered_area

    def coverage_snapshot(self, channel: int) -> Dict[str, float]:
        """Unified planner/trace view of coverage state; never a certificate."""
        cert = self.certs[channel]
        return {
            "coverage_ratio": float(cert.heuristic_coverage_ratio),
            "coverage_debt": float(cert.coverage_debt),
            "uncovered_area": float(cert.uncovered_area),
            "active_scan_count": float(cert.active_scan_count),
            "backbone_anchor_count": float(cert.fallback_anchor_count),
            "directional_local_certificate": float(cert.directional_local_certificate),
        }

    def directional_counterexample(self, channel: int) -> Optional[DirectionalCounterexample]:
        """Return the latest actionable P4 local geometry witness, if any.

        This is a diagnostic/planner signal, not an arena-wide absence proof.
        """
        return self.certs[channel].directional_counterexample

    def coverage_gain(self, channel: int, q: Point) -> float:
        return self.map.coverage_gain(channel, q)

    def batch_coverage_gain(self, channels, q: Point, weights=None) -> float:
        return self.map.batch_coverage_gain(channels, q, weights)

    def remaining_holes(self, channel: int) -> List[Point]:
        return self.map.remaining_holes(channel)

    def suggest_completion_points(
        self, channel: int, robot_pos: Point, max_points: Optional[int] = None
    ) -> List[Point]:
        """Dynamic-order completion waypoints from the fallback anchors (§6.7 V1):
        greedy by coverage gain per unit travel+scan cost. Empty for a channel that
        is already present/resolved."""
        cert = self.certs[channel]
        if cert.present or channel in self._cleared:
            return []
        scored: List[Tuple[float, float, Point]] = []
        for j, a in enumerate(self.anchors):
            if j in cert.visited_anchor_idx:
                continue
            gain = self.map.coverage_gain(channel, a)
            cost = _dist(robot_pos, a) / self.speed + self.scan_time_s
            scored.append((gain, cost, a))
        scored.sort(key=lambda t: (-t[0], t[1]))   # best gain first, cheapest to break ties
        pts = [a for (_g, _c, a) in scored]
        return pts if max_points is None else pts[:max_points]

    # -- aggregate views (EXIT guard / metrics) ----------------------------

    def present_channels(self) -> List[int]:
        return sorted(self._present)

    def present_count(self) -> int:
        return len(self._present)

    def cleared_count(self) -> int:
        return len(self._cleared)

    def backbone_anchor_progress(self, channel: int) -> int:
        """Distinct fixed backbone anchors scanned NO_SIGNAL for ``channel`` — the
        legacy-backbone completion progress (Invariant C). Read by the pipeline's
        no-progress guard so that walking the Way3 fallback template counts as real
        forward progress; it certifies nothing (that's ``legacy_backbone_complete``)."""
        return len(self.certs[channel].visited_anchor_idx)

    def certificate_source(self, channel: int) -> Optional[CertificateSource]:
        return self.certs[channel].completion_source

    def metrics(self) -> Dict[str, float]:
        """DESIGN.md §15 certificate metrics (Way4 should beat Way3's fixed cost):
        how much certification came from active scanning vs. the fallback backbone."""
        sources = [c.completion_source for c in self.certs.values() if c.completion_source]
        n_cert = len(sources)
        by_active = sum(1 for s in sources if s is CertificateSource.ARBITRARY_DISC_COVER)
        total_anchor_visits = sum(c.fallback_anchor_count for c in self.certs.values())
        total_active_scans = sum(c.active_scan_count for c in self.certs.values())
        return {
            "certified_channels": float(n_cert),
            "active_scan_certificate_fraction": (by_active / n_cert) if n_cert else 0.0,
            "fallback_anchor_count": float(total_anchor_visits),
            "active_scan_count": float(total_active_scans),
            "present_count": float(len(self._present)),
        }
