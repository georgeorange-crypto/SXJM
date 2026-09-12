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
from .fallback import omni_fallback_anchors
from .hard_disc_cover import HardDiscCoverVerifier
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
    hard_complete: bool = False                       # arbitrary-disc-cover verdict (sticky True)
    completion_source: Optional[CertificateSource] = None
    active_scan_count: int = 0                        # real NO_SIGNAL scans folded in
    fallback_anchor_count: int = 0                    # distinct Way3 anchors reached
    present: bool = False                             # a positive obs stopped certificate work
    visited_anchor_idx: Set[int] = field(default_factory=set)
    _hard_checked_count: int = -1                     # scan count at last nominal verify


class CertificateManager:
    """Owns coverage certificates for all channels (§6.4)."""

    def __init__(
        self,
        n_channels: int = 20,
        max_sources: int = 16,               # cardinality upper bound (line 27)
        min_sources: int = 10,               # cardinality lower bound (§6.5; total ∈ [10,16])
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
    ) -> None:
        self.n_channels = n_channels
        self.max_sources = max_sources
        self.min_sources = min_sources
        self.anchor_reached_tol = anchor_reached_tol
        self.eager_hard_verify = eager_hard_verify
        self.eager_ratio_threshold = eager_ratio_threshold
        self.rescue_only_in_verification = rescue_only_in_verification
        self.speed = speed
        self.scan_time_s = scan_time_s

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
        self.anchors: List[Point] = (
            list(fallback_anchors) if fallback_anchors is not None else omni_fallback_anchors()
        )
        self.certs: Dict[int, OmniChannelCertificate] = {
            c: OmniChannelCertificate(channel=c) for c in range(1, n_channels + 1)
        }
        # Invariant-D-pure sets: written ONLY from real positive observations / clears.
        self._present: Set[int] = set()
        self._cleared: Set[int] = set()

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
            # Invariant D — the SOLE write path into ``_present``: a real positive
            # observation. Nothing derived from a *counting* argument (e.g.
            # ``forced_present_channels``' cardinality lower bound) may ever be added
            # here, or the =16 pigeonhole would feed on a guess and could certify a
            # real channel ABSENT (§6.10 危险放大器). ``mark_cleared`` is the only
            # other writer, and only for a channel a deterministic hit confirmed.
            self._present.add(channel)
            return
        if not obs.is_no_signal:
            return
        if cert.present or channel in self._cleared:
            return

        p = (float(point[0]), float(point[1]))
        cert.negative_scan_points.append(p)
        cert.active_scan_count += 1
        self.map.add_no_signal(channel, p)
        cert.heuristic_coverage_ratio = self.map.coverage_ratio(channel)

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
        heuristic-ratio threshold and skipped once a scan set has already failed."""
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
        """Way3 guaranteed template: every fixed omni anchor scanned NO_SIGNAL for
        this channel (Invariant C). Sound because the anchor discs cover D_1800
        with margin (asserted by the quadtree in tests)."""
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

    # -- cardinality LOWER bound (§6.5; planning-only, NEVER certifies) -----

    def cardinality_bounds(self, belief: "BeliefState") -> Tuple[int, int, int, int]:
        """Propagate the mission cardinality window ``T ∈ [min_sources, max_sources]``
        into a bound on how many of the still-UNKNOWN channels must hold a source.

        With ``p`` channels proven PRESENT (Invariant-D-pure ``_present``) and ``u``
        channels still UNKNOWN — neither present nor certified absent, i.e. the *live
        candidates* for a remaining source — the count ``k`` of sources hiding among
        the unknowns satisfies ``T = p + k`` with ``T ∈ [min_sources, max_sources]``:

            k ∈ [q_min, q_max],   q_min = max(0, min_sources − p),
                                   q_max = min(u, max_sources − p).

        ``belief`` MUST be this manager's companion (same run): ``p`` is read from
        ``_present`` while ``u`` is read from ``belief.unknown_channels()``, so a
        mismatched belief would desync the two counts and make ``forced_present``
        over-eager. In-tree they stay consistent —
        ``macro_executor._fold_observation`` updates belief and certificate from the
        *same* observation atomically, and a present channel is DETECTED/LOCALIZED in
        belief (never in ``unknown_channels()``).

        Returns ``(p, u, q_min, q_max)``. PURE READ — touches neither ``_present``,
        ``ChannelStatus`` nor any certificate. Recomputed from live counts each call,
        so it is inherently non-sticky / reversible (a later positive obs that raises
        ``p`` simply shifts the window on the next call)."""
        p = len(self._present)
        u = sum(1 for c in belief.unknown_channels() if c not in self._present)
        q_min = max(0, self.min_sources - p)
        # Clamp q_max at 0: if p already exceeds max_sources (an upstream invariant
        # break) we return a benign degenerate window rather than a negative bound —
        # fallback-to-Way3 posture over a crash. q_min > q_max then flags infeasible.
        q_max = max(0, min(u, self.max_sources - p))
        return p, u, q_min, q_max

    def forced_present_channels(self, belief: "BeliefState") -> Set[int]:
        """The UNKNOWN channels the cardinality LOWER bound proves MUST hold a source.

        Fires only when ``q_min == u`` (equivalently ``p + u == min_sources`` while
        ``p < min_sources``): the live candidates number exactly the minimum total,
        so every remaining unknown is necessarily occupied and can be named
        forced-present. Otherwise we know only "≥ q_min of the unknowns are occupied"
        without knowing WHICH → the empty set (never name a specific channel on a
        pure counting argument). The infeasible case ``q_min > u`` (some absence
        over-certified upstream) also yields the empty set — safe, no crash.

        PLANNING-ONLY, and deliberately kept OUT of the certificate machinery:
          * never writes ``ChannelStatus`` — belief owns channel state (§4);
          * never enters ``_present`` nor the cardinality COUNT. A forced-present that
            fed the =16 pigeonhole would let a lower-bound *guess* certify a real
            channel ABSENT — the §6.10 Invariant-D "危险放大器". This method only READS.
        The planner consumes it to prefer INITIALIZE/search over futile absence
        coverage on channels already proven occupied (P0-C consumer)."""
        p, u, q_min, q_max = self.cardinality_bounds(belief)
        if u > 0 and q_min == u:
            return {c for c in belief.unknown_channels() if c not in self._present}
        return set()

    def cardinality_feasible(self, belief: "BeliefState") -> bool:
        """False-absence detector (§6.10). Returns whether the cardinality window is
        still satisfiable: ``q_min <= q_max``, equivalently ``p + u >= min_sources``.

        The window goes infeasible (``p + u < min_sources``) ONLY when a channel that
        truly holds a source was wrongly certified ABSENT — it left ``u`` without ever
        entering ``_present``, so the live candidates can no longer reach the mission
        minimum of ``min_sources``. That is precisely the §6.10 "危险放大器" catastrophe
        the design fears most, and here it is a *free* detector: the counts already
        exist. (A ``min_sources > max_sources`` misconfiguration would also trip this,
        but that is a construction-time error, not a runtime false-absent.)

        PURE READ, and deliberately a PREDICATE, not a raise: the certificate layer
        keeps its "fallback-to-Way3 over crash" posture (禁令). The consumer chooses
        the response — a loud assert in verification / tests, a soft warning in a
        live run, but NEVER an action that could drop full-clear (§16 禁令10)."""
        _p, _u, q_min, q_max = self.cardinality_bounds(belief)
        return q_min <= q_max

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
        elif self.absent_by_cardinality(channel):
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

    # -- planner-facing heuristics (never certify) -------------------------

    def heuristic_coverage_ratio(self, channel: int) -> float:
        return self.map.coverage_ratio(channel)

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
