"""CertificateManager unit tests (DESIGN.md §6, M3).

Covers the three-source disjunction (§6.5), the two-layer separation (§6.2,
Invariant B: the heuristic map never certifies), cardinality purity (Invariant D),
the legacy backbone (Invariant C), and the planner-facing queries (§6.4/§6.7).
"""

import copy
import math

from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager, CertificateSource
from way4.core import Observation


def _grid(spacing, lim):
    pts, n = [], int(lim // spacing)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            pts.append((i * spacing, j * spacing))
    return pts


def _feed_no_signal(mgr, channel, points):
    for p in points:
        mgr.record_observation(channel, p, Observation.no_signal())


# --- NO_SIGNAL accumulation, never present (Invariant A/D) --------------------


def test_no_signal_accumulates_and_never_present():
    mgr = CertificateManager()
    _feed_no_signal(mgr, 3, [(0.0, 0.0), (500.0, 0.0), (-500.0, 300.0)])
    cert = mgr.certs[3]
    assert cert.present is False
    assert cert.active_scan_count == 3
    assert len(cert.negative_scan_points) == 3
    assert 3 not in mgr.present_channels()
    # a few sparse scans cannot cover the arena
    assert mgr.is_absent_certified(3, force=True) is False


def test_positive_stops_certificate_and_ignores_later_no_signal():
    mgr = CertificateManager()
    mgr.record_observation(7, (100.0, 0.0), Observation.bearing(42.0))
    assert mgr.certs[7].present is True
    assert 7 in mgr.present_channels()
    # NO_SIGNAL after a positive must not accrue coverage (a present channel needs
    # no absence certificate) and must never certify it absent.
    _feed_no_signal(mgr, 7, _grid(400.0, 2000.0))
    assert mgr.certs[7].active_scan_count == 0
    assert mgr.is_absent_certified(7, force=True) is False


# --- Invariant B: the heuristic map NEVER certifies ---------------------------


def test_heuristic_map_never_certifies():
    # No fallback anchors (legacy disabled) and the hard verifier stubbed to fail:
    # even a heuristically-'complete' map must not certify absence.
    mgr = CertificateManager(fallback_anchors=[])
    mgr._verifier.is_covered = lambda pts: False        # type: ignore[assignment]
    mgr._rescue_verifiers = []
    _feed_no_signal(mgr, 2, _grid(400.0, 2000.0))
    assert mgr.heuristic_coverage_ratio(2) > 0.9         # map thinks it's essentially covered
    assert mgr.map.is_fully_covered_heuristic(2) or mgr.heuristic_coverage_ratio(2) > 0.9
    assert mgr.is_absent_certified(2, force=True) is False   # but nothing certifies it


# --- arbitrary disc cover (§6.5, source ARBITRARY_DISC_COVER) -----------------


def test_arbitrary_disc_cover_certifies():
    mgr = CertificateManager()
    _feed_no_signal(mgr, 5, _grid(400.0, 2000.0))
    assert mgr.is_absent_certified(5, force=True) is True
    assert mgr.certificate_source(5) is CertificateSource.ARBITRARY_DISC_COVER
    assert mgr.hard_coverage_complete(5) is True


# --- legacy backbone (§6.5, Invariant C) --------------------------------------


def test_legacy_backbone_completes_when_all_anchors_scanned():
    mgr = CertificateManager()
    anchors = list(mgr.anchors)
    assert len(anchors) > 0
    for a in anchors:
        mgr.record_observation(4, a, Observation.no_signal())
    assert mgr.legacy_backbone_complete(4) is True
    # partial visits do not complete the backbone
    mgr2 = CertificateManager()
    for a in anchors[:-1]:
        mgr2.record_observation(4, a, Observation.no_signal())
    assert mgr2.legacy_backbone_complete(4) is False


def test_legacy_backbone_source_when_arbitrary_cannot_certify():
    # Force the arbitrary verifier to fail so the legacy disjunct is the one that fires.
    mgr = CertificateManager()
    mgr._verifier.is_covered = lambda pts: False        # type: ignore[assignment]
    mgr._rescue_verifiers = []
    for a in mgr.anchors:
        mgr.record_observation(9, a, Observation.no_signal())
    assert mgr.is_absent_certified(9, force=True) is True
    assert mgr.certificate_source(9) is CertificateSource.LEGACY_BACKBONE


# --- cardinality (§6.5, Invariant D) ------------------------------------------


def test_cardinality_certifies_remaining_unknowns():
    mgr = CertificateManager(n_channels=20, max_sources=16)
    belief = BeliefState(n_channels=20)
    # 16 channels confirmed PRESENT by a real positive observation (both layers).
    for c in range(1, 17):
        belief[c].record_bearing((0.0, 0.0), 30.0)
        mgr.record_observation(c, (0.0, 0.0), Observation.bearing(30.0))
    assert mgr.present_count() == 16
    # every remaining UNKNOWN channel is now absent by pigeonhole
    for c in range(17, 21):
        assert mgr.absent_by_cardinality(c) is True
        assert mgr.is_absent_certified(c) is True
        assert mgr.certificate_source(c) is CertificateSource.CARDINALITY
    # a PRESENT channel is never absent by cardinality
    assert mgr.absent_by_cardinality(1) is False
    assert mgr.is_absent_certified(1) is False


def test_cardinality_purity_requires_real_positive():
    # 15 real positives -> cardinality must NOT trigger (Invariant D: no guessing).
    mgr = CertificateManager(max_sources=16)
    for c in range(1, 16):
        mgr.record_observation(c, (0.0, 0.0), Observation.near())
    assert mgr.present_count() == 15
    assert mgr.absent_by_cardinality(20) is False
    assert mgr.is_absent_certified(20) is False


# --- cardinality LOWER bound (§6.5; P0-B, planning-only) -----------------------


def _make_present(mgr, belief, channels):
    """Confirm ``channels`` PRESENT in BOTH layers via a real positive observation."""
    for c in channels:
        belief[c].record_bearing((0.0, 0.0), 30.0)
        mgr.record_observation(c, (0.0, 0.0), Observation.bearing(30.0))


def test_cardinality_bounds_window():
    """(p, u, q_min, q_max) with p present and u live-unknown, T ∈ [10, 16]."""
    mgr = CertificateManager(n_channels=20, min_sources=10, max_sources=16)
    belief = BeliefState(n_channels=20)
    _make_present(mgr, belief, range(1, 7))              # p = 6
    p, u, q_min, q_max = mgr.cardinality_bounds(belief)
    assert p == 6
    assert u == 14                                       # 20 - 6 still UNKNOWN
    assert q_min == max(0, 10 - 6) == 4                  # ≥4 of the unknowns occupied
    assert q_max == min(14, 16 - 6) == 10


def test_forced_present_fires_only_when_qmin_equals_u():
    """p=6 present, and exactly u=4 channels left as live candidates (the other 10
    certified absent) -> q_min == u == 4 -> all four unknowns are forced-present."""
    mgr = CertificateManager(n_channels=20, min_sources=10, max_sources=16)
    belief = BeliefState(n_channels=20)
    _make_present(mgr, belief, range(1, 7))              # channels 1..6 present
    # certify channels 11..20 absent (full active cover), leaving 7,8,9,10 unknown
    for c in range(11, 21):
        _feed_no_signal(mgr, c, _grid(400.0, 2000.0))
    mgr.apply_certifications(belief, force=True)
    for c in range(11, 21):
        assert belief[c].status is ChannelStatus.ABSENT_CERTIFIED

    p, u, q_min, q_max = mgr.cardinality_bounds(belief)
    assert (p, u, q_min) == (6, 4, 4)                    # q_min == u
    forced = mgr.forced_present_channels(belief)
    assert forced == {7, 8, 9, 10}


def test_forced_present_empty_when_slack_remains():
    """p=6, u=14 (nothing certified absent) -> q_min=4 < u -> we know ≥4 unknowns are
    occupied but not WHICH -> forced set is empty (never name on a counting slack)."""
    mgr = CertificateManager(n_channels=20, min_sources=10, max_sources=16)
    belief = BeliefState(n_channels=20)
    _make_present(mgr, belief, range(1, 7))
    assert mgr.forced_present_channels(belief) == set()


def test_forced_present_empty_at_or_above_min():
    """p >= min_sources -> q_min = 0 -> no unknown is forced (the lower bound is
    already satisfied by the present channels alone)."""
    mgr = CertificateManager(n_channels=20, min_sources=10, max_sources=16)
    belief = BeliefState(n_channels=20)
    _make_present(mgr, belief, range(1, 13))             # p = 12 >= 10
    _p, _u, q_min, _q_max = mgr.cardinality_bounds(belief)
    assert q_min == 0
    assert mgr.forced_present_channels(belief) == set()


def test_forced_present_never_pollutes_present_or_pigeonhole():
    """The §6.10 guard: forced-present is a PLANNING hint only. Computing it must not
    add anything to ``_present`` nor let the =16 pigeonhole fire on a lower-bound
    guess (that would certify a real channel ABSENT — the danger amplifier)."""
    mgr = CertificateManager(n_channels=20, min_sources=10, max_sources=16)
    belief = BeliefState(n_channels=20)
    _make_present(mgr, belief, range(1, 7))              # p = 6
    for c in range(11, 21):
        _feed_no_signal(mgr, c, _grid(400.0, 2000.0))
    mgr.apply_certifications(belief, force=True)

    # True no-mutation guard: snapshot the manager's entire mutable certificate state
    # before the read-only queries and require it byte-identical afterwards. This
    # exercises the actual "computing forced-present mutates nothing" contract, rather
    # than leaning on the =16 pigeonhole check below (which is vacuous at p=6, since
    # the pigeonhole needs len(_present) >= 16 and forced-present can never push
    # _present past 10 anyway).
    before_present = set(mgr._present)
    before_cleared = set(mgr._cleared)
    before_certs = copy.deepcopy(mgr.certs)

    forced = mgr.forced_present_channels(belief)
    _ = mgr.cardinality_bounds(belief)
    _ = mgr.cardinality_feasible(belief)

    assert mgr._present == before_present
    assert mgr._cleared == before_cleared
    assert mgr.certs == before_certs

    assert forced == {7, 8, 9, 10}
    # purity: _present is still exactly the 6 real positives, not 6 + 4 forced
    assert mgr.present_count() == 6
    assert set(mgr.present_channels()) == set(range(1, 7))
    # and the pigeonhole must NOT treat a forced channel as present -> no false absent
    for c in forced:
        assert mgr.absent_by_cardinality(c) is False
        assert mgr.is_absent_certified(c) is False
        assert belief[c].status is ChannelStatus.UNKNOWN


def test_forced_present_survives_infeasible_window_without_crash():
    """If upstream over-certifies absence so that q_min would exceed u, the method
    returns empty (fallback posture) rather than raising."""
    mgr = CertificateManager(n_channels=12, min_sources=10, max_sources=16)
    belief = BeliefState(n_channels=12)
    _make_present(mgr, belief, range(1, 3))              # p = 2
    # certify 9 of the remaining 10 absent -> u = 1, but q_min = 8 > u
    for c in range(4, 13):
        _feed_no_signal(mgr, c, _grid(400.0, 2000.0))
    mgr.apply_certifications(belief, force=True)
    p, u, q_min, _q_max = mgr.cardinality_bounds(belief)
    assert q_min > u                                     # infeasible-looking window
    assert mgr.forced_present_channels(belief) == set()  # empty, no exception


def test_cardinality_feasible_detects_false_absence():
    """``cardinality_feasible`` is the §6.10 false-absent detector: the window stays
    satisfiable (True) as long as the live candidates can still reach min_sources,
    and goes False exactly when p + u < min_sources — reachable ONLY by a wrong
    ABSENT certification (a truly-present channel dropped from the count)."""
    # Healthy: p=2, u=10 -> p+u=12 >= 10 -> feasible even though few are present yet.
    mgr = CertificateManager(n_channels=12, min_sources=10, max_sources=16)
    belief = BeliefState(n_channels=12)
    _make_present(mgr, belief, range(1, 3))              # p = 2, u = 10
    assert mgr.cardinality_feasible(belief) is True
    # exactly at the boundary p + u == min_sources is still feasible (q_min == q_max path)
    _p, _u, q_min, q_max = mgr.cardinality_bounds(belief)
    assert q_min <= q_max

    # Now over-certify absence so only 1 unknown remains: p=2, u=1 -> p+u=3 < 10.
    for c in range(4, 13):                               # certify 9 channels absent
        _feed_no_signal(mgr, c, _grid(400.0, 2000.0))
    mgr.apply_certifications(belief, force=True)
    p, u, _q_min, _q_max = mgr.cardinality_bounds(belief)
    assert p + u < mgr.min_sources                       # window can no longer be filled
    assert mgr.cardinality_feasible(belief) is False     # -> a false absence is flagged
    # detector is a PURE READ predicate — it certifies nothing and raises nothing
    assert belief[4].status is ChannelStatus.ABSENT_CERTIFIED


# --- apply_certifications marks belief absent (禁止6: only the manager) --------


def test_apply_certifications_marks_only_unknown_covered_channels():
    mgr = CertificateManager(n_channels=3)
    belief = BeliefState(n_channels=3)
    # channel 1: active full cover -> certifiable absent
    _feed_no_signal(mgr, 1, _grid(400.0, 2000.0))
    # channel 2: real present (both layers) -> must NOT be marked absent
    belief[2].record_near((10.0, 10.0))
    mgr.record_observation(2, (10.0, 10.0), Observation.near())
    # channel 3: only a couple scans -> not certifiable
    _feed_no_signal(mgr, 3, [(0.0, 0.0), (100.0, 0.0)])

    newly = mgr.apply_certifications(belief, force=True)
    assert newly == [1]
    assert belief[1].status is ChannelStatus.ABSENT_CERTIFIED
    assert belief[2].status in (ChannelStatus.DETECTED, ChannelStatus.LOCALIZED)
    assert belief[3].status is ChannelStatus.UNKNOWN


# --- planner-facing queries (§6.4/§6.6/§6.7) ----------------------------------


def test_coverage_gain_decreases_after_scanning_there():
    mgr = CertificateManager()
    g0 = mgr.coverage_gain(6, (0.0, 0.0))
    assert g0 > 0.0
    mgr.record_observation(6, (0.0, 0.0), Observation.no_signal())
    g1 = mgr.coverage_gain(6, (0.0, 0.0))
    assert g1 < g0            # already-covered grid centres no longer count as new
    assert mgr.heuristic_coverage_ratio(6) > 0.0


def test_suggest_completion_points_orders_and_shrinks():
    mgr = CertificateManager()
    n_anchors = len(mgr.anchors)
    pts = mgr.suggest_completion_points(1, (0.0, 0.0))
    assert len(pts) == n_anchors                      # nothing scanned yet
    # scan one suggested anchor -> it drops out of the next suggestion
    mgr.record_observation(1, pts[0], Observation.no_signal())
    pts2 = mgr.suggest_completion_points(1, (0.0, 0.0))
    assert len(pts2) == n_anchors - 1
    # a present channel yields no completion work
    mgr.record_observation(2, (0.0, 0.0), Observation.bearing(10.0))
    assert mgr.suggest_completion_points(2, (0.0, 0.0)) == []


def test_metrics_report_active_fraction():
    mgr = CertificateManager()
    _feed_no_signal(mgr, 5, _grid(400.0, 2000.0))
    assert mgr.is_absent_certified(5, force=True) is True
    m = mgr.metrics()
    assert m["certified_channels"] == 1.0
    assert m["active_scan_certificate_fraction"] == 1.0
