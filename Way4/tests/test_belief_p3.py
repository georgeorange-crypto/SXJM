"""Way4 P3 belief tests (DESIGN.md M2, §15).

Two soundness properties dominate:
  * true source never excluded by a positive update (the +(1°+eps) margin);
  * NO false clear: whenever a channel is clearable, the blind-clear point (MEC
    centre) is within the clear radius of the true source.
Plus Invariant D structurally: NO_SIGNAL never makes a channel PRESENT.
"""

import math
import random

from way4.belief import BeliefState, ChannelBelief, ChannelStatus

ARENA = 1800.0


def _in_convex_polygon(p, poly, eps=1e-6):
    n = len(poly)
    sign = 0
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        s = 1 if cr > eps else (-1 if cr < -eps else 0)
        if s != 0:
            if sign == 0:
                sign = s
            elif s != sign:
                return False
    return True


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _rand_source(rng):
    ang = rng.uniform(0, 2 * math.pi)
    rad = ARENA * math.sqrt(rng.random())
    return (rad * math.cos(ang), rad * math.sin(ang))


def _bearing(src, obs):
    return math.degrees(math.atan2(src[1] - obs[1], src[0] - obs[0])) % 360.0


# --- Invariant D: NO_SIGNAL never confers presence ---------------------------


def test_no_signal_never_makes_present():
    cb = ChannelBelief(channel=3)
    for _ in range(20):
        cb.record_no_signal((random.uniform(-1800, 1800), random.uniform(-1800, 1800)))
    assert cb.status == ChannelStatus.UNKNOWN
    assert cb.is_present is False
    assert cb.is_clearable is False
    assert cb.F_c is None
    assert len(cb.negative_discs) == 20


def test_no_signal_disc_excludes_points():
    cb = ChannelBelief(channel=1)
    cb.record_no_signal((0.0, 0.0))
    assert cb.excludes((10.0, 0.0)) is True      # within 1000 of the scan point
    assert cb.excludes((1500.0, 0.0)) is False   # outside the exclusion disc


def test_no_signal_updates_effective_feasible_set_and_hypotheses():
    cb = ChannelBelief(channel=11)
    before = cb.effective_area(spacing=60.0)
    cb.record_no_signal((0.0, 0.0))
    after = cb.effective_area(spacing=60.0)
    assert after < before
    assert cb.contains_possible_source((1200.0, 0.0)) is True
    assert cb.contains_possible_source((500.0, 0.0)) is False
    assert all(cb.contains_possible_source(p) for p in cb.sample_effective_hypotheses(spacing=60.0))


def test_positive_outer_geometry_and_negative_effective_geometry_are_distinct():
    cb = ChannelBelief(channel=12)
    cb.record_bearing((0.0, 0.0), 0.0)
    outer_area = cb.area
    effective_before = cb.effective_area(spacing=60.0)
    cb.record_no_signal((500.0, 0.0))
    assert cb.area == outer_area  # safety outer geometry is unchanged
    assert cb.effective_area(spacing=60.0) < effective_before


def test_geometry_summary_has_anisotropy_and_axis():
    cb = ChannelBelief(channel=13)
    cb.record_bearing((0.0, 0.0), 0.0)
    assert math.isfinite(cb.kappa)
    assert cb.principal_axis is not None
    assert math.isclose(math.hypot(*cb.principal_axis), 1.0, rel_tol=1e-6)


# --- the true-source-never-excluded property --------------------------------


def _feed_bearings(cb, source, obs_points, rng):
    for s in obs_points:
        true_b = _bearing(source, s)
        cb.record_bearing(s, true_b + rng.uniform(-1.0, 1.0))


def test_true_source_never_excluded():
    rng = random.Random(2024)
    for _ in range(2000):
        source = _rand_source(rng)
        k = rng.randint(1, 4)
        obs = []
        while len(obs) < k:
            # A positive detection can only occur within R_eff (<=1500 m) of the
            # source, so a physically-realizable observation point is within that
            # range. Place obs at a random bearing and distance in [60, 1490].
            a = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(60.0, 1490.0)
            obs.append((source[0] + d * math.cos(a), source[1] + d * math.sin(a)))
        cb = ChannelBelief(channel=5)
        _feed_bearings(cb, source, obs, rng)
        assert cb.F_c, "feasible set empty"
        assert _in_convex_polygon(source, cb.F_c, eps=1e-3), f"source excluded: {source}"
        # Safety corollary: MEC of F_c contains the source.
        assert _dist(cb.clear_target, source) <= cb.mec_radius + 1e-4


# --- the no-false-clear safety invariant -------------------------------------


def test_no_false_clear_monte_carlo():
    """Whenever a channel becomes clearable, the blind-clear point must be within
    the clear radius of the true source (else /clear could miss)."""
    rng = random.Random(99)
    clearable_seen = 0
    for _ in range(1500):
        source = _rand_source(rng)
        # Three observation points spread ~120 deg around the source, moderate
        # range -> the +-1 deg wedges intersect in a small feasible lens.
        d = rng.uniform(90.0, 240.0)
        base = rng.uniform(0, 2 * math.pi)
        obs = []
        for j in range(3):
            a = base + j * 2 * math.pi / 3 + rng.uniform(-0.3, 0.3)
            obs.append((source[0] + d * math.cos(a), source[1] + d * math.sin(a)))
        cb = ChannelBelief(channel=7)
        _feed_bearings(cb, source, obs, rng)
        if cb.is_clearable:
            clearable_seen += 1
            assert cb.mec_radius <= cb.clear_threshold + 1e-9
            gap = _dist(cb.clear_target, source)
            assert gap <= cb.clear_threshold + 1e-4, f"UNSAFE CLEAR: gap={gap}"
            assert gap < 20.0  # would actually succeed against the 20 m clear radius
    assert clearable_seen > 100, f"test rarely reached clearable ({clearable_seen})"


# --- near / status transitions -----------------------------------------------


def test_near_makes_clearable_and_present():
    rng = random.Random(1)
    source = (500.0, -300.0)
    near_pt = (source[0] + 3.0, source[1] - 2.0)  # within 5 m, as 'near' implies
    cb = ChannelBelief(channel=2)
    cb.record_near(near_pt)
    assert cb.is_present is True
    assert cb.is_clearable is True
    assert _dist(cb.clear_target, source) <= cb.mec_radius + 1e-6
    assert _dist(cb.clear_target, source) < 20.0


def test_cardinality_forced_presence_lifecycle():
    cb = ChannelBelief(channel=14)
    assert cb.mark_present_unobserved() is True
    assert cb.status == ChannelStatus.PRESENT_UNOBSERVED
    assert cb.is_present is True
    cb.record_bearing((0.0, 0.0), 0.0)
    assert cb.status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED, ChannelStatus.LOCALIZED)


def test_initialization_requires_non_degenerate_bearing_baseline():
    cb = ChannelBelief(1)
    cb.record_bearing((0.0, 0.0), 10.0)
    cb.record_bearing((100.0, 0.0), 12.0)
    assert cb.initialization_ready is False
    assert cb.status is ChannelStatus.DETECTED
    cb.record_bearing((200.0, 0.0), 40.0)
    assert cb.initialization_ready is True
    assert cb.status in (ChannelStatus.INITIALIZED, ChannelStatus.LOCALIZED)


def test_readiness_score_is_bounded_and_penalizes_travel():
    cb = ChannelBelief(1)
    cb.record_bearing((0.0, 0.0), 0.0)
    cb.record_bearing((100.0, 0.0), 45.0)
    near = cb.readiness_score((0.0, 0.0), coverage_debt=0.0)
    far = cb.readiness_score((1800.0, 1800.0), coverage_debt=1.0)
    assert 0.0 <= far <= near <= 1.0


def test_effective_geometry_reports_connected_components():
    cb = ChannelBelief(1)
    cb.record_no_signal((0.0, 0.0))
    cb.record_no_signal((1200.0, 0.0))
    assert cb.connected_components >= 1
    assert cb.effective_components(spacing=120.0) == cb.connected_components


def test_status_progression_unknown_detected_localized():
    source = (200.0, 100.0)
    cb = ChannelBelief(channel=4)
    assert cb.status == ChannelStatus.UNKNOWN
    # One distant bearing: present but MEC still huge -> DETECTED.
    cb.record_bearing((0.0, 0.0), _bearing(source, (0.0, 0.0)))
    assert cb.status == ChannelStatus.DETECTED
    assert cb.is_present and not cb.is_clearable
    # A few more crossing bearings from nearby -> LOCALIZED.
    for s in [(200.0, 260.0), (330.0, 90.0), (70.0, 90.0)]:
        cb.record_bearing(s, _bearing(source, s))
    assert cb.status == ChannelStatus.LOCALIZED
    assert cb.is_clearable


def test_mark_absent_only_when_unknown():
    cb = ChannelBelief(channel=8)
    assert cb.mark_absent_certified() is True  # UNKNOWN -> ABSENT_CERTIFIED
    assert cb.status == ChannelStatus.ABSENT_CERTIFIED
    # A present channel must never be markable absent.
    cb2 = ChannelBelief(channel=9)
    cb2.record_near((0.0, 0.0))
    assert cb2.mark_absent_certified() is False
    assert cb2.status == ChannelStatus.LOCALIZED


def test_cleared_status_sticks():
    cb = ChannelBelief(channel=10)
    cb.record_near((0.0, 0.0))
    cb.mark_cleared()
    assert cb.status == ChannelStatus.CLEARED
    assert cb.is_resolved
    # Further observations must not resurrect a cleared channel.
    cb.record_bearing((100.0, 0.0), 180.0)
    assert cb.status == ChannelStatus.CLEARED


# --- BeliefState container ---------------------------------------------------


def test_belief_state_aggregates():
    bs = BeliefState(n_channels=20)
    assert bs.present_count() == 0
    assert not bs.all_resolved()
    bs[1].record_near((0.0, 0.0))
    bs[1].mark_cleared()
    bs[2].record_bearing((0.0, 0.0), 45.0)
    assert set(bs.present_channels()) == {1, 2}
    assert bs.present_count() == 2
    # Resolve everything: clear 2, certify the remaining 18 UNKNOWN absent.
    bs[2].record_near((0.0, 0.0))
    bs[2].mark_cleared()
    for c in bs.unknown_channels():
        bs[c].mark_absent_certified()
    assert bs.all_resolved()
