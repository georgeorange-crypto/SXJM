"""Way4 P3 belief tests (DESIGN.md M2, §15).

Two soundness properties dominate:
  * true source never excluded by a positive update (the +(1°+eps) margin);
  * NO false clear: whenever a channel is clearable, the blind-clear point (MEC
    centre) is within the clear radius of the true source.
Plus Invariant D structurally: NO_SIGNAL never makes a channel PRESENT.
"""

import math
import random

import pytest

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


# --- P0-A: NO_SIGNAL enters the effective feasible set (planning only) --------


def test_effective_diameter_never_exceeds_convex_diameter():
    """The effective diameter is a max over a subset of F_c vertices, so it can
    only be <= the convex-superset diameter — never an over-claim."""
    source = (600.0, 0.0)
    cb = ChannelBelief(channel=1)
    # one distant bearing -> a long thin sliver (large diameter)
    cb.record_bearing((0.0, 0.0), _bearing(source, (0.0, 0.0)))
    d_convex = cb.diameter
    # no exclusion discs yet -> effective == convex (never claim an unprovable shrink)
    assert cb.effective_diameter() == d_convex


def test_effective_diameter_is_max_over_surviving_vertices():
    """White-box: with a known square F_c and a disc clipping the two far corners,
    the effective diameter is the diameter over the SURVIVING vertices — strictly
    below the convex diameter, and never touching mec_*/clear geometry."""
    from way4.belief.channel import ExclusionDisc
    cb = ChannelBelief(channel=1)
    # a 1000 x 1000 square; convex diameter = the 1414 m diagonal
    cb.F_c = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)]
    cb.diameter = math.hypot(1000.0, 1000.0)
    # a disc that swallows only the two right-hand corners (x=1000), leaving the
    # left edge (x=0). Survivors: (0,0),(0,1000) -> effective diameter = 1000.
    cb.negative_discs.append(ExclusionDisc((1000.0, 500.0), 600.0))
    survivors = cb.effective_vertices()
    assert (0.0, 0.0) in survivors and (0.0, 1000.0) in survivors
    assert (1000.0, 0.0) not in survivors and (1000.0, 1000.0) not in survivors
    assert cb.effective_diameter() == pytest.approx(1000.0)
    assert cb.effective_diameter() < cb.diameter


def test_no_signal_never_shrinks_clear_geometry():
    """Adding a real NO_SIGNAL disc to a DETECTED channel leaves the CLEAR geometry
    (mec_*, clear_target, is_clearable, area, status) byte-for-byte on the sound
    convex superset — only the planning-side effective diameter may move (§3.3,
    禁令10). effective_diameter stays <= the convex diameter throughout."""
    source = (600.0, 0.0)
    cb = ChannelBelief(channel=1)
    cb.record_bearing((0.0, 0.0), _bearing(source, (0.0, 0.0)))  # sliver east along +x
    d_convex_before = cb.diameter
    snap = (cb.mec_center, cb.mec_radius, cb.area, cb.status, list(cb.F_c))

    # NO_SIGNAL far behind the source on the -x axis (>1500 m from the true source,
    # so the non-detection is sound whatever R_eff∈[1000,1500] is).
    cb.record_no_signal((-950.0, 0.0))
    assert cb.negative_discs, "disc recorded"
    assert cb.effective_diameter() <= d_convex_before + 1e-9
    assert cb.diameter == d_convex_before               # convex superset untouched
    assert (cb.mec_center, cb.mec_radius, cb.area, cb.status, list(cb.F_c)) == snap
    assert all(not cb.excludes(v) for v in cb.effective_vertices())


def test_effective_diameter_falls_back_when_all_vertices_excluded():
    """Degenerate guard: if every F_c vertex sits inside a disc (numeric edge), fall
    back to the convex diameter rather than claim a zero/unprovable shrink."""
    from way4.belief.channel import ExclusionDisc
    source = (100.0, 0.0)
    cb = ChannelBelief(channel=1)
    cb.record_bearing((0.0, 0.0), _bearing(source, (0.0, 0.0)))
    # a giant disc swallowing the whole region -> <2 survivors -> fall back
    cb.negative_discs.append(ExclusionDisc((0.0, 0.0), 1.0e7))
    assert cb.effective_diameter() == cb.diameter


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
