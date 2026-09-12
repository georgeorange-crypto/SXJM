"""Way4 M8 hypothesis-layer tests (DESIGN.md §3.1, §3.2, §14).

The milestone gate (§14): *"随机合法定向源，sound negative rule 不消除真源所在保守
cell."* — for a random legal directional source, the sound negative rule must
never eliminate the hypothesis (conservative cell + true heading bin) the source
actually occupies.

The dominant property is soundness, exactly as in the P3 tests: an *honest*
sensor is simulated (a scan detects the source iff the source is within its true
effective range **and**, when directional, the scan point lies within the ±90°
arc of the antenna). Whatever that sensor reports NO_SIGNAL / BEARING / NEAR, the
hypothesis layer must keep the true (cell, bin) alive. We adversarially pick the
*worst* legal effective range — the guaranteed lower bound 1000 m, which yields
the most NO_SIGNALs and hence the most elimination pressure — and adversarial
scan points, including ones directly behind the antenna (off-arc but in range:
the case that must spare the true bin while killing others).
"""

import math
import random

import numpy as np

from way4.belief.hypothesis import (
    BIN_WIDTH_DEG,
    HEADING_BINS,
    ChannelHypotheses,
    HypothesisGrid,
    HypothesisLayer,
)
from way4.core.observation import Observation

ARENA = 1800.0
R_LB = 1000.0   # guaranteed lower bound of R_eff (禁止5)
R_UP = 1500.0   # upper bound of R_eff


# --- honest source + sensor model --------------------------------------------


def _rand_source(rng):
    """Uniform point in the arena disc (area-correct via sqrt)."""
    ang = rng.uniform(0, 2 * math.pi)
    rad = ARENA * math.sqrt(rng.random())
    return (rad * math.cos(ang), rad * math.sin(ang))


def _bearing(src, obs):
    return math.degrees(math.atan2(src[1] - obs[1], src[0] - obs[0])) % 360.0


def _angle_sep(a, b):
    d = abs((a % 360.0) - (b % 360.0)) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _true_bin(heading_deg):
    """The heading bin a directional source with the given antenna heading sits in."""
    return int(heading_deg % 360.0 // BIN_WIDTH_DEG)


def _detects(src, heading_deg, is_dir, scan_pt, r_eff):
    """Honest sensor: does a scan at ``scan_pt`` detect this source?

    Detection iff within the true effective range and (directional) the scan
    point lies within the ±90° arc of the antenna. ``r_eff`` is the true range,
    somewhere in [R_LB, R_UP]; the layer only knows the bound R_LB.
    """
    if math.hypot(src[0] - scan_pt[0], src[1] - scan_pt[1]) > r_eff:
        return False
    if is_dir:
        # arc: bearing(source -> scan_pt) within 90° of the antenna heading.
        b = _bearing(scan_pt, src)  # source -> scan point
        if _angle_sep(b, heading_deg) > 90.0:
            return False
    return True


def _observe(src, heading_deg, is_dir, scan_pt, r_eff, rng):
    """Produce the honest :class:`Observation` for one scan."""
    if not _detects(src, heading_deg, is_dir, scan_pt, r_eff):
        return Observation.no_signal()
    d = math.hypot(src[0] - scan_pt[0], src[1] - scan_pt[1])
    if d <= 5.0:
        return Observation.near()
    # bearing within ±1° of truth.
    true_b = _bearing(src, scan_pt)
    return Observation.bearing((true_b + rng.uniform(-1.0, 1.0)) % 360.0)


def _assert_true_hyp_alive(h, src, heading_deg, is_dir, where):
    """The true source's conservative cell (and, if directional, true heading
    bin) must still be alive."""
    i = h.grid.cell_index(src)
    assert i >= 0, f"[{where}] true source {src} fell outside every kept cell box"
    if is_dir:
        b = _true_bin(heading_deg)
        assert h.dir_alive[i, b], (
            f"[{where}] eliminated the TRUE directional hypothesis: cell={i} "
            f"bin={b} (heading={heading_deg:.1f}°), src={src}"
        )
    else:
        assert h.omni_alive[i], (
            f"[{where}] eliminated the TRUE omni hypothesis: cell={i}, src={src}"
        )


# --- §14 GATE: negative rule never kills the true directional hypothesis ------


def test_negative_rule_preserves_true_directional_cell():
    """The headline M8 property. Random legal directional sources, a hostile
    stream of NO_SIGNAL-inducing scans at the worst-case range R_LB; the true
    (cell, bin) must survive every fold."""
    rng = random.Random(20260912)
    grid = HypothesisGrid(arena_radius=ARENA, spacing=60.0)

    for _ in range(400):
        src = _rand_source(rng)
        heading = rng.uniform(0.0, 360.0)
        h = ChannelHypotheses(grid, channel=1)

        for _ in range(40):
            scan = _rand_source(rng)
            obs = _observe(src, heading, True, scan, R_LB, rng)
            # only fold the negative rule here — this test isolates it
            if obs.is_no_signal:
                h.record_no_signal(scan)
        _assert_true_hyp_alive(h, src, heading, True, "neg-directional")


def test_negative_rule_targets_behind_antenna():
    """Sharpest adversarial case: scans placed *directly behind* the antenna and
    well inside R_LB. These are genuine NO_SIGNALs (off-arc) that put maximum
    pressure on the arc test — the true bin must still survive while the layer is
    free to (soundly) eliminate the antipodal bins."""
    rng = random.Random(7)
    grid = HypothesisGrid(arena_radius=ARENA, spacing=60.0)

    for _ in range(300):
        src = _rand_source(rng)
        heading = rng.uniform(0.0, 360.0)
        h = ChannelHypotheses(grid, channel=2)
        # a ring of scan points behind the antenna (heading ± [90°,180°]).
        for _ in range(30):
            off = rng.uniform(95.0, 180.0) * rng.choice((-1.0, 1.0))
            dist = rng.uniform(50.0, R_LB - 1.0)
            ang = math.radians(heading + off)
            scan = (src[0] + dist * math.cos(ang), src[1] + dist * math.sin(ang))
            obs = _observe(src, heading, True, scan, R_LB, rng)
            if obs.is_no_signal:
                h.record_no_signal(scan)
        _assert_true_hyp_alive(h, src, heading, True, "behind-antenna")


def test_negative_rule_preserves_true_omni_cell():
    """The omni analogue: NO_SIGNAL never kills the true omni cell (worst-case
    range R_LB)."""
    rng = random.Random(101)
    grid = HypothesisGrid(arena_radius=ARENA, spacing=60.0)

    for _ in range(400):
        src = _rand_source(rng)
        h = ChannelHypotheses(grid, channel=3)
        for _ in range(40):
            scan = _rand_source(rng)
            obs = _observe(src, 0.0, False, scan, R_LB, rng)
            if obs.is_no_signal:
                h.record_no_signal(scan)
        _assert_true_hyp_alive(h, src, 0.0, False, "neg-omni")


# --- full honest stream (all observation kinds) mixed together ---------------


def test_full_honest_stream_preserves_truth():
    """End-to-end: fold every honest observation (NO_SIGNAL / BEARING / NEAR),
    directional and omni, at a true range chosen adversarially in [R_LB, R_UP].
    The true hypothesis must survive positive AND negative updates together."""
    rng = random.Random(555)
    grid = HypothesisGrid(arena_radius=ARENA, spacing=60.0)

    for t in range(500):
        is_dir = (t % 2 == 0)
        src = _rand_source(rng)
        heading = rng.uniform(0.0, 360.0)
        r_eff = rng.uniform(R_LB, R_UP)          # true range unknown to the layer
        h = ChannelHypotheses(grid, channel=4)
        for _ in range(30):
            scan = _rand_source(rng)
            obs = _observe(src, heading, is_dir, scan, r_eff, rng)
            h.record_observation(scan, obs)
        _assert_true_hyp_alive(h, src, heading, is_dir, "full-stream")


def test_near_observation_preserves_truth():
    """A NEAR return (≤5 m) is the tightest positive: the layer keeps only cells
    near the scan. The true cell contains the source within 5 m of the scan, so
    it must survive."""
    rng = random.Random(999)
    grid = HypothesisGrid(arena_radius=ARENA, spacing=60.0)
    for _ in range(300):
        src = _rand_source(rng)
        heading = rng.uniform(0.0, 360.0)
        is_dir = rng.random() < 0.5
        h = ChannelHypotheses(grid, channel=5)
        # scan within 5 m of the source, on-arc for directional.
        while True:
            off = (rng.uniform(-4.5, 4.5), rng.uniform(-4.5, 4.5))
            scan = (src[0] + off[0], src[1] + off[1])
            if math.hypot(off[0], off[1]) <= 5.0:
                break
        obs = _observe(src, heading, is_dir, scan, R_LB, rng)
        # by construction within 5 m ⇒ NEAR (arc irrelevant at ≤5 m in our model
        # only if on-arc; if the model returned NO_SIGNAL because off-arc, that's
        # still an honest negative and must also preserve truth).
        h.record_observation(scan, obs)
        _assert_true_hyp_alive(h, src, heading, is_dir, "near")


# --- structural invariants (Invariant B: never certifies absence) -------------


def test_layer_exposes_no_absence_certificate():
    """H_c must never expose a completion / absence certificate (Invariant B,
    禁止6). Only counts, fractions and gains are permitted."""
    h = ChannelHypotheses(HypothesisGrid(), channel=0)
    forbidden = ("is_complete", "is_absent", "certify", "mark_absent",
                 "mark_cleared", "is_cleared", "absent")
    for name in forbidden:
        assert not hasattr(h, name), f"H_c must not expose {name!r} (Invariant B)"


def test_monotone_elimination_and_gain_bounds():
    """Alive counts never increase; elimination_gain ∈ [0,1] and predicts the
    actual drop of a NO_SIGNAL fold exactly."""
    rng = random.Random(3)
    grid = HypothesisGrid(arena_radius=ARENA, spacing=60.0)
    h = ChannelHypotheses(grid, channel=7)
    prev = h.n_alive
    for _ in range(25):
        scan = _rand_source(rng)
        g = h.elimination_gain(scan)
        assert 0.0 <= g <= 1.0
        before = h.n_alive
        predicted = int(round(g * before))
        h.record_no_signal(scan)
        after = h.n_alive
        assert after <= prev                     # monotone
        assert (before - after) == predicted     # gain is exact
        prev = after


def test_grid_covers_every_legal_source():
    """Every legal in-arena point lands inside some kept cell's box — the
    property the corner-based soundness proof relies on at the rim."""
    rng = random.Random(42)
    grid = HypothesisGrid(arena_radius=ARENA, spacing=60.0)
    for _ in range(5000):
        p = _rand_source(rng)
        assert grid.cell_index(p) >= 0, f"{p} not covered by any cell box"
    # explicit rim stress: points just inside the boundary at many angles.
    for k in range(360):
        a = math.radians(k)
        p = ((ARENA - 1e-3) * math.cos(a), (ARENA - 1e-3) * math.sin(a))
        assert grid.cell_index(p) >= 0
    assert 2000 <= grid.n <= 3600            # ~2800 in-arena cells (§3.1)


def test_layer_multichannel_isolation():
    """Per-channel masks are independent; a scan on one channel never touches
    another."""
    layer = HypothesisLayer(arena_radius=ARENA, spacing=60.0)
    layer.record_observation(1, (0.0, 0.0), Observation.no_signal())
    assert layer.channel(1).n_alive < layer.channel(2).n_alive
    assert layer.channel(2).n_alive == grid_full(layer)


def grid_full(layer):
    n = layer.grid.n
    return n * (1 + HEADING_BINS)
