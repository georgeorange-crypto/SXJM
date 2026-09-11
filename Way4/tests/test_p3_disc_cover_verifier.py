"""P3 hard disc-cover verifier tests (Way4/DESIGN.md §15).

The overriding goal is to hunt FALSE POSITIVES: the verifier may be as
conservative as it likes, but ``is_covered() is True`` must imply the arena disc
is genuinely covered. Monte-Carlo witness tests enforce exactly that.
"""

import math
import random

from way4.certificate.hard_disc_cover import (
    CoverageCell,
    HardDiscCoverVerifier,
    cell_fully_covered_by_disc,
    distance,
)

ARENA = 1800.0
R = 1000.0


def covered_by_union(p, pts, r=R):
    return any(distance(p, s) <= r for s in pts)


def make_grid(spacing, lim):
    """Symmetric grid of points at multiples of ``spacing`` within [-lim, lim]^2."""
    n = int(round(lim / spacing))
    coords = [i * spacing for i in range(-n, n + 1)]
    return [(float(x), float(y)) for x in coords for y in coords]


def sample_arena(rng):
    """Uniform point in the arena disc D(0, ARENA)."""
    ang = rng.uniform(0.0, 2.0 * math.pi)
    rad = ARENA * math.sqrt(rng.random())
    return (rad * math.cos(ang), rad * math.sin(ang))


# --- structural / boolean cases ------------------------------------------------


def test_empty_scan_set_is_not_complete():
    assert HardDiscCoverVerifier().is_covered([]) is False


def test_single_disc_cannot_cover_arena():
    # One 1000 m disc cannot cover an 1800 m arena.
    assert HardDiscCoverVerifier(min_cell_size=100).is_covered([(0.0, 0.0)]) is False


def test_dense_grid_covers():
    # A 400 m grid out to +/-2000 (covers the boundary rind, Conservatism note B).
    v = HardDiscCoverVerifier(min_cell_size=100)
    assert v.is_covered(make_grid(400, 2000)) is True


def test_center_hole_is_not_complete():
    # Remove every scan point within 1300 m of the origin -> the centre is uncovered.
    v = HardDiscCoverVerifier(min_cell_size=100)
    pts = [p for p in make_grid(400, 2000) if distance(p, (0.0, 0.0)) > 1300.0]
    assert v.is_covered(pts) is False


def test_offcenter_hole_is_not_complete():
    # Remove every disc within 1000 m of an off-center target: that target then
    # has no disc within radius, so it is genuinely uncovered -> verifier False.
    # (Removing a single disc from this dense grid would NOT make a hole: the
    # 400 m-spaced neighbours still cover the spot.)
    v = HardDiscCoverVerifier(min_cell_size=50)
    full = make_grid(400, 2000)
    target = (1000.0, 0.0)
    pts = [p for p in full if distance(p, target) > R]
    assert not covered_by_union(target, pts)  # sanity: the test really makes a hole
    assert v.is_covered(pts) is False


def test_permutation_invariant():
    v = HardDiscCoverVerifier(min_cell_size=150)
    pts = make_grid(500, 2000)
    base = v.is_covered(pts)
    shuffled = pts[:]
    random.Random(0).shuffle(shuffled)
    assert v.is_covered(shuffled) == base


def test_superset_monotonicity():
    # S1 subset S2 and is_covered(S1) => is_covered(S2). Adding discs never uncovers.
    v = HardDiscCoverVerifier(min_cell_size=150)
    rng = random.Random(1)
    base = make_grid(400, 2000)
    assert v.is_covered(base) is True
    extra = base + [(rng.uniform(-1800, 1800), rng.uniform(-1800, 1800)) for _ in range(30)]
    assert v.is_covered(extra) is True


# --- soundness: NO FALSE POSITIVE ---------------------------------------------


def test_no_false_positive_on_perturbed_grids():
    """Randomly drop ~10% of a comfortably-covering grid; whenever the verifier
    still reports covered, every sampled arena point must truly be union-covered."""
    v = HardDiscCoverVerifier(min_cell_size=100)
    rng = random.Random(42)
    base = make_grid(400, 2000)
    exercised_true = 0
    for _ in range(30):
        pts = [p for p in base if rng.random() > 0.10]
        if v.is_covered(pts):
            exercised_true += 1
            for _ in range(2000):
                q = sample_arena(rng)
                assert covered_by_union(q, pts), f"FALSE POSITIVE: {q} uncovered"
    assert exercised_true > 0, "test never exercised a positive verdict"


def test_no_false_positive_known_true_config():
    v = HardDiscCoverVerifier(min_cell_size=100)
    pts = make_grid(400, 2000)
    assert v.is_covered(pts) is True
    rng = random.Random(7)
    for _ in range(20000):
        q = sample_arena(rng)
        assert covered_by_union(q, pts), f"FALSE POSITIVE: {q} uncovered"


def test_adversarial_uncovered_witness_forces_false():
    """Plant a legal arena point far from every scan point; verifier must say False."""
    v = HardDiscCoverVerifier(min_cell_size=50)
    rng = random.Random(9)
    for _ in range(25):
        witness = sample_arena(rng)
        # scan points all >= ~1050 m away from the witness
        pts = []
        while len(pts) < 60:
            q = (rng.uniform(-2000, 2000), rng.uniform(-2000, 2000))
            if distance(q, witness) > 1050.0:
                pts.append(q)
        assert v.is_covered(pts) is False


# --- helper ------------------------------------------------------------------


def test_cell_fully_covered_by_disc_helper():
    cell = CoverageCell(-10.0, 10.0, -10.0, 10.0, 0)  # half-diagonal ~14.14
    assert cell_fully_covered_by_disc(cell, (0.0, 0.0), radius=20.0) is True
    assert cell_fully_covered_by_disc(cell, (0.0, 0.0), radius=14.0) is False
