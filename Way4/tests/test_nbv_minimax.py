"""M5 acceptance — minimax NBV (DESIGN.md §8, §14 M5).

Headline acceptance (§14 M5): "随机单源下 NBV 后 region diameter 显著优于随机第二点."
We draw random single sources, take a first noisy bearing from the origin, then
compare the realised feasible-region diameter after (a) the NBV-recommended second
viewpoint and (b) a random *still-detecting* second viewpoint. The random baseline
is deliberately favourable (it always detects the source), so any win is due to
NBV's crossing-angle geometry, not detection luck.

Plus unit tests locking the minimax objective, the perpendicular-beats-collinear
property, and the out-of-range guard (a far viewpoint must NOT score as ideal).
"""

import math
import random

import pytest

from sxjm_core.geometry import bearing_deg, dist, norm_deg, polygon_diameter
from way4.belief import ChannelBelief
from way4.sensing import MinimaxNBV

RANGE = 1500.0


def _first_bearing_belief(apex, svd):
    b = ChannelBelief(channel=1)
    b.record_bearing(apex, svd)
    return b


def _realised_diameter(first_apex, first_svd, q, source, noise2):
    """Diameter after a second measurement from ``q`` (detect if in range)."""
    b = _first_bearing_belief(first_apex, first_svd)
    if dist(q, source) <= RANGE:
        b.record_bearing(q, norm_deg(bearing_deg(q, source) + noise2))
    else:
        b.record_no_signal(q)   # out of range -> no bearing, sliver unchanged
    return b.diameter


def test_nbv_beats_random_second_point_on_region_diameter():
    rng = random.Random(20260911)
    nbv = MinimaxNBV(lambda_t=0.0)
    apex = (0.0, 0.0)

    nbv_diams, rand_diams, wins = [], [], 0
    n_trials = 40
    for _ in range(n_trials):
        # a random detectable source (within range of the origin), off-centre
        rho = rng.uniform(300.0, 1400.0)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        source = (rho * math.cos(phi), rho * math.sin(phi))

        noise1 = rng.uniform(-1.0, 1.0)
        first_svd = norm_deg(bearing_deg(apex, source) + noise1)

        # NBV's choice, realised with fresh measurement noise
        res = nbv.choose(_first_bearing_belief(apex, first_svd), robot_pos=apex)
        assert res is not None
        d_nbv = _realised_diameter(apex, first_svd, res.point, source, rng.uniform(-1.0, 1.0))

        # a favourable random second point: uniformly in the disc around the source
        # (so it always detects) — isolates crossing-angle quality
        rr = RANGE * math.sqrt(rng.random())
        ra = rng.uniform(0.0, 2.0 * math.pi)
        q_rand = (source[0] + rr * math.cos(ra), source[1] + rr * math.sin(ra))
        d_rand = _realised_diameter(apex, first_svd, q_rand, source, rng.uniform(-1.0, 1.0))

        nbv_diams.append(d_nbv)
        rand_diams.append(d_rand)
        wins += 1 if d_nbv <= d_rand + 1e-6 else 0

    mean_nbv = sum(nbv_diams) / n_trials
    mean_rand = sum(rand_diams) / n_trials
    # "显著优于": NBV shrinks the region far more on average and wins most trials.
    assert mean_nbv < 0.6 * mean_rand, (mean_nbv, mean_rand)
    assert wins >= int(0.7 * n_trials), (wins, n_trials)


def test_worst_case_shrinks_a_fresh_sliver():
    """On a single-bearing sliver, the NBV viewpoint's worst-case diameter is
    strictly below the current (unrefined) diameter — a real triangulation gain."""
    nbv = MinimaxNBV(lambda_t=0.0)
    b = _first_bearing_belief((0.0, 0.0), 30.0)
    res = nbv.choose(b, robot_pos=(0.0, 0.0))
    assert res is not None
    assert res.improved is True
    assert res.worst_case_diameter < res.current_diameter
    assert res.expected_shrink > 0.0


def test_perpendicular_beats_collinear_viewpoint():
    """A perpendicular standoff gives a shorter worst-case region than a viewpoint
    almost collinear with the first bearing (the crux NBV is built to exploit).

    Both viewpoints sit inside the *guaranteed* detection radius (≤1000 m) of the
    whole region, so the comparison isolates crossing-angle quality — not the
    range guard (which would tie both at the full diameter, see the guard test)."""
    nbv = MinimaxNBV(lambda_t=0.0)
    b = _first_bearing_belief((0.0, 0.0), 0.0)   # first LOS points east (+x)
    poly = b.F_c
    hyps = nbv.representative_hypotheses(poly)
    current = polygon_diameter(poly)

    # region spans x∈[0,1500] along the axis; view its centre (750,0) from
    # on-axis (collinear, poor cut) vs from the side (perpendicular, good cut).
    collinear = (750.0, 0.0)        # on the +x axis -> wedge ~parallel to the sliver
    perpendicular = (750.0, 600.0)  # broadside -> ~90° crossing
    u_col = nbv.worst_case_diameter(poly, collinear, hyps, current)
    u_perp = nbv.worst_case_diameter(poly, perpendicular, hyps, current)
    assert u_perp < u_col


def test_out_of_range_viewpoint_is_not_falsely_ideal():
    """The out-of-range guard: a viewpoint 5 km away (sees nothing) must score at
    the full current diameter, NOT ~0 — otherwise NBV would flee the arena."""
    nbv = MinimaxNBV(lambda_t=0.0)
    b = _first_bearing_belief((0.0, 0.0), 45.0)
    poly = b.F_c
    hyps = nbv.representative_hypotheses(poly)
    current = polygon_diameter(poly)

    far = (5000.0, 5000.0)
    u_far = nbv.worst_case_diameter(poly, far, hyps, current)
    assert u_far == pytest.approx(current, rel=1e-9)

    # and the chosen point is never that far-away non-informative one
    res = nbv.choose(b, robot_pos=(0.0, 0.0))
    assert res.worst_case_diameter < current


def test_guard_uses_guaranteed_detection_radius_not_the_optimistic_upper_bound():
    """A viewpoint is credited a shrink only where a detection is GUARANTEED — the
    hypothesis within the R_eff *lower* bound (1000 m), not merely within the 1500 m
    upper bound. Same geometry, same hypothesis: a 900 m standoff cuts the sliver, a
    1200 m one (in range under 1500 but not guaranteed) is scored as NO_SIGNAL. This
    is what stops the planner betting on a far standoff whose true R_eff (∈[1000,1500])
    might return only NO_SIGNAL — the zero-progress re-scan behind the seed-1017 stall."""
    nbv = MinimaxNBV(lambda_t=0.0)
    b = _first_bearing_belief((0.0, 0.0), 0.0)   # east sliver, x∈[0,1500]
    poly = b.F_c
    current = polygon_diameter(poly)
    h = (1000.0, 0.0)                            # a source hypothesis on the sliver

    guaranteed = nbv.post_measurement_diameter(poly, (1000.0, 900.0), h, current)
    uncertain = nbv.post_measurement_diameter(poly, (1000.0, 1200.0), h, current)
    assert guaranteed < current - 1e-9           # within 1000 m -> real cut credited
    assert uncertain == pytest.approx(current, rel=1e-9)   # 1200 m -> not credited
    assert nbv.detect_lower_bound == 1000.0


def test_choose_returns_none_when_nothing_to_refine():
    nbv = MinimaxNBV()
    assert nbv.choose(ChannelBelief(channel=1), robot_pos=(0.0, 0.0)) is None


# --- P0-A: NO_SIGNAL prunes the planning hypothesis set -----------------------


def test_choose_prunes_hypotheses_ruled_out_by_no_signal():
    """A NO_SIGNAL disc that provably rules out part of F_c drops those hypotheses
    from the minimax max_j, so NBV no longer plans against impossible sources. The
    result is still a valid REFINE (planning only — belief F_c / mec unchanged)."""
    nbv = MinimaxNBV(lambda_t=0.0)
    b = _first_bearing_belief((0.0, 0.0), 0.0)   # east sliver x∈[0,1500]
    poly_before = list(b.F_c)
    mec_before = (b.mec_center, b.mec_radius)

    res_no_disc = nbv.choose(b, robot_pos=(0.0, 0.0))
    assert res_no_disc is not None

    # add a NO_SIGNAL far out on +x: rules out the far end of the sliver.
    b.record_no_signal((2600.0, 0.0))
    res = nbv.choose(b, robot_pos=(0.0, 0.0))
    assert res is not None
    # the hypotheses NBV actually optimised against are all still-possible sources
    assert all(not b.excludes(h) for h in res.hypotheses)
    # belief geometry is untouched by the planning-time pruning (§3.3, 禁令)
    assert list(b.F_c) == poly_before
    assert (b.mec_center, b.mec_radius) == mec_before


def test_choose_falls_back_when_all_hypotheses_excluded():
    """If (numeric edge) every hypothesis is inside a disc, NBV must still return a
    viewpoint using the unfiltered set — never crash or act on an empty pool."""
    from way4.belief.channel import ExclusionDisc
    nbv = MinimaxNBV(lambda_t=0.0)
    b = _first_bearing_belief((0.0, 0.0), 0.0)
    b.negative_discs.append(ExclusionDisc((0.0, 0.0), 1.0e7))  # swallows everything
    res = nbv.choose(b, robot_pos=(0.0, 0.0))
    assert res is not None
    assert len(res.hypotheses) > 0


def test_hypotheses_capped():
    nbv = MinimaxNBV(n_hypotheses=8)
    b = _first_bearing_belief((0.0, 0.0), 10.0)
    hyps = nbv.representative_hypotheses(b.F_c)
    assert 0 < len(hyps) <= 8
