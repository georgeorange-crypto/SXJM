"""P0 #4/#5/#7 — unified remaining-task route + subadditive joint FutureCost.

These pin the *ranking* upgrade behind ``planner=spatial`` (the ``joint_route`` flag on
``FutureCostEstimator``). Like Phase C's object work, they are unit-level and touch no
pipeline — the flag is selected by ``planner_mode`` once the shared ``pipeline.py`` is
free. What is pinned:

  * default flag OFF => ``estimate`` dispatches to the *byte-identical* additive path
    (legacy full-clear provably unchanged).
  * the joint estimate NEVER exceeds the additive one (the subadditive cap) — Phase D
    must never score a plan as worse than legacy would.
  * when tasks are co-located far from the robot, the joint route pays the trip out
    ONCE and beats the additive sum (this is what makes a ``SpatialStop`` bundle win).
  * when tasks are on opposite sides (join does not help), the joint estimate falls
    back to the exact additive result.
  * ``RouteEstimator.insertion_delta`` — the ΔL = d(a,c)+d(c,b)-d(a,b) marginal cost of
    folding a clear into a route (#5): ~0 when the point already lies on the way.
"""

from math import hypot

import pytest

from way4.planner.future_cost import CostView, DetectedRegion, FutureCostEstimator
from way4.routing import RouteEstimator


def _fce(joint: bool) -> FutureCostEstimator:
    # fixed kinematics so the arithmetic below is exact
    return FutureCostEstimator(
        speed=5.0, measure_s=5.0, clear_hit_s=5.0, detection_radius=1000.0, joint_route=joint
    )


# --- dispatch: default flag off is the legacy additive path --------------------


def test_default_flag_is_off_and_dispatches_to_additive():
    fce = FutureCostEstimator()
    assert fce.joint_route is False
    view = CostView(
        pos=(0.0, 0.0),
        clearable_targets=[(300.0, 0.0), (400.0, 100.0)],
        detected=[DetectedRegion((800.0, 0.0), 200.0, 150.0, 0.5)],
        unknown_holes=[(500.0, 0.0)],
        n_unknown=1,
        anchors=[(500.0, 0.0)],
    )
    got = fce.estimate(view)
    exp = fce._estimate_additive(view)
    assert (got.total, got.j_route, got.j_localization, got.j_exploration,
            got.j_certificate, got.n_cover_stops) == (
        exp.total, exp.j_route, exp.j_localization, exp.j_exploration,
        exp.j_certificate, exp.n_cover_stops)


# --- subadditive cap: joint never exceeds additive -----------------------------


def _views():
    return [
        # co-located clearable + cover stop, far from pos
        CostView((0.0, 0.0), [(1000.0, 0.0)], [], [(1010.0, 0.0)], 1, [(1010.0, 0.0)]),
        # opposite sides (join cannot help)
        CostView((0.0, 0.0), [(1000.0, 0.0)], [], [(-1000.0, 0.0)], 1, [(-1000.0, 0.0)]),
        # a mix with a detected region
        CostView((0.0, 0.0), [(600.0, 0.0), (650.0, 20.0)],
                 [DetectedRegion((680.0, 0.0), 150.0, 120.0, 0.5)],
                 [(640.0, 0.0)], 1, [(640.0, 0.0)]),
        # nothing to do
        CostView((0.0, 0.0), [], [], [], 0, []),
    ]


def test_joint_never_exceeds_additive():
    add, joint = _fce(False), _fce(True)
    for v in _views():
        assert joint.estimate(v).total <= add.estimate(v).total + 1e-9


# --- co-location is credited (the bundle-wins property) ------------------------


def test_joint_credits_colocation():
    add, joint = _fce(False), _fce(True)
    v = CostView((0.0, 0.0), [(1000.0, 0.0)], [], [(1010.0, 0.0)], 1, [(1010.0, 0.0)])
    a = add.estimate(v)
    j = joint.estimate(v)
    # additive re-pays pos->cluster for BOTH the clear tour and the cover tour
    # (2010 m); the joint route pays it once + 10 m intra-cluster (1010 m). Dwells
    # (one clear + one measure) are identical, so the whole gap is travel: 1000 m / 5.
    assert j.total < a.total
    assert j.total == pytest.approx(a.total - 1000.0 / 5.0)


# --- disjoint tasks fall back to the exact additive result ---------------------


def test_joint_disjoint_falls_back_to_additive():
    add, joint = _fce(False), _fce(True)
    v = CostView((0.0, 0.0), [(1000.0, 0.0)], [], [(-1000.0, 0.0)], 1, [(-1000.0, 0.0)])
    a = add.estimate(v)
    j = joint.estimate(v)
    # the cap binds: the single tour across pos is longer than two start-rooted tours,
    # so the joint mode returns the additive object unchanged (j_certificate kept > 0,
    # which the joint decomposition would have zeroed).
    assert j.total == pytest.approx(a.total)
    assert j.j_certificate == pytest.approx(a.j_certificate)
    assert j.j_certificate > 0.0


def test_joint_empty_view_is_zero_both_modes():
    v = CostView((0.0, 0.0), [], [], [], 0, [])
    assert _fce(False).estimate(v).total == 0.0
    assert _fce(True).estimate(v).total == 0.0


# --- insertion delta ΔL (#5) ---------------------------------------------------


def test_insertion_delta_zero_when_on_the_way():
    d = RouteEstimator.insertion_delta([(0.0, 0.0), (100.0, 0.0)], (50.0, 0.0))
    assert d == pytest.approx(0.0, abs=1e-9)


def test_insertion_delta_offside_matches_formula():
    d = RouteEstimator.insertion_delta([(0.0, 0.0), (100.0, 0.0)], (50.0, 50.0))
    # between (0,0) and (100,0): d(a,c)+d(c,b)-d(a,b) = 2*hypot(50,50) - 100
    assert d == pytest.approx(2.0 * hypot(50.0, 50.0) - 100.0)


def test_insertion_delta_appends_past_the_tail():
    # a point beyond the last node is cheapest appended: d(last, c) = 50
    d = RouteEstimator.insertion_delta([(0.0, 0.0), (100.0, 0.0)], (150.0, 0.0))
    assert d == pytest.approx(50.0)


def test_insertion_delta_empty_route_is_zero():
    assert RouteEstimator.insertion_delta([], (10.0, 10.0)) == 0.0


def test_insertion_delta_never_negative():
    route = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    for p in [(50.0, 50.0), (-30.0, 10.0), (200.0, 200.0), (100.0, 50.0)]:
        assert RouteEstimator.insertion_delta(route, p) >= -1e-9
