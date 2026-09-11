"""§10 FutureCostEstimator — the four-term analytical cost-to-go ``Ĵ(B)``.

Locks each term's meaning on directly-built ``CostView``s (no belief needed), the
additive total, the greedy hole set-cover's stop count, and the ``build_cost_view``
snapshot of a live belief/certificate. All terms are seconds.
"""

import pytest

from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager
from way4.core import RobotState
from way4.planner import (
    CostView,
    DetectedRegion,
    FutureCostEstimator,
    build_cost_view,
)
from way4.routing import LocalizationCostModel


def test_empty_view_is_zero():
    fc = FutureCostEstimator().estimate(CostView(pos=(0.0, 0.0)))
    assert fc.total == 0.0
    assert (fc.j_route, fc.j_localization, fc.j_exploration, fc.j_certificate) == (0, 0, 0, 0)


def test_j_route_counts_travel_and_clear_dwell():
    fce = FutureCostEstimator(speed=5.0, clear_hit_s=5.0)
    v1 = CostView(pos=(0.0, 0.0), clearable_targets=[(500.0, 0.0)])
    fc1 = fce.estimate(v1)
    assert fc1.j_route == pytest.approx(500.0 / 5.0 + 5.0)          # 100 s travel + 5 s clear
    assert fc1.total == pytest.approx(fc1.j_route)                  # nothing else pending

    v2 = CostView(pos=(0.0, 0.0), clearable_targets=[(500.0, 0.0), (500.0, 100.0)])
    assert fce.estimate(v2).j_route > fc1.j_route                   # more targets cost more


def test_j_localization_uses_fitted_cost_and_is_monotone():
    loc = LocalizationCostModel(a0=10.0, a1=0.5, a2=0.2, a3=30.0)
    fce = FutureCostEstimator(loc_model=loc, speed=5.0, clear_hit_s=5.0)
    # region centred on the robot -> zero approach, so only L̂ + clear varies
    small = CostView(pos=(0.0, 0.0), detected=[DetectedRegion((0.0, 0.0), 100.0, 200.0, 1.0)])
    big = CostView(pos=(0.0, 0.0), detected=[DetectedRegion((0.0, 0.0), 300.0, 200.0, 1.0)])
    j_small = fce.estimate(small).j_localization
    j_big = fce.estimate(big).j_localization
    assert j_small == pytest.approx(loc.estimate(100.0, 200.0, 1.0) + 5.0)
    assert j_big > j_small                                          # larger MEC => costlier


def test_j_certificate_greedy_cover_stop_count():
    fce = FutureCostEstimator(detection_radius=1000.0, measure_s=5.0)
    anchors = [(0.0, 0.0), (3000.0, 0.0)]
    holes_one = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]           # only (0,0) covers these
    holes_two = holes_one + [(3000.0, 0.0), (2900.0, 0.0)]         # second cluster needs (3000,0)

    one = fce.estimate(CostView((0.0, 0.0), unknown_holes=holes_one, n_unknown=1, anchors=anchors))
    two = fce.estimate(CostView((0.0, 0.0), unknown_holes=holes_two, n_unknown=1, anchors=anchors))
    assert one.n_cover_stops == 1
    assert two.n_cover_stops == 2
    assert two.j_exploration == pytest.approx(5.0 * 2)             # dwell per stop
    assert two.j_certificate > 0.0                                # travel to the far anchor


def test_j_certificate_zero_when_no_unknowns_or_no_holes():
    fce = FutureCostEstimator()
    assert fce.estimate(CostView((0.0, 0.0), unknown_holes=[(0.0, 0.0)], n_unknown=0,
                                 anchors=[(0.0, 0.0)])).j_certificate == 0.0
    assert fce.estimate(CostView((0.0, 0.0), unknown_holes=[], n_unknown=3,
                                 anchors=[(0.0, 0.0)])).j_certificate == 0.0


def test_greedy_cover_skips_useless_anchors():
    fce = FutureCostEstimator(detection_radius=1000.0)
    anchors = [(0.0, 0.0), (9999.0, 9999.0)]                      # second covers nothing
    holes = [(0.0, 0.0), (300.0, 0.0)]
    chosen = fce._greedy_cover(holes, anchors)
    assert chosen == [(0.0, 0.0)]


def test_total_is_weighted_sum_of_parts():
    fce = FutureCostEstimator()
    v = CostView(
        pos=(1000.0, 0.0),
        clearable_targets=[(0.0, 0.0)],
        detected=[DetectedRegion((500.0, 500.0), 200.0, 400.0, 0.5)],
        unknown_holes=[(0.0, 0.0), (100.0, 0.0)],
        n_unknown=2,
        anchors=[(0.0, 0.0)],
    )
    fc = fce.estimate(v)
    assert fc.total == pytest.approx(fc.j_route + fc.j_localization + fc.j_exploration + fc.j_certificate)


def test_build_cost_view_snapshots_belief():
    belief = BeliefState(n_channels=4)
    cert = CertificateManager(n_channels=4)
    belief[2].record_near((0.0, 0.0))                 # clearable
    belief[3].record_bearing((0.0, 0.0), 30.0)        # detected
    # ch1, ch4 remain UNKNOWN
    state = RobotState(100.0, 0.0)

    v = build_cost_view(belief, cert, state)
    assert v.pos == (100.0, 0.0)
    assert len(v.clearable_targets) == 1
    assert len(v.detected) == 1 and v.detected[0].r_mec == pytest.approx(belief[3].mec_radius)
    assert v.n_unknown == 2
    assert v.unknown_holes                            # fresh UNKNOWN channels => holes exist
    assert v.anchors                                  # fallback template present
