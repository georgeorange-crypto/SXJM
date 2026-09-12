"""Route decomposition diagnostics (P0 #9) — way4.metrics.

Unit tests for the pure ``compute_route_metrics`` function (no env needed) plus a
pipeline-level consistency check that the six fields land on ``EpisodeResult`` and
agree with the primitive counts. These metrics drive nothing, so the point is only
that they decompose a realised route correctly and never affect correctness.
"""

import math

import pytest

from way4.metrics import (
    ROLE_CLEAR,
    ROLE_COVERAGE,
    ROLE_LOCALIZE,
    RouteEvent,
    compute_route_metrics,
)


def _cov(loc):
    return RouteEvent(loc=loc, kind="MEASURE", channel=1, role=ROLE_COVERAGE)


def _loc(loc, ch=2):
    return RouteEvent(loc=loc, kind="MEASURE", channel=ch, role=ROLE_LOCALIZE)


def _clr(loc, ch=3, cleared=True):
    return RouteEvent(loc=loc, kind="CLEAR", channel=ch, role=ROLE_CLEAR, cleared=cleared)


def test_empty_route_is_all_zero():
    m = compute_route_metrics([], (0.0, 0.0))
    assert m.n_stops == 0 and m.n_services == 0
    assert m.services_per_stop == 0.0
    assert m.pure_refine_travel == 0.0
    assert m.certificate_only_travel == 0.0
    assert m.revisit_distance == 0.0
    assert m.shared_stop_ratio == 0.0
    assert m.clear_insertion_delta == 0.0


def test_batch_coverage_scan_is_one_shared_stop():
    # three coverage measures at one waypoint = one stop with three services.
    events = [_cov((100.0, 0.0)), _cov((100.0, 0.0)), _cov((100.0, 0.0))]
    m = compute_route_metrics(events, (0.0, 0.0))
    assert m.n_stops == 1 and m.n_services == 3
    assert m.services_per_stop == pytest.approx(3.0)
    assert m.shared_stop_ratio == pytest.approx(1.0)
    # single-purpose coverage stop: its whole leg-in counts as certificate-only travel.
    assert m.certificate_only_travel == pytest.approx(100.0)
    assert m.pure_refine_travel == 0.0
    assert m.clear_insertion_delta == 0.0
    assert m.revisit_distance == 0.0


def test_pure_localize_stop_counts_as_refine_travel():
    m = compute_route_metrics([_loc((200.0, 0.0))], (0.0, 0.0))
    assert m.pure_refine_travel == pytest.approx(200.0)
    assert m.certificate_only_travel == 0.0
    assert m.shared_stop_ratio == 0.0        # a single service is not "shared"
    assert m.services_per_stop == pytest.approx(1.0)


def test_opportunistic_clear_bundles_and_costs_no_detour():
    # a near-return clear shares the scan waypoint: one stop, two services, and the
    # clear adds no insertion delta (P0 #5: the good case).
    events = [_loc((50.0, 0.0)), _clr((50.0, 0.0))]
    m = compute_route_metrics(events, (0.0, 0.0))
    assert m.n_stops == 1 and m.n_services == 2
    assert m.shared_stop_ratio == pytest.approx(1.0)
    assert m.clear_insertion_delta == 0.0     # not a dedicated clear stop
    # mixed-role stop counts toward neither single-purpose travel bucket.
    assert m.pure_refine_travel == 0.0
    assert m.certificate_only_travel == 0.0


def test_dedicated_clear_stop_charges_its_detour():
    # start -> A (coverage) -> C (standalone clear) -> B (coverage).
    A, C, B = (100.0, 0.0), (100.0, 100.0), (200.0, 0.0)
    events = [_cov(A), _clr(C), _cov(B)]
    m = compute_route_metrics(events, (0.0, 0.0))
    assert m.n_stops == 3
    expected = (
        math.dist(A, C) + math.dist(C, B) - math.dist(A, B)
    )
    assert m.clear_insertion_delta == pytest.approx(expected)
    assert expected > 0.0                     # a real detour, the legacy waste


def test_terminal_dedicated_clear_is_one_way_detour():
    A, C = (100.0, 0.0), (100.0, 100.0)
    events = [_cov(A), _clr(C)]               # clear is the final stop
    m = compute_route_metrics(events, (0.0, 0.0))
    assert m.clear_insertion_delta == pytest.approx(math.dist(A, C))


def test_revisit_distance_flags_backtracking():
    # A -> far B -> back near A (within revisit_tol): the leg back is revisit waste.
    A, B, A2 = (100.0, 0.0), (500.0, 0.0), (100.0, 10.0)
    events = [_cov(A), _cov(B), _cov(A2)]
    m = compute_route_metrics(events, (0.0, 0.0), revisit_tol=50.0)
    assert m.n_stops == 3
    assert m.revisit_distance == pytest.approx(math.dist(B, A2))
    # A and B themselves are not revisits (first occurrences).


def test_stop_tol_clusters_a_small_neighbourhood():
    # two services 3 m apart (< stop_tol=5) collapse into one stop.
    events = [_cov((100.0, 0.0)), _cov((103.0, 0.0))]
    m = compute_route_metrics(events, (0.0, 0.0), stop_tol=5.0)
    assert m.n_stops == 1 and m.n_services == 2
    # and split into two stops when the tolerance is tightened.
    m2 = compute_route_metrics(events, (0.0, 0.0), stop_tol=1.0)
    assert m2.n_stops == 2


# --------------------------------------------------------------------------- #
# pipeline-level: the six fields land on EpisodeResult and stay self-consistent
# --------------------------------------------------------------------------- #
@pytest.fixture
def way3_env():
    from way4.executor import load_way3_environment

    mod = load_way3_environment()
    if mod is None:
        pytest.skip("Way3 environment.py not available on disk")
    return mod


def test_pipeline_populates_route_metrics(way3_env):
    from way4.executor import Way3EngineAdapter
    from way4.pipeline import Way4Pipeline

    Jammer, Case = way3_env.Jammer, way3_env.Case
    case = Case(
        [Jammer(1, 600.0, 0.0, 1400.0, "omni"), Jammer(2, -600.0, 300.0, 1400.0, "omni")],
        field=way3_env.ConstantField(1.0),
    )
    eng = way3_env.Engine(case)
    eng.enter()
    result = Way4Pipeline(Way3EngineAdapter(eng), n_channels=20, max_steps=4000).run()

    assert result.error is None
    # the decomposition saw exactly the measured + cleared primitives.
    assert result.route_n_services == result.n_measure + result.n_clear
    assert result.route_n_stops >= 1
    assert result.services_per_stop == pytest.approx(
        result.route_n_services / result.route_n_stops
    )
    assert 0.0 <= result.shared_stop_ratio <= 1.0
    # all six diagnostics are finite and non-negative.
    for v in (
        result.pure_refine_travel,
        result.certificate_only_travel,
        result.revisit_distance,
        result.clear_insertion_delta,
    ):
        assert math.isfinite(v) and v >= 0.0
