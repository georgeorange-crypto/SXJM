"""M6 acceptance — routing (DESIGN.md §9, §14 M6).

Headline (§14 M6): "Held–Karp ... 小 n 对拍 brute-force 一致". We check the exact
open-TSP kernel against the permutation oracle on many random instances, both
euclidean and asymmetric (the TSPN metric is asymmetric). Plus: the set-tour cache
re-solves only when the clearable set changes (§9 工程约束), the TSPN approach
metric honours ``d_N = max(0, ‖x−m‖−r)``, and the offline localisation-cost model
is monotone in the right variables.
"""

import random

import pytest

from way4.routing import (
    LocalizationCostModel,
    Neighborhood,
    guaranteed_clear_point,
    RouteEstimator,
    brute_force_open,
    held_karp_min_path,
    held_karp_open,
    nearest_neighbor_open,
    two_opt_open,
)


def _rand_points(rng, n, span=1000.0):
    return [(rng.uniform(-span, span), rng.uniform(-span, span)) for _ in range(n)]


def test_held_karp_matches_brute_force_euclidean():
    rng = random.Random(6006)
    est = RouteEstimator()
    for _ in range(120):
        n = rng.randint(1, 7)
        start = (rng.uniform(-1000, 1000), rng.uniform(-1000, 1000))
        pts = _rand_points(rng, n)
        hk = est.optimal_clear_order(start, pts)
        bf = est.brute_force_clear_order(start, pts)
        assert hk.length == pytest.approx(bf.length, rel=1e-9, abs=1e-6)


def test_held_karp_matches_brute_force_asymmetric():
    """Held–Karp must be correct for asymmetric costs too (the TSPN metric is)."""
    rng = random.Random(4242)
    for _ in range(80):
        n = rng.randint(1, 7)
        start_cost = [rng.uniform(1.0, 100.0) for _ in range(n)]
        cost = [[0.0 if i == j else rng.uniform(1.0, 100.0) for j in range(n)] for i in range(n)]
        _, hk_len = held_karp_open(start_cost, cost)
        _, bf_len = brute_force_open(start_cost, cost)
        assert hk_len == pytest.approx(bf_len, rel=1e-9, abs=1e-6)


def test_min_path_free_endpoints_matches_brute_force():
    """The start-free minimum path (both endpoints free) equals the best over all
    permutations with zero start cost."""
    rng = random.Random(99)
    for _ in range(60):
        n = rng.randint(1, 7)
        pts = _rand_points(rng, n)
        cost = [[((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5 for b in pts] for a in pts]
        _, hk = held_karp_min_path(cost)
        _, bf = brute_force_open([0.0] * n, cost)
        assert hk == pytest.approx(bf, rel=1e-9, abs=1e-6)


def test_set_tour_cache_resolves_only_on_set_change():
    est = RouteEstimator()
    pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

    est.set_tour_length(pts)
    assert est.solves == 1
    # same set, permuted order + moved start -> cache hit, no new solve
    est.estimate_clear_travel((500.0, 500.0), list(reversed(pts)))
    est.set_tour_length(pts)
    assert est.solves == 1
    # a different set -> exactly one more solve
    est.set_tour_length(pts + [(50.0, 50.0)])
    assert est.solves == 2


def test_estimate_clear_travel_is_nearest_entry_plus_set_tour():
    est = RouteEstimator()
    pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
    start = (0.0, -30.0)
    nearest = 30.0                              # to (0,0)
    tour = est.set_tour_length(pts)             # 100 + 100 (an L path)
    assert est.estimate_clear_travel(start, pts) == pytest.approx(nearest + tour)
    assert tour == pytest.approx(200.0)


def test_two_opt_never_worse_than_nn_seed():
    rng = random.Random(2024)
    for _ in range(30):
        n = rng.randint(4, 10)
        start = (0.0, 0.0)
        pts = _rand_points(rng, n)
        start_cost = [((start[0]-p[0])**2 + (start[1]-p[1])**2) ** 0.5 for p in pts]
        cost = [[((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5 for b in pts] for a in pts]
        seed, seed_len = nearest_neighbor_open(start_cost, cost)
        _, opt_len = two_opt_open(seed, start_cost, cost)
        assert opt_len <= seed_len + 1e-9


def test_tspn_approach_metric_uses_radius():
    """A big neighbourhood radius shortens the approach: d_N = max(0, d − r)."""
    est = RouteEstimator()
    start = (0.0, 0.0)
    far_point = Neighborhood(center=(1000.0, 0.0), radius=0.0)
    far_disc = Neighborhood(center=(1000.0, 0.0), radius=300.0)
    assert est.tspn_route(start, [far_point]).length == pytest.approx(1000.0)
    assert est.tspn_route(start, [far_disc]).length == pytest.approx(700.0)


def test_tspn_adds_localization_cost_without_changing_order():
    est = RouteEstimator()
    start = (0.0, 0.0)
    neigh = [
        Neighborhood((100.0, 0.0), radius=10.0, localization_cost=40.0),
        Neighborhood((200.0, 0.0), radius=10.0, localization_cost=25.0),
    ]
    plan = est.tspn_route(start, neigh)
    assert plan.localization_cost == pytest.approx(65.0)
    assert plan.total == pytest.approx(plan.length + 65.0)
    assert plan.order[0] == (100.0, 0.0)   # nearer neighbourhood first


def test_localization_cost_model_monotone():
    m = LocalizationCostModel(a0=10.0, a1=0.5, a2=0.2, a3=30.0)
    base = m.estimate(r_mec=100.0, diameter=200.0, sin_gamma=1.0)
    assert m.estimate(r_mec=200.0, diameter=200.0, sin_gamma=1.0) > base   # larger MEC costs more
    assert m.estimate(r_mec=100.0, diameter=400.0, sin_gamma=1.0) > base   # larger region costs more
    assert m.estimate(r_mec=100.0, diameter=200.0, sin_gamma=0.1) > base   # worse crossing costs more


def test_guaranteed_clear_route_uses_K_sets_and_declines_when_empty():
    est = RouteEstimator()
    feasible = [Neighborhood((100.0, 0.0), radius=10.0),
                Neighborhood((300.0, 0.0), radius=30.0)]
    route = est.guaranteed_clear_route((0.0, 0.0), feasible, clear_radius=50.0)
    assert route is not None
    assert route.order == [(100.0, 0.0), (300.0, 0.0)]
    assert est.guaranteed_clear_route((0.0, 0.0), feasible, clear_radius=20.0) is None


def test_polygon_vertex_K_certificate_is_conservative():
    square = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
    cert = guaranteed_clear_point(square, clear_radius=15.0)
    assert cert is not None
    assert cert[0] == pytest.approx((0.0, 0.0))
    assert cert[1] == pytest.approx(14.1421356)
    assert guaranteed_clear_point(square, clear_radius=10.0) is None


def test_empty_and_singleton_routes():
    est = RouteEstimator()
    assert est.optimal_clear_order((0.0, 0.0), []).length == 0.0
    single = est.optimal_clear_order((3.0, 4.0), [(0.0, 0.0)])
    assert single.length == pytest.approx(5.0)
    assert single.order == [(0.0, 0.0)]
