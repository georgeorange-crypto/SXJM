from way4.routing.estimator import Neighborhood, RouteEstimator, guaranteed_clear_neighborhood


def test_disk_guaranteed_clear_set_is_exact_residual_disk():
    f = Neighborhood((10.0, 20.0), radius=3.0)
    k = guaranteed_clear_neighborhood(f, clear_radius=10.0)
    assert k is not None
    assert k.center == f.center
    assert k.radius == 7.0


def test_empty_guaranteed_set_is_explicit_when_feasible_region_too_wide():
    assert guaranteed_clear_neighborhood(Neighborhood((0.0, 0.0), 11.0), 10.0) is None


def test_guaranteed_route_uses_clear_sets_and_rejects_unsafe_target():
    est = RouteEstimator()
    route = est.guaranteed_clear_route(
        (0.0, 0.0), [Neighborhood((20.0, 0.0), 2.0)], clear_radius=10.0
    )
    assert route is not None
    assert route.order == [(20.0, 0.0)]
    # The robot starts at the origin: approach to K is 20 - 8 = 12 m.
    assert route.length == 12.0
    assert est.guaranteed_clear_route(
        (0.0, 0.0), [Neighborhood((20.0, 0.0), 11.0)], clear_radius=10.0
    ) is None


def test_clear_detour_value_matches_acceptance_formula():
    assert RouteEstimator.clear_detour_value(100.0, 5.0) == 10.0
