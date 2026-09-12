from way4.routing import Neighborhood, RouteEstimator


def test_gtsp_is_optional_discretized_tspn_solver():
    est = RouteEstimator(strategy="gtsp", exact_max_n=4)
    plan = est.optimal_neighborhood_order(
        (0.0, 0.0), [Neighborhood((10.0, 0.0), 3.0), Neighborhood((20.0, 0.0), 3.0)]
    )
    assert plan.exact is True
    assert len(plan.order) == 2
    assert plan.length <= 20.0
