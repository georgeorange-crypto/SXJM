from way4.planner.future_cost import CostView, DetectedRegion, FutureCostEstimator


def test_future_cost_has_four_named_components_and_uncertainty_prediction():
    view = CostView(
        pos=(0.0, 0.0),
        detected=[DetectedRegion((10.0, 0.0), 20.0, 40.0, 0.2, channel=7)],
    )
    out = FutureCostEstimator().estimate(view)
    assert out.total >= 0.0
    assert out.measurement_predictions[7] > 1
    assert out.j_route >= 0.0 and out.j_localization >= 0.0
    assert out.j_exploration >= 0.0 and out.j_certificate >= 0.0


def test_target_set_cache_does_not_depend_on_robot_motion():
    est = FutureCostEstimator()
    a = CostView((0.0, 0.0), clearable_targets=[(100.0, 0.0)])
    b = CostView((5.0, 0.0), clearable_targets=[(100.0, 0.0)])
    est.estimate(a)
    solves_after_a = est.route.solves
    est.estimate(b)
    assert est.route.solves == solves_after_a
