from way4.metrics import compute_efficiency_metrics


def test_positive_route_regret_is_reported_as_time_waste():
    result = compute_efficiency_metrics(total_time_s=10, route_regret_waste_s=2.5)
    assert result.route_regret_waste_s == 2.5
