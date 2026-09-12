import pytest
from way4.metrics import compute_efficiency_metrics


def test_efficiency_metrics_match_definitions():
    m = compute_efficiency_metrics(total_time_s=1000, no_progress_time_s=100,
        move_distance_m=500, backtrack_m=50, repeated_edge_m=25,
        unnecessary_return_m=10, scan_time_s=300, useful_observations=6,
        lower_bound_s=800, route_lower_bound_m=400)
    assert m.wasted_time_ratio == pytest.approx(.1)
    assert m.backtrack_ratio == pytest.approx(.1)
    assert m.repeated_edge_ratio == pytest.approx(.05)
    assert m.unnecessary_return_ratio == pytest.approx(.02)
    assert m.scan_efficiency_s_per_useful_observation == 50
    assert m.lower_bound_ratio == pytest.approx(1.25)
    assert m.route_efficiency == pytest.approx(1.25)


def test_zero_denominators_are_defined_and_bad_inputs_rejected():
    assert compute_efficiency_metrics(total_time_s=0).lower_bound_ratio is None
    with pytest.raises(ValueError):
        compute_efficiency_metrics(total_time_s=1, backtrack_m=-1)
    with pytest.raises(ValueError):
        compute_efficiency_metrics(total_time_s=1, lower_bound_s=0)
