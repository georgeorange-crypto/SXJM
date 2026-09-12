import pytest
from way4.planner.lower_bounds import (
    certified_gap, mission_lower_bound, open_mst_lower_bound, service_lower_bound,
)


def test_open_mst_bound_is_not_above_a_simple_open_route():
    lb = open_mst_lower_bound((0.0, 0.0), [(3.0, 0.0), (3.0, 4.0)])
    assert lb == pytest.approx(7.0)


def test_mission_bound_uses_max_travel_to_avoid_double_counting():
    lb = mission_lower_bound(
        start=(0.0, 0.0), coverage_points=[(10.0, 0.0)],
        source_points=[(10.0, 0.0)], clear_centers=[(10.0, 0.0)],
        speed=5.0, n_measure_min=1, n_sources=1, n_switch_min=0,
    )
    assert lb == pytest.approx(12.0)  # 5 + 5 + max(10/5, 10/5, 10/5)


def test_service_and_certified_gap_are_explicit():
    assert service_lower_bound(2, 3, 1) == 26.0
    assert certified_gap(30.0, 20.0) == pytest.approx(0.5)
