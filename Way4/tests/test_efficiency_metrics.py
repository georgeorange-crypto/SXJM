import pytest
from way4.metrics import (backbone_pruning_rate, compute_efficiency_metrics,
                          route_waste_penalty_seconds)


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


def test_coverage_and_route_overlap_are_separate_ratios():
    m = compute_efficiency_metrics(
        total_time_s=100, move_distance_m=200, repeated_edge_m=40,
        coverage_overlap_gain=3, coverage_total_gain=10,
    )
    assert m.coverage_overlap_ratio == pytest.approx(0.3)
    assert m.route_overlap_ratio == pytest.approx(0.2)
    with pytest.raises(ValueError):
        compute_efficiency_metrics(total_time_s=1, coverage_overlap_gain=2,
                                    coverage_total_gain=1)


def test_route_waste_penalty_is_masked_by_task_type_and_in_seconds():
    explore = route_waste_penalty_seconds("explore", backtrack_m=50, speed_mps=5)
    clear = route_waste_penalty_seconds("clear", backtrack_m=50, speed_mps=5)
    refine_fast = route_waste_penalty_seconds("refine", backtrack_m=50,
                                               speed_mps=5, info_per_second=4)
    assert explore == 10.0
    assert clear == pytest.approx(1.0)
    assert refine_fast == pytest.approx(0.5)


def test_zero_denominators_are_defined_and_bad_inputs_rejected():
    assert compute_efficiency_metrics(total_time_s=0).lower_bound_ratio is None
    with pytest.raises(ValueError):
        compute_efficiency_metrics(total_time_s=1, backtrack_m=-1)
    with pytest.raises(ValueError):
        compute_efficiency_metrics(total_time_s=1, lower_bound_s=0)


def test_backbone_pruning_rate_counts_retired_nodes():
    assert backbone_pruning_rate(10, 4) == pytest.approx(.6)
    assert compute_efficiency_metrics(total_time_s=1, backbone_initial_count=10,
                                      backbone_visited_count=4).backbone_pruning_rate == pytest.approx(.6)
