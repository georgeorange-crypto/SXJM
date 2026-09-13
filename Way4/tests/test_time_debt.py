import pytest

from way4.planner.lower_bounds import time_debt
from way4.rl.features import (candidate_time_block, progress_per_second_block,
                               future_route_block, route_efficiency_block,
                               history_behavior_block, time_debt_block,
                               candidate_time_debt_block)


def test_time_debt_is_excess_estimated_future_time():
    assert time_debt(120.0, 80.0) == 40.0
    assert time_debt(50.0, 80.0) == 0.0
    with pytest.raises(ValueError):
        time_debt(-1.0, 0.0)


def test_time_debt_v3_block_exposes_before_after_and_delta():
    assert time_debt_block(before_est_s=120, after_est_s=150,
                           remaining_lb_s=80) == [0.04, 0.07, 0.03]


def test_candidate_time_block_preserves_real_subcomponents_and_total():
    c = type("C", (), {"expected_time": 1000.0, "meta": {
        "predicted_move_time_s": 100, "predicted_measure_time_s": 20,
        "predicted_switch_time_s": 5, "predicted_service_time_s": 25}})()
    assert candidate_time_block(c) == [0.1, 0.02, 0.005, 0.025, 1.0]


def test_candidate_time_debt_block_keeps_signed_change():
    c = type("C", (), {"meta": {"time_debt_before_s": 100.,
                                  "time_debt_after_s": 60.}})()
    assert candidate_time_debt_block(c) == pytest.approx([.1, .06, -.04])


def test_route_efficiency_block_keeps_route_fields_distinct():
    c = type("C", (), {"meta": {"route_delta": 100, "route_rank": 2,
        "detour_ratio": .2, "backtrack_m": 180, "repeated_edge_m": 90,
        "repeated_edge_ratio": .1, "unnecessary_return_m": 45,
        "avoidable_crossing_gain": 30, "corridor_distance": 60,
        "next_task_distance": 120}})()
    out = route_efficiency_block(c)
    assert len(out) == 10 and out[0] == .1 and out[5] == .1


def test_progress_per_second_block_normalizes_all_progress_channels():
    c = type("C", (), {"expected_time": 10., "meta": {
        "coverage_gain": 5, "certificate_gain": 2, "information_gain": 4,
        "clear_probability": 1, "localization_gain": 3}})()
    assert progress_per_second_block(c) == [.5, .2, .4, .1, .3]


def test_future_route_block_exposes_six_required_fields():
    c = type("C", (), {"meta": {"remaining_route_lb_before": 100,
        "remaining_route_lb_after": 80, "future_cost_delta": -20,
        "delta_time_debt": -10, "nearest_next_task": 180,
        "task_cluster_size": 3}})()
    assert future_route_block(c) == [.1, .08, -.02, -.01, .1, 3.0]


def test_history_behavior_block_normalizes_time_distance_and_checks_ratio():
    c = type("C", (), {"meta": {"times_region_visited": 2,
        "times_channel_measured_nearby": 3, "time_since_last_progress": 500,
        "distance_since_last_progress": 900, "recent_route_overlap_ratio": .4}})()
    assert history_behavior_block(c) == [2., 3., .5, .5, .4]
    c.meta["recent_route_overlap_ratio"] = 1.2
    with pytest.raises(ValueError):
        history_behavior_block(c)
