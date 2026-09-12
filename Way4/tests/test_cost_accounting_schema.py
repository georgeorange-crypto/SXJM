from types import SimpleNamespace

from way4.evaluation_metrics import episode_row


def test_episode_row_exposes_versioned_accounting_and_equivalent_cost():
    result = SimpleNamespace(
        full_clear=True,
        virtual_time_s=123.0,
        total_distance_m=500.0,
        n_measure=10,
        n_empty_scan=4,
        n_longjump=1,
        n_crossing=0,
        time_move_s=100.0,
        time_measure_s=15.0,
        time_switch_s=2.0,
        time_clear_s=5.0,
        time_other_s=1.0,
        time_accounting_error_s=0.0,
        equivalent_route_cost_m=800.0,
        planner_version="way4-route-math-v1",
        metric_schema_version="way4-metrics-v2",
    )
    row = episode_row(result, n_sources=5)
    assert row["L_equivalent_m"] == 800.0
    assert row["T_other_s"] == 1.0
    assert row["T_accounting_error_s"] == 0.0
    assert row["planner_version"] == "way4-route-math-v1"
    assert row["metric_schema_version"] == "way4-metrics-v2"
