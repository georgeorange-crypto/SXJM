from types import SimpleNamespace
from way4.audit_report import build_hard_trace_audit


def test_hard_trace_audit_contains_route_measurement_and_time_sections():
    result = build_hard_trace_audit(
        route_metrics=SimpleNamespace(total_distance=100, backtrack_m=10,
                                      repeated_edge_m=5, unnecessary_return_m=2,
                                      avoidable_crossing_m=1, certificate_only_travel=3),
        primitive_trace=[{"action": "measure", "useful": True},
                         {"action": "measure", "repeat": True}],
        total_time_s=80, time_buckets={"useful_time_s": 50})
    assert result["route"]["backtrack_m"] == 10
    assert result["measurement"]["total_measurements"] == 2
    assert result["measurement"]["no_progress_measurements"] == 1
    assert result["time"]["useful_time_s"] == 50
