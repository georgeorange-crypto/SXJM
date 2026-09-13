from scripts.audit_performance_gates import audit


def test_incomplete_rows_are_not_benchmark_passes():
    result = audit({"complete": True, "results": {"probe": [{"virtual_time_s": 10.0,
        "full_clear": False, "n_channels": 2}]}})
    cell = result["cells"]["probe"]
    assert cell["evidence_status"] == "insufficient_evidence"
    assert cell["gates"] == {}


def test_ready_rows_require_independent_correctness_fields():
    row = {"virtual_time_s": 100.0, "full_clear": True, "n_channels": 2,
           "clear_correct": True, "certificate_sound": True}
    result = audit({"complete": True, "results": {"ok": [row]}},
                   {"all_passed": True})
    cell = result["cells"]["ok"]
    assert cell["evidence_status"] == "ready"
    assert set(cell["gates"]) == {"G1", "G2", "G3", "FINAL"}
    assert cell["median_time_s_per_target"] == 50.0
    assert cell["p90_time_s_per_target"] == 50.0
    assert cell["max_time_s_per_target"] == 50.0
