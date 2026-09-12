from way4.evaluation_protocol import (
    DEFAULT_SPLITS, DETERMINISTIC_ABLATIONS, assert_ablation_matrix_valid,
    summarize_ablation_results,
)


def test_evaluation_splits_and_ablation_matrix_are_locked():
    DEFAULT_SPLITS.assert_disjoint()
    assert_ablation_matrix_valid()
    assert len(DETERMINISTIC_ABLATIONS) >= 5
    assert DEFAULT_SPLITS.split_of(4000) == "adversarial"
    assert DEFAULT_SPLITS.split_of(9999) is None


def test_ablation_summary_preserves_missing_and_failure_states():
    rows = {
        "way4_full": [{"success": True, "virtual_time_s": 10.0},
                      {"success": False, "virtual_time_s": 20.0}],
        "minus_no_signal": [{"success": True, "virtual_time_s": 12.0},
                            {"success": None, "error": "deadline"}],
    }
    report = summarize_ablation_results(DETERMINISTIC_ABLATIONS, rows)
    full = next(c for c in report["cells"] if c["name"] == "way4_full")
    missing = next(c for c in report["cells"] if c["name"] == "minus_cardinality")
    partial = next(c for c in report["cells"] if c["name"] == "minus_no_signal")
    assert full["full_clear_rate"] == 0.5
    assert missing["status"] == "missing"
    assert partial["status"] == "partial"


def test_ablation_summary_distinguishes_real_failure_from_missing_cell():
    report = summarize_ablation_results(
        DETERMINISTIC_ABLATIONS,
        {"minus_no_signal": [{"success": False, "error": "no_progress_stall"}]},
    )
    failed = next(c for c in report["cells"] if c["name"] == "minus_no_signal")
    missing = next(c for c in report["cells"] if c["name"] == "minus_cardinality")
    assert failed["status"] == "failed"
    assert failed["errors"] == ["no_progress_stall"]
    assert missing["status"] == "missing"
    assert report["complete"] is False


def test_ablation_summary_accepts_rows_from_multiple_real_batches():
    rows = {
        "way4_full": [{"success": True, "virtual_time_s": 10.0}],
    }
    rows.update({
        "adaptive_rerank": [{"success": True, "virtual_time_s": 12.0}],
    })
    report = summarize_ablation_results(DETERMINISTIC_ABLATIONS, rows)
    full = next(c for c in report["cells"] if c["name"] == "way4_full")
    adaptive = next(c for c in report["cells"] if c["name"] == "adaptive_rerank")
    assert full["n_valid"] == 1
    assert adaptive["n_valid"] == 1
