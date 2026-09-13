from way4.evaluation_protocol import RECOMMENDED_ABLATIONS, compare_ablation_steps


def test_recommended_matrix_has_m0_to_p4_and_explicit_missing_pairs():
    assert [s.name for s in RECOMMENDED_ABLATIONS] == ["M0", "M1", "M2", "M3", "P0", "P1", "P2", "P3", "P4"]
    assert all(row["status"] == "insufficient_data" for row in compare_ablation_steps({}))


def test_ablation_step_comparison_reports_real_delta():
    rows = {name: {"status": "complete", "full_clear_rate": 1.0, "time_mean": i * 10.0}
            for i, name in enumerate(("M0", "M1", "M2", "M3", "P4"))}
    assert compare_ablation_steps(rows)[0]["delta_time_mean"] == 10.0
