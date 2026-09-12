from way4.analysis import summarize_decision_trace


def test_attribution_uses_only_realized_outcomes():
    out = summarize_decision_trace([
        {"decision_id": 1, "selected": True, "candidate_type": "EXPLORE",
         "information_gain_est": 2.0, "realized": {"time_until_next_replan": 10,
         "distance": 4, "belief_area_reduction": 3, "certificate_gain": 1}},
        {"decision_id": 1, "selected": False, "candidate_type": "CLEAR", "realized": None},
    ])
    assert out["n_decisions"] == 1
    assert out["soft_information_selected"] == 1
    assert out["by_candidate_type"]["EXPLORE"]["mean_time_s"] == 10
