from way4.analysis import algorithm_gate


def test_algorithm_gate_blocks_full_clear_regression():
    result = algorithm_gate(
        [{"seed": 1, "success_ground_truth": True, "virtual_time_s": 10, "resolved": 5}],
        [{"seed": 1, "success_ground_truth": False, "virtual_time_s": 5, "resolved": 4}],
    )
    assert result["full_clear_regressions"] == [1]
    assert result["promote_default"] is False
