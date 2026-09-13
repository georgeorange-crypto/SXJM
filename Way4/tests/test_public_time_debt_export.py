from way4.rl import candidate_time_debt_block


def test_time_debt_feature_is_publicly_exported():
    c = type("C", (), {"meta": {"time_debt_before_s": 10., "time_debt_after_s": 5.}})()
    assert candidate_time_debt_block(c) == [0.01, 0.005, -0.005]
