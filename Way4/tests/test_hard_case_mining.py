import pytest
from way4.rl import hard_case_weight, prioritized_case_weights


def test_time_over_lower_bound_and_tags_raise_priority():
    easy = {"time_s": 100, "lower_bound_s": 100}
    hard = {"time_s": 300, "lower_bound_s": 100,
            "tags": ["boundary", "long_route"]}
    assert hard_case_weight(hard) > hard_case_weight(easy)
    probs = prioritized_case_weights([easy, hard])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[1] > probs[0]
