import pytest
from way4.rl.candidate_ppo import PPOConfig, entropy_coefficient


def test_entropy_anneals_from_point_one_to_point_zero_one():
    cfg = PPOConfig(entropy_coef=.01, entropy_end=.001)
    assert entropy_coefficient(cfg, 0, 3) == pytest.approx(.01)
    assert entropy_coefficient(cfg, 1, 3) == pytest.approx(.0055)
    assert entropy_coefficient(cfg, 2, 3) == pytest.approx(.001)


def test_entropy_schedule_rejects_invalid_counts():
    with pytest.raises(ValueError): entropy_coefficient(PPOConfig(), -1, 2)
    with pytest.raises(ValueError): entropy_coefficient(PPOConfig(), 0, 0)
