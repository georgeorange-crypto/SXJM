import pytest
from way4.rl import gamma_lambda_sweep


def test_gamma_lambda_sweep_includes_full_discount_and_high_gae():
    sweep = gamma_lambda_sweep()
    assert len(sweep) == 9
    assert {row["gamma"] for row in sweep} == {1.0, .995, .99}
    assert {row["gae_lambda"] for row in sweep} == {.95, .97, .99}


def test_gamma_lambda_sweep_rejects_invalid_values():
    with pytest.raises(ValueError):
        gamma_lambda_sweep(gammas=(0.0,))
