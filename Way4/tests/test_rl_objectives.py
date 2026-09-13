import pytest
from way4.rl.objectives import (
    RewardConfig, critic_ablation_values, critic_target, episode_return,
    dense_time_reward, terminal_reward, transition_reward, potential_shaping, state_potential,
)


def test_time_is_primary_transition_reward():
    assert transition_reward(2.5) == -2.5
    assert transition_reward(-3.0) == 0.0


def test_full_clear_is_a_strict_terminal_cliff():
    cfg = RewardConfig()
    assert terminal_reward(True, cfg) == 0.0
    assert terminal_reward(False, cfg, 2) == -102.0
    assert episode_return(10.0, True, cfg) > episode_return(0.0, False, cfg, 1)


def test_critic_is_negative_remaining_time_plus_bounded_residual_contract():
    assert critic_target(30.0, 2.0) == -28.0
    vals = critic_ablation_values(30.0, 2.0)
    assert vals == {
        "analytic_future_cost_only": -30.0,
        "learned_critic_only": 2.0,
        "analytic_plus_learned_residual": -28.0,
    }


def test_potential_shaping_is_small_and_state_based():
    before = state_potential(certificate_progress=1, unresolved_count=3)
    after = state_potential(certificate_progress=2, unresolved_count=2)
    assert potential_shaping(before, after, eta=.05) == .1
    assert dense_time_reward(100.0, cfg=RewardConfig(shaping_eta=.05),
                             phi_before=before, phi_after=after) == pytest.approx(-.1 + .1)


def test_dense_time_reward_charges_explicit_detour_seconds():
    assert dense_time_reward(100.0, detour_time_s=50.0) == pytest.approx(-.15)
