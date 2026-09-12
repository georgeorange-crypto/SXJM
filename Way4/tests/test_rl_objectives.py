from way4.rl.objectives import (
    RewardConfig, critic_ablation_values, critic_target, episode_return,
    terminal_reward, transition_reward,
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
