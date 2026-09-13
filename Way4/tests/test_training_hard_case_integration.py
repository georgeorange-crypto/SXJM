import inspect
from scripts import train_candidate_ppo as training


def test_training_loop_uses_hard_case_seed_sampler_and_persists_lb_when_available():
    source = inspect.getsource(training.main)
    assert "choose_seed(seeds, case_stats, rng)" in source
    assert "lower_bound_s" in source
