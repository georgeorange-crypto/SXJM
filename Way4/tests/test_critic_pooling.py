import pytest

torch = pytest.importorskip("torch")
from way4.rl.candidate_ppo import CandidateActorCritic


def test_critic_supports_mean_max_pooling():
    model = CandidateActorCritic(candidate_dim=20, hidden=16, heads=2,
                                 layers=1, critic_pooling="mean_max")
    x = torch.zeros(2, 3, 20)
    logits, value = model(x, torch.ones(2, 3, dtype=torch.bool))
    assert logits.shape == (2, 3)
    assert value.shape == (2,)
