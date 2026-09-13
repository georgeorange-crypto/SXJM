import pytest

torch = pytest.importorskip("torch")
from way4.rl.candidate_ppo import CandidateActorCritic
from way4.rl.features import critic_time_state_block


def test_critic_accepts_explicit_temporal_global_state():
    model = CandidateActorCritic(candidate_dim=20, hidden=16, heads=2,
                                 layers=1, global_time_dim=7)
    x = torch.zeros(2, 3, 20)
    temporal = torch.tensor([critic_time_state_block(
        elapsed_time_s=10, unresolved_count=3, cleared_count=2,
        certificate_coverage=.5, remaining_route_lb_s=20, time_debt_s=4, phase=1),
        critic_time_state_block(elapsed_time_s=20, unresolved_count=2, cleared_count=3,
                                certificate_coverage=.7, remaining_route_lb_s=10,
                                time_debt_s=1, phase=2)])
    logits, value = model(x, torch.ones(2, 3, dtype=torch.bool), temporal)
    assert logits.shape == (2, 3)
    assert value.shape == (2,)
