import pytest

torch = pytest.importorskip("torch")
from way4.rl.candidate_ppo import HierarchicalPolicy


def test_hierarchical_policy_emits_spatial_and_channel_heads():
    policy = HierarchicalPolicy(hidden=8, n_channels=4)
    spatial, channel = policy(torch.zeros(2, 3, 8),
                              candidate_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool))
    assert spatial.shape == (2, 3)
    assert channel.shape == (2, 3, 5)
    assert HierarchicalPolicy.joint_log_probability(torch.tensor(1.), torch.tensor(2.)) == 3.
