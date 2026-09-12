"""Hard acceptance tests for the V4 relational Candidate-PPO policy."""

import pytest

torch = pytest.importorskip("torch")

from way4.rl.candidate_ppo import CandidateActorCritic


def _batch():
    torch.manual_seed(7)
    return torch.randn(1, 3, 210)


def test_padding_does_not_change_valid_logits_or_critic():
    model = CandidateActorCritic(210, hidden=16, layers=1).eval()
    x = _batch()
    padded = torch.cat([x, torch.randn(1, 2, 210) * 100], dim=1)
    mask = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        expected_logits, expected_value = model(x)
        logits, value = model(padded, mask)
    assert torch.allclose(logits[:, :3], expected_logits, atol=1e-5)
    assert torch.allclose(value, expected_value, atol=1e-5)
    assert (logits[:, 3:] < -1e8).all()


def test_candidate_context_can_change_a_fixed_candidate_logit():
    """Moving B must be able to change A's score while A stays identical."""
    model = CandidateActorCritic(210).eval()
    x1 = _batch(); x2 = x1.clone()
    x1[0, 0, :2] = torch.tensor([500.0, 0.0])
    x2[0, 0, :2] = x1[0, 0, :2]
    x1[0, 1, :2] = torch.tensor([510.0, 0.0])
    x2[0, 1, :2] = torch.tensor([-1500.0, 0.0])
    with torch.no_grad():
        l1, _ = model(x1); l2, _ = model(x2)
    assert not torch.allclose(l1[0, 0], l2[0, 0])


def test_candidate_policy_is_permutation_equivariant():
    model = CandidateActorCritic(210).eval()
    x = _batch(); perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        logits, value = model(x)
        logits_p, value_p = model(x[:, perm])
    assert torch.allclose(logits_p[:, torch.argsort(perm)], logits, atol=1e-5)
    assert torch.allclose(value_p, value, atol=1e-5)


def test_route_context_is_visible_to_candidate_representation():
    model = CandidateActorCritic(210).eval()
    x1 = _batch(); x2 = x1.clone()
    # Candidate-local geometry is unchanged; route context fields change.
    x1[0, 1, 5] = 0.0; x2[0, 1, 5] = 20.0
    with torch.no_grad():
        l1, _ = model(x1); l2, _ = model(x2)
    assert not torch.allclose(l1[0, 0], l2[0, 0])
