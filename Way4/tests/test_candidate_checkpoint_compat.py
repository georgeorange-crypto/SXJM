import torch
from way4.rl.candidate_ppo import MODEL_SCHEMA
from way4.rl.final_planner import load_candidate_policy


def test_legacy_candidate_checkpoint_with_matching_shape_loads(tmp_path):
    from way4.rl.candidate_ppo import CandidateActorCritic
    p = tmp_path / "legacy.pt"
    model = CandidateActorCritic(210, hidden=128)
    torch.save({"state_dict": model.state_dict(), "input_dim": 210, "problem": 4}, p)
    _policy, loaded = load_candidate_policy(str(p))
    assert loaded is not None


def test_mismatched_candidate_checkpoint_is_rejected(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"state_dict": {}, "input_dim": 211, "problem": 4, "schema": MODEL_SCHEMA}, p)
    try:
        load_candidate_policy(str(p))
    except ValueError:
        return
    raise AssertionError("mismatched input dimension must be rejected")
