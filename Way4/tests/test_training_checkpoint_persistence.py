import torch

from scripts.train_candidate_ppo import save_candidate_checkpoint
from way4.rl.candidate_ppo import CandidateActorCritic, MODEL_SCHEMA


def test_training_checkpoint_persists_partial_and_complete_metadata(tmp_path):
    model = CandidateActorCritic(210)
    out = tmp_path / "candidate.pt"
    rows = [{'seed': 10000, 'cleared': 13, 'total': 13, 'time_s': 123.0}]
    save_candidate_checkpoint(model, out, rows=rows, transitions=7, complete=False)
    partial = torch.load(out, map_location="cpu")
    assert partial["schema"] == MODEL_SCHEMA
    assert partial["complete"] is False
    assert partial["seeds"] == [10000]
    assert partial["transitions"] == 7
    save_candidate_checkpoint(model, out, rows=rows, transitions=7, complete=True)
    complete = torch.load(out, map_location="cpu")
    assert complete["complete"] is True
    assert not (tmp_path / "candidate.pt.tmp").exists()
