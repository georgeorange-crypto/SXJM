import pytest
from scripts import train_candidate_ppo as training


def test_training_script_rejects_test_overlap(monkeypatch):
    monkeypatch.setattr(training, "CandidateActorCritic", lambda _: None)
    monkeypatch.setattr(training, "CandidatePPOTrainer", lambda *args: None)
    monkeypatch.setattr(training.sys, "argv", ["train", "--profile", "smoke",
        "--seeds", "1", "--validation-seeds", "2", "--test-seeds", "2"])
    with pytest.raises(ValueError, match="non-overlapping"):
        training.main()
