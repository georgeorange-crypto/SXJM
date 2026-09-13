import inspect
from scripts import train_candidate_ppo as training


def test_training_entrypoint_exposes_resume_and_state_output_flags():
    source = inspect.getsource(training.main)
    assert "--resume-state" in source
    assert "--state-out" in source
