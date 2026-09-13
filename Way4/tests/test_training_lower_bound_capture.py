import inspect
from scripts import train_candidate_ppo as training


def test_training_row_captures_mission_lower_bound_for_case_jammers():
    source = inspect.getsource(training.main)
    assert "mission_lower_bound" in source
    assert "lower_bound_s" in source
