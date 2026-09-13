import pytest
from way4.rl.architecture_contract import contract, validate_policy_output


def test_candidate_ppo_boundary_is_machine_readable():
    c = contract()
    assert c['objective'] == 'min_total_time_under_full_clear_hard_constraint'
    assert c['policy_output'] == 'candidate_index'
    assert c['hard_constraints'] == ['full_clear', 'illegal_clear', 'safety_violation']


def test_policy_cannot_emit_arbitrary_coordinate_or_invalid_index():
    assert validate_policy_output(1, 2) == 1
    with pytest.raises(ValueError): validate_policy_output((1., 2.), 2)
    with pytest.raises(ValueError): validate_policy_output(2, 2)
