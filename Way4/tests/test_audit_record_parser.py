import pytest
from way4.rl.candidate_rollout import audit_from_record


def test_pipeline_record_parser_validates_and_preserves_audit_fields():
    a = audit_from_record({'virtual_time_before': 10, 'virtual_time_after': 20,
        'delta_move_time_s': 2, 'delta_measure_time_s': 3,
        'delta_switch_time_s': 1, 'delta_clear_time_s': 4,
        'repeat_measure': True})
    assert a.accounted_time_s == 10
    assert a.repeat_measure


def test_pipeline_record_parser_rejects_bad_accounting():
    with pytest.raises(ValueError):
        audit_from_record({'virtual_time_before': 0, 'virtual_time_after': 10,
                           'delta_move_time_s': 9})
