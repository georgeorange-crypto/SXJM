import pytest
from way4.rl.checkpoint_gate import validation_gate


def row(seed=11000):
    return dict(seed=seed, full_clear=True, greedy=True, time_s=100.,
                illegal_clear=0, safety_violation=0, error=None)


def test_complete_safe_validation_can_compare_time():
    result = validation_gate([row(), row(11001)], [11000,11001])
    assert result['passed'] and result['mean_time_s'] == 100.


@pytest.mark.parametrize('change', [dict(full_clear=False), dict(safety_violation=None),
    dict(illegal_clear=1), dict(greedy=False), dict(time_s=float('nan'))])
def test_fast_or_unmeasured_unsafe_checkpoint_is_ineligible(change):
    r = row(); r.update(change)
    assert not validation_gate([r], [11000])['passed']


def test_duplicate_seed_cannot_substitute_for_missing_seed():
    assert not validation_gate([row(), row()], [11000, 11001])['passed']
