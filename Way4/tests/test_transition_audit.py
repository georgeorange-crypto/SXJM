from way4.rl.candidate_rollout import TransitionAudit
from way4.rl.objectives import dense_time_reward
import pytest


@pytest.mark.parametrize('before,after,move,measure', [
    (0, float('nan'), 0, 0), (0, float('inf'), float('inf'), 0),
    (0, 1, -1, 2), (2, 1, 0, 0), (0, 10, 1, 1),
])
def test_invalid_clock_or_buckets_are_rejected(before, after, move, measure):
    with pytest.raises(ValueError):
        TransitionAudit(before, after, move, measure).validate()


def test_transition_audit_reconciles_time_and_exports_fields():
    a = TransitionAudit(10.0, 117.0, 100.0, 5.0, 1.0, 1.0,
                        delta_move_distance_m=500.0, delta_clear=1)
    a.validate()
    row = a.to_record()
    assert row["delta_virtual_time_s"] == 107.0
    assert row["accounted_time_s"] == 107.0
    assert row["delta_move_distance_m"] == 500.0


def test_dense_time_reward_is_immediate_and_bounded():
    assert dense_time_reward(100.0) == -0.1
    assert dense_time_reward(100.0, no_progress=True) == pytest.approx(-0.3)
    assert dense_time_reward(100.0, no_progress=True, repeat_measure=True) == pytest.approx(-0.3)
