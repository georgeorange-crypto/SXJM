from way4.rl.candidate_rollout import timed_policy_transitions
import pytest


def test_repeat_measure_flag_changes_only_the_marked_decision_cost():
    rows = [dict(observation=[[1.]], action=0, log_prob=0., value=0.,
                 virtual_time_before=0., no_progress=False),
            dict(observation=[[1.]], action=0, log_prob=0., value=0.,
                 virtual_time_before=100., no_progress=True, repeat_measure=True)]
    ts = timed_policy_transitions(rows, 200., full_clear=True, n_unresolved=0)
    assert ts[0].reward == -0.1
    assert ts[1].reward == pytest.approx(-0.3)
