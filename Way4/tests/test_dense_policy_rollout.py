import pytest
from way4.rl.candidate_rollout import timed_policy_transitions


def records(*times):
    return [dict(observation=[[1.]], action=0, log_prob=0., value=0.,
                 virtual_time_before=t) for t in times]


def test_each_decision_pays_time_and_last_pays_rescue_tail():
    ts = timed_policy_transitions(records(0., 100., 150.), 350.,
                                  full_clear=True, n_unresolved=0)
    assert [t.reward for t in ts] == pytest.approx([-.1, -.05, -.2])
    assert sum(t.reward for t in ts) == pytest.approx(-.35)
    assert [t.done for t in ts] == [False, False, True]


def test_failure_adjustment_is_terminal_and_time_is_still_charged():
    ts = timed_policy_transitions(records(0., 100.), 150.,
                                  full_clear=False, n_unresolved=2)
    assert [t.reward for t in ts] == pytest.approx([-.1, -102.05])


def test_clock_regression_rejected():
    with pytest.raises(ValueError):
        timed_policy_transitions(records(100., 50.), 200.,
                                 full_clear=True, n_unresolved=0)
