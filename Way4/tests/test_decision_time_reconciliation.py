from way4.rl.candidate_rollout import TransitionAudit


def test_transition_audit_rejects_unreconciled_buckets():
    a = TransitionAudit(0., 10., delta_move_time_s=9.)
    try:
        a.validate()
    except ValueError as exc:
        assert 'reconcile' in str(exc)
    else:
        raise AssertionError('unreconciled decision time was accepted')
