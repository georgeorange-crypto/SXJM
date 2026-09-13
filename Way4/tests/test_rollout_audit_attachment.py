from way4.rl.candidate_rollout import timed_policy_transitions


def test_dense_rollout_attaches_real_decision_audit_when_buckets_exist():
    records = [{"observation": [[0.]], "action": 0, "log_prob": 0., "value": 0.,
                "virtual_time_before": 0., "virtual_time_after": 3., "delta_move_time_s": 1.,
                "delta_measure_time_s": 2., "delta_switch_time_s": 0.,
                "delta_clear_time_s": 0.}]
    transitions = timed_policy_transitions(records, 3., full_clear=True, n_unresolved=0)
    assert transitions[0].audit is not None
    assert transitions[0].audit.accounted_time_s == 3.


def test_rollout_preserves_progress_audit_fields():
    records = [{"observation": [[0.]], "action": 0, "log_prob": 0., "value": 0.,
                "virtual_time_before": 0., "virtual_time_after": 3.,
                "delta_move_time_s": 1., "delta_measure_time_s": 2.,
                "delta_clear": 1, "delta_certificate": .25,
                "delta_localization_area": 4., "delta_mec_radius": 2.,
                "delta_switch_time_s": 0., "delta_clear_time_s": 0.}]
    transition = timed_policy_transitions(records, 3., full_clear=True,
                                          n_unresolved=0)[0]
    assert transition.audit.delta_clear == 1
    assert transition.audit.delta_certificate == .25
    assert transition.audit.delta_localization == 4.
    assert transition.audit.delta_mec == 2.
