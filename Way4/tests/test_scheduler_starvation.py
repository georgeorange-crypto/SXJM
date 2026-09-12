from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.channels import ChannelScheduler, SchedulerMode


def test_starved_channel_is_promoted_into_plan():
    belief = BeliefState(n_channels=3)
    cert = CertificateManager(n_channels=3)
    scheduler = ChannelScheduler(starvation_limit=2)
    q = (0.0, 0.0)
    scheduler.record_plan([1], n_channels=3)
    scheduler.record_plan([1], n_channels=3)
    plan = scheduler.select(q, belief, cert, current_channel=1, mode=SchedulerMode.EARLY)
    assert 2 in plan.channels or 3 in plan.channels
    assert scheduler.starvation_debt(2) == 2


def test_continuous_scan_debt_includes_elapsed_time_and_resolved_zero():
    belief = BeliefState(n_channels=2)
    cert = CertificateManager(n_channels=2)
    scheduler = ChannelScheduler()
    belief[1].last_scan_time = 0.0
    d0 = scheduler.scan_debt(belief[1], certificate=cert, now_s=0.0)
    d1 = scheduler.scan_debt(belief[1], certificate=cert, now_s=600.0)
    assert d1 > d0
    belief[1].mark_cleared()
    assert scheduler.scan_debt(belief[1], certificate=cert, now_s=600.0) == 0.0
    snap = scheduler.debt_snapshot(2, belief, cert, now_s=600.0)
    assert "continuous_scan_debt" in snap and "skipped_plan_count" in snap


def test_long_trajectory_has_no_permanent_unknown_starvation():
    belief = BeliefState(n_channels=8)
    cert = CertificateManager(n_channels=8)
    scheduler = ChannelScheduler(starvation_limit=3, early_cap=1)
    selected = set()
    for step in range(40):
        plan = scheduler.select((0.0, 0.0), belief, cert, current_channel=1,
                                mode=SchedulerMode.EARLY)
        assert plan.channels
        selected.update(plan.channels)
        scheduler.record_plan(plan.channels, n_channels=8)
        # leave beliefs unresolved: this is specifically a scheduler starvation test
    assert selected == set(range(1, 9))
