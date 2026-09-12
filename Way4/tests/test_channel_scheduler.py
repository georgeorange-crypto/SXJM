"""§7 ChannelScheduler — batch-scan set selection at a waypoint.

Locks the dwell economics (value per second, switch only on channel change) and
the hard rules: resolved channels are never re-scanned, the current channel is
measured first to save the switch, VERIFICATION scans only still-UNKNOWN &
un-certified channels, and ``must_include`` pins a channel regardless of value.
"""

import pytest

from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager
from way4.channels import ChannelScheduler, SchedulerMode


def _fresh(n=6):
    return BeliefState(n_channels=n), CertificateManager(n_channels=n)


def test_n_switch_and_dwell_formula():
    sch = ChannelScheduler()
    assert sch.n_switch([2, 5, 7], current_channel=2) == 2   # current measured first
    assert sch.n_switch([2, 5, 7], current_channel=9) == 3   # must switch into the set
    assert sch.n_switch([], 1) == 0
    # dwell = 5·|G| + 1·N_switch
    assert sch._dwell([2, 5, 7], 2) == (5 * 3 + 2, 2)
    assert sch._dwell([2, 5, 7], 9) == (5 * 3 + 3, 3)


def test_resolved_channels_never_scanned():
    belief, cert = _fresh()
    belief[4].mark_cleared()
    belief[5].status = ChannelStatus.ABSENT_CERTIFIED
    belief[6].record_bearing((0.0, 0.0), 0.0)     # DETECTED
    # localize ch6 -> should not be scanned (clear it instead)
    belief[3].record_near((10.0, 0.0))
    belief[3].status = ChannelStatus.LOCALIZED

    q = (300.0, 400.0)
    for c in (3, 4, 5):
        _, scannable = sch_val(belief, cert, c, q)
        assert scannable is False

    plan = ChannelScheduler().select(q, belief, cert, current_channel=1)
    assert 4 not in plan.channels and 5 not in plan.channels and 3 not in plan.channels


def sch_val(belief, cert, c, q):
    return ChannelScheduler().channel_value(c, q, belief, cert)


def test_current_channel_measured_first():
    belief, cert = _fresh()
    # several fresh UNKNOWN channels -> all have equal coverage value at q
    q = (0.0, 0.0)
    plan = ChannelScheduler().select(q, belief, cert, current_channel=3)
    assert plan.channels, "expected a non-empty batch"
    if 3 in plan.channels:
        assert plan.channels[0] == 3
        assert plan.n_switch == len(plan.channels) - 1


def test_zero_value_channel_excluded_but_pinnable():
    belief, cert = _fresh()
    q = (0.0, 0.0)
    # pre-cover ch2 at q: a repeat scan there adds no new coverage -> value ~0
    from way4.core import Observation
    cert.record_observation(2, q, Observation.no_signal())
    val2, scannable2 = ChannelScheduler().channel_value(2, q, belief, cert)
    assert scannable2 is True and val2 == pytest.approx(0.0, abs=1e-9)

    plan = ChannelScheduler().select(q, belief, cert, current_channel=1)
    assert 2 not in plan.channels                     # dropped for zero value
    pinned = ChannelScheduler().select(q, belief, cert, current_channel=1, must_include=[2])
    assert 2 in pinned.channels                        # pinned regardless


def test_verification_mode_only_unknown_uncertified():
    belief, cert = _fresh()
    belief[2].record_bearing((0.0, 0.0), 0.0)         # DETECTED, not UNKNOWN -> excluded
    # certify ch3 absent via the legacy backbone (all template anchors visited)
    cert.certs[3].visited_anchor_idx = set(range(len(cert.anchors)))
    assert cert.is_absent_certified(3) is True

    q = (100.0, 100.0)
    plan = ChannelScheduler().select(q, belief, cert, current_channel=1, mode=SchedulerMode.VERIFICATION)
    assert 2 not in plan.channels        # DETECTED
    assert 3 not in plan.channels        # already certified
    assert 1 in plan.channels            # still UNKNOWN & un-certified


def test_mid_mode_caps_batch_size():
    belief, cert = _fresh(n=20)
    q = (0.0, 0.0)
    sch = ChannelScheduler(mid_cap=4)
    plan = sch.select(q, belief, cert, current_channel=1, mode=SchedulerMode.MID)
    assert len(plan.channels) <= 4


def test_ratio_prefers_focused_batch_over_padding_with_low_value():
    """With one high-value and many tiny-value channels, the value-per-second ratio
    should not pad the batch with near-worthless channels."""
    belief, cert = _fresh(n=20)
    q = (0.0, 0.0)
    from way4.core import Observation
    # pre-cover channels 2..20 at q so only ch1 carries real value here
    for c in range(2, 21):
        cert.record_observation(c, q, Observation.no_signal())
    plan = ChannelScheduler().select(q, belief, cert, current_channel=1)
    assert plan.channels == [1]
