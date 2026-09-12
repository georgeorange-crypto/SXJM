from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager


def test_cardinality_presence_only_when_all_unknown_are_forced():
    belief = BeliefState(n_channels=20)
    mgr = CertificateManager(n_channels=20)
    # Ten channels are hard-certified absent; the remaining ten must all hold
    # sources because the problem lower bound is ten.
    for c in range(11, 21):
        mgr.certs[c].visited_anchor_idx = set(range(len(mgr.anchors)))
    forced = mgr.apply_cardinality_presence(belief)
    assert forced == list(range(1, 11))
    assert all(belief[c].status == ChannelStatus.PRESENT_UNOBSERVED for c in forced)


def test_partial_lower_bound_does_not_mark_every_unknown_present():
    belief = BeliefState(n_channels=20)
    mgr = CertificateManager(n_channels=20)
    mgr._present.update(range(1, 6))
    for c in range(1, 6):
        mgr.certs[c].present = True
    assert mgr.apply_cardinality_presence(belief) == []
