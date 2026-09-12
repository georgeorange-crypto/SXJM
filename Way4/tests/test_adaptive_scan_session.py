from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.channels import AdaptiveScanSession, ChannelScheduler


def test_adaptive_session_selects_one_and_reranks_after_observe():
    belief = BeliefState(n_channels=3)
    cert = CertificateManager(n_channels=3)
    session = AdaptiveScanSession(ChannelScheduler(), (0.0, 0.0), belief, cert, 1)
    first = session.choose()
    assert len(first.channels) == 1
    session.observe(first.channels[0])
    second = session.choose()
    assert len(second.channels) <= 1
    assert first.channels[0] not in second.channels


def test_adaptive_session_returns_empty_plan_as_stop():
    belief = BeliefState(n_channels=2)
    cert = CertificateManager(n_channels=2)
    session = AdaptiveScanSession(ChannelScheduler(), (0.0, 0.0), belief, cert, 1,
                                  min_value=2.0)
    assert session.choose().is_empty
    assert session.choose().stop_reason == "no_positive_voi"
