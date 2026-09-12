from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager
from way4.channels import ChannelScheduler


def test_scheduler_uses_unified_action_mask_for_new_states():
    belief = BeliefState(n_channels=3)
    cert = CertificateManager(n_channels=3)
    sch = ChannelScheduler()
    belief[1].status = ChannelStatus.LOCALIZED
    belief[2].status = ChannelStatus.PRESENT_UNOBSERVED
    belief[3].status = ChannelStatus.INITIALIZED
    assert sch.channel_value(1, (0.0, 0.0), belief, cert)[1] is False
    assert sch.channel_value(2, (0.0, 0.0), belief, cert)[1] is True
    assert sch.channel_value(3, (0.0, 0.0), belief, cert)[1] is True
