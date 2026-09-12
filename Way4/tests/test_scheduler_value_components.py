from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.channels import ChannelScheduler


def test_unified_value_components_and_eta():
    belief = BeliefState(n_channels=2)
    cert = CertificateManager(n_channels=2)
    sch = ChannelScheduler()
    comp = sch.value_components(1, (0.0, 0.0), belief, cert)
    assert set(comp) == {"V_E", "V_I", "V_R", "V_C", "V_D"}
    assert comp["V_E"] >= 0.0 and comp["V_D"] >= 0.0
    assert sch.eta(1, (0.0, 0.0), belief, cert, 1) >= 0.0
