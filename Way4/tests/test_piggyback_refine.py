from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.planner import CandidateGenerator


def test_refine_piggyback_api_keeps_dedicated_and_adds_backbone_variants():
    belief = BeliefState(n_channels=1)
    belief[1].status = ChannelStatus.DETECTED
    cert = CertificateManager(n_channels=1)
    gen = CandidateGenerator()
    # Keep the test focused on the T01 contract even when the NBV fixture has
    # no geometric evidence from which to synthesize a dedicated action.
    dedicated = MacroCandidate(MacroActionType.REFINE, (100., 0.), scan_channels=(1,))
    dedicated.meta["refine_channel"] = 1
    gen._refine_candidates = lambda *args, **kwargs: [dedicated]
    out = gen.refine_with_piggyback(belief, cert, RobotState(), [(0., 0.)])
    assert any(not c.meta.get("piggyback") for c in out)
    assert any(c.meta.get("piggyback") for c in out)
