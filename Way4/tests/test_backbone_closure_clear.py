from way4.belief import BeliefState, ChannelStatus
from way4.core import RobotState
from way4.planner import BackboneManager, BackboneNode, BackboneStatus, CandidateGenerator


def test_backbone_closure_selects_largest_certificate_debt():
    manager = BackboneManager([
        BackboneNode("small", (1., 0.), certificate_holes=("h",)),
        BackboneNode("large", (2., 0.), certificate_holes=("a", "b")),
    ])
    out = CandidateGenerator().backbone_closure_candidates(manager, RobotState())
    assert out[0].meta["backbone_kind"] == "BackboneClosure"
    assert out[0].meta["backbone_node"] == "large"


def test_clear_can_be_tagged_as_backbone_piggyback():
    belief = BeliefState(n_channels=1)
    belief[1].status = ChannelStatus.LOCALIZED
    belief[1].F_c = [(10., 0.)]
    belief[1].mec_center = (10., 0.)
    belief[1].mec_radius = 1.0
    out = CandidateGenerator().piggyback_clear_candidates(belief, [(10., 0.)], RobotState())
    assert out and out[0].meta["backbone_kind"] == "PiggybackClear"
