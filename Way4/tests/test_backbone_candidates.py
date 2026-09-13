from way4.core import RobotState
from way4.planner import BackboneManager, BackboneNode, BackboneStatus, CandidateGenerator


def test_backbone_next_and_bridge_are_exposed():
    manager = BackboneManager([BackboneNode("b0", (100., 0.)), BackboneNode("b1", (20., 0.))])
    manager.route = ["b0", "b1"]
    out = CandidateGenerator().backbone_candidates(manager, RobotState())
    assert out[0].meta["backbone_kind"] == "BackboneNext"
    assert any(c.meta["backbone_kind"] == "BackboneBridge" for c in out)


def test_backbone_skip_targets_later_node_after_replacement():
    manager = BackboneManager([
        BackboneNode("b0", (100., 0.), status=BackboneStatus.REPLACED),
        BackboneNode("b1", (20., 0.)),
        BackboneNode("b2", (30., 0.)),
    ])
    manager.route = ["b0", "b1", "b2"]
    out = CandidateGenerator().backbone_candidates(manager, RobotState())
    skip = [c for c in out if c.meta["backbone_kind"] == "BackboneSkip"]
    assert skip and skip[0].target == (30., 0.)
