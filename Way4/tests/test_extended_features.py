from types import SimpleNamespace

from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.rl.features import (BACKBONE_DECISIONS, CHANNEL_FEATURE_DIM,
                               backbone_feature_block, channel_context_block)


def test_per_channel_feature_block_contains_effective_geometry_and_debt():
    belief = BeliefState(n_channels=2)
    cert = CertificateManager(n_channels=2)
    candidate = MacroCandidate(MacroActionType.EXPLORE, (0.0, 0.0), scan_channels=(1,))
    evaluation = SimpleNamespace(candidate=candidate)
    block = channel_context_block(evaluation, belief, RobotState(), cert)
    assert len(block) == CHANNEL_FEATURE_DIM


def test_backbone_features_encode_follow_and_deviation_semantics():
    class Candidate:
        meta = {"backbone_kind": "BackboneSkip", "backbone_index": 3,
                "skip_gain": 2.0}
    block = backbone_feature_block(Candidate())
    assert len(block) == 1 + len(BACKBONE_DECISIONS) + 7
    assert block[0] == 1.0
    assert block[1 + BACKBONE_DECISIONS.index("SKIP")] == 1.0
    assert all(v >= 0.0 for v in block)
