from types import SimpleNamespace

from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.rl.features import CHANNEL_FEATURE_DIM, channel_context_block


def test_per_channel_feature_block_contains_effective_geometry_and_debt():
    belief = BeliefState(n_channels=2)
    cert = CertificateManager(n_channels=2)
    candidate = MacroCandidate(MacroActionType.EXPLORE, (0.0, 0.0), scan_channels=(1,))
    evaluation = SimpleNamespace(candidate=candidate)
    block = channel_context_block(evaluation, belief, RobotState(), cert)
    assert len(block) == CHANNEL_FEATURE_DIM
    assert all(v >= 0.0 for v in block)
