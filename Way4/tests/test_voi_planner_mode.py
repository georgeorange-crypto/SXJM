from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.planner import RecedingHorizonPlanner


def test_voi_weight_is_optional_and_preserves_default_mode():
    belief = BeliefState(n_channels=2)
    cert = CertificateManager(n_channels=2)
    candidates = [
        MacroCandidate(MacroActionType.EXPLORE, (0.0, 0.0), scan_channels=(1,), expected_time=5.0),
        MacroCandidate(MacroActionType.EXPLORE, (100.0, 0.0), scan_channels=(2,), expected_time=25.0),
    ]
    result = RecedingHorizonPlanner(voi_weight=0.0).plan(
        belief, cert, RobotState(), candidates
    )
    assert result.best is not None
