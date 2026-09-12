from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.planner import RecedingHorizonPlanner


def test_synergy_weight_changes_ranking_only_when_enabled():
    b, cert, s = BeliefState(n_channels=1), CertificateManager(n_channels=1), RobotState()
    slow = MacroCandidate(MacroActionType.REFINE, (0, 0), expected_time=1,
                          route_synergy=0)
    bundled = MacroCandidate(MacroActionType.REFINE, (100, 0), expected_time=100,
                             route_synergy=200)
    legacy = RecedingHorizonPlanner().plan(b, cert, s, [slow, bundled])
    spatial = RecedingHorizonPlanner(route_synergy_weight=1.0).plan(b, cert, s, [slow, bundled])
    assert legacy.best is slow
    assert spatial.best is bundled
