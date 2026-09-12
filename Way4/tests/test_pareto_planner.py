from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.planner import RecedingHorizonPlanner


def test_spatial_pareto_keeps_protected_actions_and_removes_dominated_scan():
    c1 = MacroCandidate(MacroActionType.REFINE, (1, 0), route_marginal=20, refinement_gain=1)
    c2 = MacroCandidate(MacroActionType.REFINE, (2, 0), route_marginal=10, refinement_gain=2)
    clear = MacroCandidate(MacroActionType.CLEAR, (3, 0), clear_channel=1)
    out = RecedingHorizonPlanner._pareto_candidates([c1, c2, clear])
    assert c2 in out and c1 not in out and clear in out


def test_legacy_planner_does_not_prune_candidates():
    b = BeliefState(n_channels=1)
    cert = CertificateManager(n_channels=1)
    c1 = MacroCandidate(MacroActionType.REFINE, (1, 0), route_marginal=20, refinement_gain=1, expected_time=1)
    c2 = MacroCandidate(MacroActionType.REFINE, (2, 0), route_marginal=10, refinement_gain=2, expected_time=2)
    p = RecedingHorizonPlanner()
    assert len(p.plan(b, cert, RobotState(), [c1, c2]).evaluations) == 2
