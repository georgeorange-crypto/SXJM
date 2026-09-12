from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.planner import RecedingHorizonPlanner
from way4.rl.features import CANDIDATE_FEATURE_DIM, FEATURE_DIM, evaluation_features


def test_multi_service_and_synergy_features_are_exposed():
    b = BeliefState(n_channels=3)
    c = MacroCandidate(MacroActionType.STOP, (0.0, 0.0), scan_channels=(1, 2),
                       route_synergy=12.0, service_density=0.25)
    ev = RecedingHorizonPlanner().plan(b, CertificateManager(n_channels=3),
                                       RobotState(), [c]).evaluations[0]
    row = evaluation_features(ev, b, RobotState(), n_candidates=1)
    assert len(row) == FEATURE_DIM
    assert row[CANDIDATE_FEATURE_DIM - 1] == 0.25
