from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.planner import RecedingHorizonPlanner
from way4.rl.features import (
    CANDIDATE_FEATURE_DIM,
    FEATURE_DIM,
    candidate_time_debt_block,
    evaluation_features,
    evaluation_features_v3,
    future_route_block,
)


def test_multi_service_and_synergy_features_are_exposed():
    b = BeliefState(n_channels=3)
    c = MacroCandidate(MacroActionType.STOP, (0.0, 0.0), scan_channels=(1, 2),
                       route_synergy=12.0, service_density=0.25)
    ev = RecedingHorizonPlanner().plan(b, CertificateManager(n_channels=3),
                                       RobotState(), [c]).evaluations[0]
    row = evaluation_features(ev, b, RobotState(), n_candidates=1)
    assert len(row) == FEATURE_DIM
    assert row[CANDIDATE_FEATURE_DIM - 1] == 0.25


def test_v3_features_extend_frozen_schema_with_time_and_route_blocks():
    b = BeliefState(n_channels=3)
    c = MacroCandidate(MacroActionType.STOP, (0.0, 0.0), scan_channels=(1,))
    c.meta.update({"time_debt_s": 120.0, "future_route_cost_s": 45.0})
    ev = RecedingHorizonPlanner().plan(b, CertificateManager(n_channels=3),
                                       RobotState(), [c]).evaluations[0]
    base = evaluation_features(ev, b, RobotState(), n_candidates=1)
    v3 = evaluation_features_v3(ev, b, RobotState(), n_candidates=1)
    assert v3[:FEATURE_DIM] == base
    assert len(v3) == FEATURE_DIM + len(candidate_time_debt_block(c)) + len(future_route_block(c))
    assert all(value == value and abs(value) < 1e9 for value in v3)
