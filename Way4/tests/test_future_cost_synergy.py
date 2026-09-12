from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import RobotState
from way4.planner.future_cost import FutureCostEstimator, build_cost_view


def test_future_cost_exposes_additive_joint_and_synergy_fields():
    belief = BeliefState(n_channels=2)
    cert = CertificateManager(n_channels=2)
    view = build_cost_view(belief, cert, RobotState())
    result = FutureCostEstimator(joint_route=True).estimate(view)
    assert result.j_joint <= result.j_additive + 1e-9
    assert result.route_synergy >= 0.0
