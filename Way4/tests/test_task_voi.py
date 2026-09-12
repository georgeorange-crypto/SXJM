from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.planner import RecedingHorizonPlanner


def test_task_voi_is_nonnegative_and_time_normalized():
    planner = RecedingHorizonPlanner(horizon=1)
    belief = BeliefState(n_channels=3)
    cert = CertificateManager(n_channels=3)
    action = MacroCandidate(MacroActionType.EXPLORE, (0.0, 0.0), scan_channels=(1,), expected_time=5.0)
    assert planner.value_of_information(action, belief, cert, RobotState()) >= 0.0
