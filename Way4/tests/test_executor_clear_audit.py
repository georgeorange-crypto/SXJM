from way4.belief import BeliefState
from way4.executor import MacroExecutor
from way4.core import RobotState


class HitEnv:
    def clear(self, x, y, channel): return True, 5.


def test_successful_unproven_clear_is_still_flagged():
    ex = MacroExecutor(HitEnv(), belief=BeliefState(n_channels=20))
    ex.step_clear(RobotState(), (0., 0.), 1)
    assert ex.clear_audit[0]['hit']
    assert not ex.clear_audit[0]['legal']


def test_near_proof_checked_at_actual_target_before_state_is_cleared():
    belief = BeliefState(n_channels=20)
    belief[1].record_near((0., 0.), 0.)
    ex = MacroExecutor(HitEnv(), belief=belief)
    ex.step_clear(RobotState(), (0., 0.), 1)
    assert ex.clear_audit[0]['proof'] == 'near_disk'


def test_distant_clear_cannot_reuse_near_certificate():
    belief = BeliefState(n_channels=20)
    belief[1].record_near((0., 0.), 0.)
    ex = MacroExecutor(HitEnv(), belief=belief)
    ex.step_clear(RobotState(), (100., 0.), 1)
    assert not ex.clear_audit[0]['legal']


def test_strict_executor_rejects_unproven_clear_without_advancing_clock():
    belief = BeliefState(n_channels=20)
    ex = MacroExecutor(HitEnv(), belief=belief, strict_clear_safety=True)
    state, hit, done = ex.step_clear(RobotState(), (0., 0.), 1)
    assert not hit and not done and state.virtual_time_s == 0.0
    assert ex.clear_audit[0]['rejected']
