from way4.belief import BeliefState, ChannelStatus
from way4.planner import RemainingTaskPool


def test_waiting_is_planner_state_not_belief_state():
    belief = BeliefState(n_channels=2)
    belief[1].status = ChannelStatus.DETECTED
    pool = RemainingTaskPool(starvation_limit=2)
    pool.sync(belief)
    assert pool.consider_wait(1, 100.0, 20.0, (3.0, 4.0))
    assert belief[1].status == ChannelStatus.DETECTED
    assert pool.snapshot()["1"]["status"] == "WAITING_FOR_ROUTE"


def test_waiting_promotes_after_starvation_limit():
    belief = BeliefState(n_channels=1)
    belief[1].status = ChannelStatus.DETECTED
    pool = RemainingTaskPool(starvation_limit=2)
    pool.sync(belief)
    assert pool.consider_wait(1, 100, 10, (0, 0))
    assert pool.consider_wait(1, 100, 10, (0, 0))
    assert not pool.consider_wait(1, 100, 10, (0, 0))
