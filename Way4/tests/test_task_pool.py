from way4.belief import BeliefState, ChannelStatus
from way4.planner import RemainingTaskPool
from way4.planner.opportunities import RemainingTask


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


def test_route_assignment_binds_cheapest_viable_future_waypoint():
    belief = BeliefState(n_channels=1)
    belief[1].status = ChannelStatus.DETECTED
    pool = RemainingTaskPool()
    pool.sync(belief)
    assigned = pool.assign_route_opportunities(
        [(0.0, 0.0), (100.0, 0.0), (1000.0, 0.0)], {1: 100.0}
    )
    assert assigned[1] == (100.0, 0.0)
    assert pool.waiting[1].assigned_waypoint == (100.0, 0.0)


def test_route_assignment_respects_feasibility_gate():
    belief = BeliefState(n_channels=1)
    belief[1].status = ChannelStatus.DETECTED
    pool = RemainingTaskPool()
    pool.sync(belief)
    assert pool.assign_route_opportunities(
        [(0, 0), (10, 0)], {1: 100}, viable=lambda _c, _p: False
    ) == {}


def test_pending_pool_exposes_all_task_families():
    belief = BeliefState(n_channels=2)
    belief[1].status = ChannelStatus.UNKNOWN
    belief[2].status = ChannelStatus.DETECTED
    pool = RemainingTaskPool()
    pool.sync(belief)
    assert set(pool.pending_tasks) == {
        "EXPLORE", "INITIALIZE", "REFINE", "REACQUIRE", "PURSUE", "CLEAR",
        "VERIFY", "BACKBONE",
    }
    assert [t.channel for t in pool.pending_tasks["REFINE"]] == [2]
    pool.register(RemainingTask("CLEAR", channel=7))
    assert pool.pending_tasks["CLEAR"][-1].channel == 7
