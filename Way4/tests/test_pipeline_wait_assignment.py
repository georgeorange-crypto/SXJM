from way4.belief import BeliefState, ChannelStatus
from way4.planner import RemainingTaskPool


def test_wait_assignment_only_accepts_candidates_that_service_channel():
    belief = BeliefState(n_channels=1)
    belief[1].status = ChannelStatus.DETECTED
    pool = RemainingTaskPool()
    pool.sync(belief)
    route = [(0.0, 0.0), (100.0, 0.0), (500.0, 0.0)]
    assignments = pool.assign_route_opportunities(
        route, {1: 100.0}, viable=lambda c, p: p == (100.0, 0.0)
    )
    assert assignments == {1: (100.0, 0.0)}
    assert pool.waiting[1].assigned_waypoint == (100.0, 0.0)
