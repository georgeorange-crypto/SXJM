from way4.sensing import MinimaxNBV, ViewpointOpportunity


class C:
    F_c = [(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)]
    mec_center = (0.0, 0.0)
    mec_radius = 30.0
    bearings = []
    negative_discs = []


def test_propose_returns_valid_bounded_opportunity_set():
    out = MinimaxNBV(ring_radii=(40.0,), n_ring_angles=4).propose(C(), (100.0, 0.0), limit=4)
    assert 0 < len(out) <= 4
    assert all(isinstance(x, ViewpointOpportunity) for x in out)
    assert all(x.information_gain >= 0 and x.route_distance >= 5 for x in out)


def test_propose_is_deterministic():
    nbv = MinimaxNBV(ring_radii=(40.0,), n_ring_angles=4)
    assert nbv.propose(C(), (100.0, 0.0), limit=8) == nbv.propose(C(), (100.0, 0.0), limit=8)
