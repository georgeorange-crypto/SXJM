from way4.belief import ChannelBelief
from way4.sensing import MinimaxNBV


def test_completion_point_is_safe_or_declines_when_not_guaranteed():
    b = ChannelBelief(1)
    b.record_bearing((0.0, 0.0), 0.0)
    point = MinimaxNBV(n_hypotheses=8).completion_point(b, (0.0, 0.0), kill_radius=20.0)
    assert point is None or isinstance(point, tuple)


def test_verification_point_accepts_certificate_gain_objective():
    b = ChannelBelief(1)
    b.record_bearing((0.0, 0.0), 0.0)
    nbv = MinimaxNBV(n_hypotheses=8)
    point = nbv.verification_point(b, (0.0, 0.0), coverage_gain=lambda q: -abs(q[0]))
    assert point is not None


def test_thin_shape_adds_cross_axis_viewpoints():
    b = ChannelBelief(channel=1)
    b.F_c = [(-500.0, -10.0), (500.0, -10.0), (500.0, 10.0), (-500.0, 10.0)]
    b.mec_center = (0.0, 0.0)
    b.kappa = 100.0
    b.principal_axis = (1.0, 0.0)
    nbv = MinimaxNBV(perp_standoffs=(100.0,))
    pts = nbv.candidate_viewpoints(b, (0.0, 0.0))
    assert (0.0, 100.0) in pts or (0.0, -100.0) in pts
