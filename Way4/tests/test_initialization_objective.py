from way4.belief import ChannelBelief, ChannelStatus
from way4.sensing import MinimaxNBV


def test_initialized_channel_has_crossing_quality_point():
    b = ChannelBelief(1)
    b.record_bearing((0.0, 0.0), 0.0)
    b.record_bearing((0.0, 500.0), 90.0)
    assert b.status == ChannelStatus.INITIALIZED
    point = MinimaxNBV(n_hypotheses=8).initialization_point(b, (0.0, 0.0))
    assert point is not None
