from way4.belief import ChannelBelief
from way4.sensing import MinimaxNBV


def test_nbv_hypotheses_respect_no_signal_exclusions():
    belief = ChannelBelief(channel=1)
    belief.record_bearing((0.0, 0.0), 0.0)
    belief.record_no_signal((500.0, 0.0))
    result = MinimaxNBV(n_hypotheses=16).choose(belief, (0.0, 0.0))
    assert result is not None
    assert result.hypotheses
    assert all(belief.contains_possible_source(p) for p in result.hypotheses)
