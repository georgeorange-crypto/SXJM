from way4.belief.hypothesis import HypothesisLayer


def test_soft_posterior_support_is_alive_and_entropy_gain_is_nonnegative():
    layer = HypothesisLayer(arena_radius=100.0, spacing=20.0)
    h = layer.channel(1)
    before = h.posterior_entropy()
    assert before > 0
    assert abs(float(h.posterior_mass().sum()) - 1.0) < 1e-9
    gain = h.information_gain((0.0, 0.0))
    assert gain >= 0.0
    assert h.posterior_entropy() == before  # lookahead does not mutate belief
