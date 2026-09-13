import pytest
from way4.metrics import route_regret


def test_route_regret_counts_time_and_plan_degradation():
    assert route_regret(20., 100., 130.) == 50.
    assert route_regret(20., 100., 80.) == 0.


def test_route_regret_rejects_invalid_time():
    with pytest.raises(ValueError):
        route_regret(-1., 1., 1.)
    with pytest.raises(ValueError):
        route_regret(float('nan'), 1., 1.)
