import pytest
from way4.planner import solve_endgame


def test_small_endgame_uses_exact_open_route():
    plan = solve_endgame((0., 0.), [(10., 0.), (10., 10.), (20., 0.)])
    assert plan.used_exact
    assert len(plan.order) == 3
    assert plan.route_length == pytest.approx(10.0 + 10.0 * 2 ** 0.5 + 10.0)


def test_empty_endgame_is_complete():
    assert solve_endgame((0., 0.), []).route_length == 0.0


def test_large_endgame_uses_deterministic_approximation():
    plan = solve_endgame((0., 0.), [(float(i), 0.) for i in range(5)], exact_limit=3)
    assert not plan.used_exact
    assert len(plan.order) == 5
