from types import SimpleNamespace
from way4.planner.receding_horizon import RecedingHorizonPlanner


def test_cost_to_go_cache_is_scoped_to_plan_call():
    p = RecedingHorizonPlanner()
    calls = []
    p.fce.estimate = lambda view: calls.append(view) or SimpleNamespace(total=7.)
    view = SimpleNamespace(pos=(0., 0.), clearable_targets=[], detected=[],
                           unknown_holes=[], n_unknown=0, anchors=[])
    assert p._cost_to_go(view, 0) == 7.
    assert p._cost_to_go(view, 0) == 7.
    assert len(calls) == 1
    p._cost_cache = {}
    assert p._cost_to_go(view, 0) == 7.
    assert len(calls) == 2
