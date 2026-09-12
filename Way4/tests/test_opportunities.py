from way4.core import MacroActionType, MacroCandidate
from way4.planner import RemainingTask, insertion_cost, pareto_prune


def test_remaining_task_is_explicit_and_immutable():
    t = RemainingTask("REFINE", ((1.0, 2.0),), channel=3, mandatory=False)
    assert t.channel == 3 and t.mandatory is False


def test_insertion_cost_uses_route_segment_and_tail():
    assert insertion_cost([(0.0, 0.0), (100.0, 0.0)], (50.0, 0.0)) == 0.0
    assert insertion_cost([(0.0, 0.0), (100.0, 0.0)], (150.0, 0.0)) == 50.0


def test_pareto_prune_removes_strictly_dominated_candidate():
    a = MacroCandidate(MacroActionType.REFINE, (0, 0), route_marginal=10, refinement_gain=1)
    b = MacroCandidate(MacroActionType.REFINE, (1, 0), route_marginal=5, refinement_gain=2)
    c = MacroCandidate(MacroActionType.REFINE, (2, 0), route_marginal=20, refinement_gain=0.5)
    assert pareto_prune([a, b, c]) == [b]
