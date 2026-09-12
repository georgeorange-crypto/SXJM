from way4.toolbox import beam_search, mpc, HyperHeuristic, shield, bounded_residual

def test_bounded_planning_and_safety():
    assert beam_search(0, lambda x: [x+1, x+2], lambda x: -x, 2, 2) == 4
    state, actions = mpc(0, lambda s,h: [s+1], lambda s,a: a, lambda s: s >= 3)
    assert state == 3 and actions == [1,2,3]
    assert shield("bad", lambda x: x == "good", "safe") == "safe"
    assert bounded_residual(10, 100, 2) == 12
    h = HyperHeuristic({"a": lambda _: 1})
    assert h.choose(None) == ("a", 1)
