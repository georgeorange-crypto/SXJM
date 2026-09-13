from way4.core import MacroActionType, MacroCandidate
from way4.planner import EndgameController


def test_endgame_controller_selects_exact_route_first_candidate():
    candidates = [MacroCandidate(MacroActionType.EXPLORE, p, expected_time=1.)
                  for p in [(10., 0.), (10., 10.), (20., 0.)]]
    selected = EndgameController(unresolved_threshold=3).select((0., 0.), candidates, 3)
    assert selected.target == (10., 0.)


def test_endgame_controller_stays_inactive_above_threshold():
    assert EndgameController(3).select((0., 0.), [], 4) is None
