from way4.core import MacroActionType, MacroCandidate
from way4.planner import RecedingHorizonPlanner


def test_beam_search_keeps_top_b_sequences_and_returns_first_action():
    p = RecedingHorizonPlanner(horizon=3, beam_width=2)
    cands = [
        MacroCandidate(MacroActionType.EXPLORE, (0., 0.), expected_time=5.),
        MacroCandidate(MacroActionType.EXPLORE, (1., 0.), expected_time=10.),
    ]
    view = type("V", (), {})()
    view.clearable_targets = []
    view.detected = []
    view.holes = []
    view.n_unknown = 0
    view.unknown_holes = []
    view.pos = (0., 0.)
    first, score, sequence = p.beam_search(cands, view)
    assert first is cands[0]
    assert len(sequence) == 3
    assert score >= 0.0
