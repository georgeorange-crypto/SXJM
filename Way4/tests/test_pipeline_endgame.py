from way4.core import MacroActionType, MacroCandidate
from way4.pipeline import Way4Pipeline


class Env:
    def measure(self, x, y, channel):
        return (False, 0.0, 0.0), 1.0


def test_final_ppo_pipeline_uses_endgame_controller_when_few_unresolved(monkeypatch):
    pipe = Way4Pipeline(Env(), n_channels=3, planner_mode="final_ppo")
    candidates = [MacroCandidate(MacroActionType.EXPLORE, p, scan_channels=(1,))
                  for p in [(10., 0.), (10., 10.), (20., 0.)]]
    monkeypatch.setattr(pipe.generator, "generate", lambda *args, **kwargs: candidates)
    selected = pipe._choose_macro()
    assert selected.meta["endgame_solver"] == "exact_open_route"
