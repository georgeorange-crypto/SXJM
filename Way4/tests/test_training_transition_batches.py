from types import SimpleNamespace
import sys

from scripts import train_candidate_ppo as training
from way4.rl.candidate_rollout import CandidateTransition


def test_update_waits_for_transition_threshold_and_persists_each_episode(monkeypatch):
    calls, saved, batches = [], [], []

    class Model:
        def state_dict(self): return {}
        def load_state_dict(self, state): pass

    class Trainer:
        def __init__(self, *args): pass
        def update(self, obs, *args):
            batches.append(len(obs))
            return {'n': len(obs)}

    def episode(model, seed, max_steps, **kwargs):
        calls.append(seed)
        ts = [CandidateTransition([[1.]], 0, 0., 0., -.1) for _ in range(3)]
        return ts, SimpleNamespace(virtual_time_s=10.), SimpleNamespace(cleared_count=1, total=1)

    monkeypatch.setattr(training, 'CandidateActorCritic', lambda _: Model())
    monkeypatch.setattr(training, 'CandidatePPOTrainer', Trainer)
    monkeypatch.setattr(training, 'episode', episode)
    monkeypatch.setattr(training, 'validate_model', lambda *args: ([], {'passed': False}))
    monkeypatch.setattr(training, 'save_candidate_checkpoint',
                        lambda *a, **kw: saved.append((kw['transitions'], kw['complete'])))
    monkeypatch.setattr(sys, 'argv', ['train', '--profile', 'smoke',
        '--seeds', '10000,10001,10002', '--validation-seeds', '11000',
        '--transitions-per-update', '7', '--updates', '2', '--episodes-per-update', '1'])
    training.main()
    assert calls == [10000, 10001, 10002, 10000, 10001, 10002]
    assert batches == [9, 9]
    assert (3, False) in saved and (6, False) in saved
    assert saved[-1] == (18, True)
