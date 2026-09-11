"""Learned world models: residual corrections and the Dreamer latent model.

The safety-critical properties, checked here:

* :class:`build_world_model` default and ``analytical`` stay **torch-free** (a
  subprocess proves no ``torch`` module leaks in) — the learned models are
  imported lazily by name only when selected.
* an **untrained** ``ResidualWorldModel`` reproduces the analytical predictions
  *exactly* (zero-init correction head), so selecting it can never degrade the
  baseline; ``fit`` then moves the weights and shifts the prediction.
* ``DreamerWorldModel`` annotates candidates **identically** to the analytical
  model (it inherits the physics), while its latent dynamics can imagine rollouts
  and learn from observed transitions.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap

import pytest

from radio_rl.core.datatypes import ActionType, CandidateAction, CandidateSource
from radio_rl.core.registry import WORLD_MODELS
from radio_rl.geometry.belief import BeliefState
from radio_rl.world_model import build_world_model
from radio_rl.world_model.analytical import AnalyticalWorldModel


def _localized_belief(channel: int = 3):
    """A 'too strong' reading localizes ``channel`` at (100, 0) (estimate + MEC)."""
    from radio_rl.core.datatypes import DetectionObservation, ObservationType

    b = BeliefState(problem=3)
    b.update(DetectionObservation(ObservationType.TOO_STRONG, channel, 100.0, 0.0, 6.0))
    return b


# --- the torch-free guarantee (the whole point of lazy-by-name loading) -----
def test_default_world_model_is_torch_free():
    script = textwrap.dedent(
        """
        import sys
        from radio_rl.world_model import build_world_model
        build_world_model()                       # default: analytical
        build_world_model({"type": "analytical"})
        leaked = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
        assert not leaked, "world_model path imported torch: " + repr(leaked)
        print("WM_TORCH_FREE_OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WM_TORCH_FREE_OK" in proc.stdout


def test_build_default_is_analytical():
    assert type(build_world_model(None)).__name__ == "AnalyticalWorldModel"
    assert type(build_world_model({"type": "analytical"})).__name__ == "AnalyticalWorldModel"


# --- residual world model ---------------------------------------------------
def test_residual_untrained_equals_analytical():
    pytest.importorskip("torch")
    ana = AnalyticalWorldModel()
    res = build_world_model({"type": "residual"})
    assert type(res).__name__ == "ResidualWorldModel"
    assert "residual" in WORLD_MODELS

    b = _localized_belief(3)
    ex, ey = b.channels[3].estimate
    for at in (int(ActionType.SCAN), int(ActionType.CLEAR), int(ActionType.EXIT)):
        assert res.predict_time(b, at, 3, ex, ey) == pytest.approx(
            ana.predict_time(b, at, 3, ex, ey))

    # clear probability: at the estimate (pinned to 1.0) and offset (mid-range 0..1)
    for tx, ty in [(ex, ey), (ex + 20.0, ey)]:
        assert res.predict_clear_probability(b, 3, tx, ty) == pytest.approx(
            ana.predict_clear_probability(b, 3, tx, ty))
    assert res.predict_region_reduction(b, 3, ex, ey) == pytest.approx(
        ana.predict_region_reduction(b, 3, ex, ey))


def test_residual_untrained_annotate_equals_analytical():
    pytest.importorskip("torch")
    ana = AnalyticalWorldModel()
    res = build_world_model({"type": "residual"})
    b = _localized_belief(3)
    ex, ey = b.channels[3].estimate

    def cand():
        return CandidateAction(candidate_id=0, action_type=int(ActionType.CLEAR),
                               channel=3, target_x=ex, target_y=ey,
                               source=int(CandidateSource.CLEAR))

    a, r = ana.annotate(b, cand()), res.annotate(b, cand())
    assert r.expected_time == pytest.approx(a.expected_time)
    assert r.predicted_clear_probability == pytest.approx(a.predicted_clear_probability)
    assert r.predicted_region_reduction == pytest.approx(a.predicted_region_reduction)
    assert r.heuristic_information_gain == pytest.approx(a.heuristic_information_gain)


def test_residual_fit_moves_weights_and_prediction():
    torch = pytest.importorskip("torch")
    ana = AnalyticalWorldModel()
    res = build_world_model({"type": "residual"})
    b = _localized_belief(3)
    ex, ey = b.channels[3].estimate
    base_t = ana.predict_time(b, int(ActionType.SCAN), 3, ex, ey)
    assert base_t > 0.0

    before = torch.cat([p.detach().flatten() for p in res.net.parameters()]).clone()
    samples = [{"belief": b, "channel": 3, "target_x": ex, "target_y": ey,
                "time": 2.0 * base_t} for _ in range(8)]
    info = res.fit(samples, steps=80, lr=1e-2)
    after = torch.cat([p.detach().flatten() for p in res.net.parameters()])

    assert info["n"] == 8 and math.isfinite(info["loss_time"])
    assert not torch.equal(before, after)                         # a real step happened
    # target time is 2x the base, so the correction should push the prediction up
    assert res.predict_time(b, int(ActionType.SCAN), 3, ex, ey) > base_t


def test_residual_fit_empty_is_noop():
    pytest.importorskip("torch")
    res = build_world_model({"type": "residual"})
    assert res.fit([]) == {"n": 0}


# --- dreamer latent world model ---------------------------------------------
def test_dreamer_annotate_matches_analytical():
    pytest.importorskip("torch")
    ana = AnalyticalWorldModel()
    dm = build_world_model({"type": "dreamer"})
    assert type(dm).__name__ == "DreamerWorldModel"
    assert "dreamer" in WORLD_MODELS

    b = _localized_belief(5)
    ex, ey = b.channels[5].estimate
    for src, at in [(CandidateSource.CLEAR, ActionType.CLEAR),
                    (CandidateSource.LOCALIZE, ActionType.SCAN)]:
        def cand():
            return CandidateAction(candidate_id=1, action_type=int(at), channel=5,
                                   target_x=ex, target_y=ey, source=int(src))
        a, d = ana.annotate(b, cand()), dm.annotate(b, cand())
        assert d.expected_time == pytest.approx(a.expected_time)
        assert d.predicted_clear_probability == pytest.approx(a.predicted_clear_probability)
        assert d.predicted_region_reduction == pytest.approx(a.predicted_region_reduction)
        assert d.heuristic_information_gain == pytest.approx(a.heuristic_information_gain)


def test_dreamer_imagine_shapes():
    torch = pytest.importorskip("torch")
    dm = build_world_model({"type": "dreamer"})
    latent, action = dm.dynamics.latent_dim, dm.dynamics.action_dim
    h0 = dm.initial_latent(2)
    assert h0.shape == (2, latent)

    def policy(h):
        return torch.zeros(h.shape[0], action)

    lat, rew = dm.imagine(h0, policy, horizon=4)
    assert lat.shape == (2, 4, latent)
    assert rew.shape == (2, 4)
    assert torch.isfinite(lat).all() and torch.isfinite(rew).all()


def test_dreamer_learn_dynamics_moves_weights():
    torch = pytest.importorskip("torch")
    dm = build_world_model({"type": "dreamer"})
    B, T = 3, 5
    obs = torch.randn(B, T, dm.dynamics.obs_dim)
    act = torch.randn(B, T, dm.dynamics.action_dim)
    rew = torch.randn(B, T)

    before = torch.cat([p.detach().flatten() for p in dm.dynamics.parameters()]).clone()
    info = dm.learn_dynamics(obs, act, rew, steps=10, lr=1e-3)
    after = torch.cat([p.detach().flatten() for p in dm.dynamics.parameters()])

    assert info["horizon"] == T and info["batch"] == B
    assert math.isfinite(info["loss"])
    assert not torch.equal(before, after)
