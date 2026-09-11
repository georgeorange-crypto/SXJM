"""PPO agent + training: the learnable path end to end.

Covers the agent selecting inside the frozen pipeline (proving the pipeline
lazily builds a matching feature builder), the eval path's determinism, the GAE
and reward maths, one PPOTrainer update actually moving weights, and a tiny
``train()`` run — all behind ``pytest.importorskip('torch')`` so the mathematical
core suite stays torch-free.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from radio_rl.algorithms.ppo import PPOAgent
from radio_rl.candidates.generator import CandidateGenerator
from radio_rl.core.config import compose
from radio_rl.core.datatypes import (
    Action,
    ActionType,
    DetectionObservation,
    EpisodeStats,
    ObservationType,
)
from radio_rl.features import DefaultFeatureBuilder, FeatureSpec
from radio_rl.geometry.belief import BeliefState
from radio_rl.models import build_agent_model
from radio_rl.pipeline import Pipeline, StepTrace
from radio_rl.training import (
    PPOConfig,
    PPOTrainer,
    RewardConfig,
    Transition,
    compute_gae,
    episode_rewards,
    train,
)


def _ppo_cfg(**overrides):
    tokens = ["algorithm=ppo", *[f"{k}={v}" for k, v in overrides.items()]]
    return compose(overrides=tokens)


def _transitions_from_one_belief(n: int) -> list[Transition]:
    """n transitions sharing a real bundle; action 0 is always a valid index."""
    builder = DefaultFeatureBuilder()
    b = BeliefState(problem=3)
    b.update(DetectionObservation(ObservationType.SIGNAL, 3, 0.0, 0.0, 5.0, bearing_deg=30.0))
    bundle = builder.build(b, CandidateGenerator().generate(b))
    return [
        Transition(bundle=bundle, action=0, log_prob=0.0, value=0.0,
                   reward=1.0, advantage=0.5, ret=1.0)
        for _ in range(n)
    ]


# --- the pipeline wiring ----------------------------------------------------
def test_pipeline_builds_ppo_and_feature_builder():
    pipe = Pipeline(_ppo_cfg())
    assert type(pipe.agent).__name__ == "PPOAgent"
    assert pipe.agent.needs_features is True
    # the pipeline lazily obtained a matching feature builder from the agent
    assert type(pipe.feature_builder).__name__ == "DefaultFeatureBuilder"
    assert pipe.feature_builder.spec.candidate_dim == pipe.agent.spec.candidate_dim


def test_ppo_agent_runs_episode():
    pipe = Pipeline(_ppo_cfg())
    stats, trace = pipe.run_episode(seed=0, max_steps=250, collect_trace=True)
    assert len(trace) >= 1                       # the network drove at least one step
    assert stats.virtual_time >= 0.0
    assert 0 <= stats.sources_cleared <= max(stats.sources_total, stats.sources_cleared)
    # every chosen index was inside the offered candidate set
    assert all(0 <= st.candidate_index < st.n_candidates for st in trace)


def test_ppo_eval_is_deterministic():
    pipe = Pipeline(_ppo_cfg())                  # untrained weights, argmax eval
    s1, t1 = pipe.run_episode(seed=7, max_steps=150, collect_trace=True)
    s2, t2 = pipe.run_episode(seed=7, max_steps=150, collect_trace=True)
    assert len(t1) == len(t2)
    assert [x.candidate_index for x in t1] == [x.candidate_index for x in t2]
    assert s1.virtual_time == s2.virtual_time


# --- reward shaping & GAE ---------------------------------------------------
def test_gae_matches_hand_computation():
    # gamma = lam = 1  =>  advantage_i = sum(future rewards) - value_i,
    # return_i = advantage_i + value_i = sum(future rewards).
    ts = [
        Transition(bundle=None, action=0, log_prob=0.0, value=0.5, reward=1.0),
        Transition(bundle=None, action=0, log_prob=0.0, value=0.3, reward=2.0),
    ]
    compute_gae(ts, gamma=1.0, lam=1.0)
    assert ts[1].advantage == pytest.approx(1.7)   # 2.0 - 0.3
    assert ts[1].ret == pytest.approx(2.0)         # r1
    assert ts[0].advantage == pytest.approx(2.5)   # 1 + 0.3 - 0.5 + 1.7
    assert ts[0].ret == pytest.approx(3.0)         # r0 + r1


def test_episode_rewards_exit_step_and_clear_bonus():
    rc = RewardConfig()   # time_coef 1, time_scale 1000, clear_success_bonus 1, success_bonus 5
    trace = [
        StepTrace(
            action=Action(ActionType.SCAN, 3, 0.0, 0.0),
            observation=DetectionObservation(ObservationType.SIGNAL, 3, 0.0, 0.0, 100.0),
            candidate_index=0, n_candidates=4, virtual_time_s=100.0,
        ),
        StepTrace(
            action=Action(ActionType.CLEAR, 5, 10.0, 0.0),
            observation=DetectionObservation(ObservationType.CLEAR_SUCCESS, 5, 10.0, 0.0, 50.0),
            candidate_index=1, n_candidates=4, virtual_time_s=150.0,
        ),
        StepTrace(   # EXIT consumes no task time; its stored obs is stale
            action=Action(ActionType.EXIT, 5, 10.0, 0.0),
            observation=DetectionObservation(ObservationType.CLEAR_SUCCESS, 5, 10.0, 0.0, 0.0),
            candidate_index=2, n_candidates=4, virtual_time_s=150.0,
        ),
    ]
    stats = EpisodeStats(sources_total=1, sources_cleared=1)
    r = episode_rewards(trace, stats, rc)

    assert r[0] == pytest.approx(-0.1)             # -1 * 100/1000, no clear
    assert r[1] == pytest.approx(0.95)             # -50/1000 + 1.0 clear bonus
    # EXIT charged no time AND its stale CLEAR_SUCCESS ignored -> only terminal 5*1.0
    assert r[2] == pytest.approx(5.0)


# --- the PPO update & the training loop -------------------------------------
def test_ppo_trainer_update_moves_weights():
    spec = FeatureSpec.default()
    model = build_agent_model(None, spec)
    agent = PPOAgent(model, spec, feature_cfg=None, device="cpu")
    trainer = PPOTrainer(agent, PPOConfig(epochs=2, minibatch_size=4))

    before = next(model.parameters()).detach().clone()
    metrics = trainer.update(_transitions_from_one_belief(8))
    after = next(model.parameters()).detach()

    assert metrics["n"] == 8
    assert math.isfinite(metrics["policy_loss"])
    assert math.isfinite(metrics["value_loss"])
    assert math.isfinite(metrics["entropy"])
    assert not torch.equal(before, after)          # a real optimisation step happened


def test_empty_update_is_a_noop():
    spec = FeatureSpec.default()
    agent = PPOAgent(build_agent_model(None, spec), spec, device="cpu")
    metrics = PPOTrainer(agent, PPOConfig()).update([])
    assert metrics["n"] == 0


def test_tiny_train_run():
    cfg = _ppo_cfg(**{
        "algorithm.train.iterations": 1,
        "algorithm.train.episodes_per_iter": 1,
        "algorithm.train.max_steps": 40,
        "algorithm.train.eval_every": 0,
        "algorithm.train.checkpoint": "null",   # OmegaConf parses to None -> no file
    })
    result = train(cfg, progress=False)
    assert isinstance(result.agent, PPOAgent)
    assert len(result.history) == 1
    rec = result.history[0]
    assert rec["iter"] == 0
    assert math.isfinite(rec["policy_loss"])
    assert "mean_return" in rec and "mean_clear_ratio" in rec
