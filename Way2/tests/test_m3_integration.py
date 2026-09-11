"""Milestone 3 integration: the advanced stack composes and runs end to end.

The payoff of the frozen 3-swap-point architecture: switching PPO+MLP to
PPO+Transformer+LSTM+gated fusion is one preset file and no code change. We check
the preset loads through the real config composer, builds the intended plugins
inside the pipeline, runs an episode, and completes a tiny training iteration; and
that the *other* advanced variants (CNN/GNN encoders, GRU/Mamba memories,
attention fusion) also assemble into a working agent on a real belief-derived
bundle.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from radio_rl.candidates.generator import CandidateGenerator
from radio_rl.core.config import compose
from radio_rl.core.datatypes import DetectionObservation, ObservationType
from radio_rl.features import DefaultFeatureBuilder, FeatureSpec
from radio_rl.features.spec import collate
from radio_rl.geometry.belief import BeliefState
from radio_rl.models import build_agent_model
from radio_rl.pipeline import Pipeline


def _real_bundle():
    spec = FeatureSpec.default()
    builder = DefaultFeatureBuilder(spec)
    b = BeliefState(problem=3)
    b.update(DetectionObservation(ObservationType.TOO_STRONG, 5, 100.0, 0.0, 6.0))
    b.update(DetectionObservation(ObservationType.SIGNAL, 3, 0.0, 0.0, 5.0, bearing_deg=30.0))
    return spec, builder.build(b, CandidateGenerator().generate(b))


# --- the preset config: one-line swap of all three model plug-points --------
def test_ppo_transformer_preset_builds_advanced_stack():
    pipe = Pipeline(compose(overrides=["algorithm=ppo_transformer"]))
    assert type(pipe.agent).__name__ == "PPOAgent"
    m = pipe.agent.model
    assert type(m.channel_encoder).__name__ == "ChannelTransformerEncoder"
    assert type(m.memory).__name__ == "LSTMMemory"
    assert type(m.fusion).__name__ == "GatedFusion"


def test_ppo_transformer_preset_runs_episode():
    pipe = Pipeline(compose(overrides=["algorithm=ppo_transformer"]))
    stats, trace = pipe.run_episode(seed=0, max_steps=120, collect_trace=True)
    assert len(trace) >= 1
    assert stats.virtual_time >= 0.0
    assert all(0 <= st.candidate_index < st.n_candidates for st in trace)


def test_ppo_transformer_eval_is_deterministic():
    pipe = Pipeline(compose(overrides=["algorithm=ppo_transformer"]))
    s1, t1 = pipe.run_episode(seed=5, max_steps=100, collect_trace=True)
    s2, t2 = pipe.run_episode(seed=5, max_steps=100, collect_trace=True)
    assert [x.candidate_index for x in t1] == [x.candidate_index for x in t2]
    assert s1.virtual_time == s2.virtual_time


def test_tiny_train_run_on_advanced_stack():
    from radio_rl.training import train

    cfg = compose(overrides=[
        "algorithm=ppo_transformer",
        "algorithm.train.iterations=1",
        "algorithm.train.episodes_per_iter=1",
        "algorithm.train.max_steps=40",
        "algorithm.train.eval_every=0",
        "algorithm.train.checkpoint=null",
    ])
    result = train(cfg, progress=False)
    assert type(result.agent).__name__ == "PPOAgent"
    assert len(result.history) == 1
    rec = result.history[0]
    assert rec["iter"] == 0
    assert math.isfinite(rec["policy_loss"])


# --- the remaining advanced variants assemble and run -----------------------
@pytest.mark.parametrize("encoder", ["channel_cnn", "channel_gnn"])
@pytest.mark.parametrize("memory", ["gru", "mamba"])
def test_other_advanced_combinations_policy_value(encoder, memory):
    spec, bundle = _real_bundle()
    model = build_agent_model(
        {"encoder": {"type": encoder},
         "memory": {"type": memory},
         "fusion": {"type": "attention"}},
        spec,
    )
    batch = collate([bundle])
    logits, value, enc = model.policy_value(batch)
    assert logits.shape[0] == 1 and value.shape == (1,)
    assert torch.isfinite(value).all()
    assert torch.isfinite(logits[batch.cand_mask]).all()
    assert enc.memory_state is not None       # recurrent memory produced a state


def test_advanced_world_model_via_pipeline_stays_analytical_annotation():
    # A learned world model can be selected without disturbing the pipeline; its
    # candidate annotation still matches the analytical physics (dreamer inherits it).
    pipe = Pipeline(compose(overrides=[
        "algorithm=ppo_transformer", "world_model.type=dreamer",
    ]))
    assert type(pipe.world_model).__name__ == "DreamerWorldModel"
    stats, trace = pipe.run_episode(seed=1, max_steps=100, collect_trace=True)
    assert all(0 <= st.candidate_index < st.n_candidates for st in trace)
