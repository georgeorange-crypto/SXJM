"""Model layer: the actor-critic network over the FeatureBundle.

Checks the frozen wiring produces the documented shapes, that the pointer actor
masks padding out of the policy distribution, that gradients flow through the
whole stack, that the default build is PPO+MLP, and that the three swap-point
plugins are registered while an unknown name fails loudly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from radio_rl.candidates.generator import CandidateGenerator
from radio_rl.core.datatypes import DetectionObservation, ObservationType
from radio_rl.core.registry import CHANNEL_ENCODERS, FUSIONS, MEMORIES
from radio_rl.features import DefaultFeatureBuilder, FeatureSpec
from radio_rl.features.spec import FeatureBundle, collate
from radio_rl.geometry.belief import BeliefState
from radio_rl.models import AgentModel, build_agent_model


def _synth_bundle(n: int, spec: FeatureSpec) -> FeatureBundle:
    """A bundle with n real candidates; channel indices are 0 (a valid gather)."""
    return FeatureBundle(
        global_feats=torch.randn(spec.global_dim),
        channel_feats=torch.randn(spec.num_channels, spec.channel_dim),
        cand_feats=torch.randn(n, spec.candidate_dim),
        cand_channel_idx=torch.zeros(n, dtype=torch.long),
        cand_mask=torch.ones(n, dtype=torch.bool),
        n_real=n,
    )


def _real_bundle(builder: DefaultFeatureBuilder, belief: BeliefState) -> FeatureBundle:
    return builder.build(belief, CandidateGenerator().generate(belief))


def test_default_build_is_ppo_mlp():
    model = build_agent_model(None, FeatureSpec.default())
    assert isinstance(model, AgentModel)
    # the three swap points resolve to the Milestone-2 defaults
    assert type(model.channel_encoder).__name__ == "ChannelMLPEncoder"
    assert type(model.memory).__name__ == "NoMemory"
    assert type(model.fusion).__name__ == "ConcatFusion"


def test_policy_value_shapes_on_real_bundles():
    spec = FeatureSpec.default()
    model = build_agent_model(None, spec)
    builder = DefaultFeatureBuilder(spec)

    b0 = BeliefState(problem=3)                       # fresh: all undetected
    b1 = BeliefState(problem=3)
    b1.update(DetectionObservation(ObservationType.TOO_STRONG, 5, 100.0, 0.0, 6.0))
    batch = collate([_real_bundle(builder, b0), _real_bundle(builder, b1)])

    logits, value, enc = model.policy_value(batch)
    n = batch.cand_feats.shape[1]
    assert logits.shape == (2, n)
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    assert torch.isfinite(logits[batch.cand_mask]).all()   # real logits are finite
    assert enc.cand_embed.shape[:2] == (2, n)


def test_pointer_actor_masks_padding():
    spec = FeatureSpec.default()
    model = build_agent_model(None, spec).eval()
    batch = collate([_synth_bundle(2, spec), _synth_bundle(6, spec)])
    assert batch.cand_feats.shape[1] == 6
    assert not bool(batch.cand_mask[0, 2:].any())          # row 0 has 4 padded slots

    with torch.no_grad():
        logits, _, _ = model.policy_value(batch)

    sentinel = torch.finfo(logits.dtype).min
    assert torch.all(logits[0, 2:] == sentinel)            # padding driven to sentinel
    probs = torch.softmax(logits, dim=-1)
    assert probs[0, 2:].sum().item() < 1e-6                # padding gets ~zero mass
    assert probs[0, :2].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_gradients_flow_through_evaluate_actions():
    spec = FeatureSpec.default()
    model = build_agent_model(None, spec)
    batch = collate([_synth_bundle(4, spec), _synth_bundle(3, spec)])
    actions = torch.zeros(2, dtype=torch.long)             # index 0 is real in both

    log_prob, entropy, value = model.evaluate_actions(batch, actions)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2,)

    loss = -log_prob.mean() + value.pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected at least one parameter to receive a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_hidden_dim_override_forwards():
    spec = FeatureSpec.default()
    model = build_agent_model({"hidden_dim": 64}, spec)
    batch = collate([_synth_bundle(3, spec)])
    logits, value, _ = model.policy_value(batch)
    assert logits.shape == (1, 3)
    assert value.shape == (1,)


def test_swap_point_plugins_registered():
    import radio_rl.models  # noqa: F401  (import registers the default plugins)

    assert "channel_mlp" in CHANNEL_ENCODERS
    assert "none" in MEMORIES
    assert "concat" in FUSIONS


def test_unknown_plugin_name_raises():
    with pytest.raises(KeyError):
        build_agent_model({"encoder": {"type": "no_such_encoder"}}, FeatureSpec.default())
