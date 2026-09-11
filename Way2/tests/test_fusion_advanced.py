"""Advanced fusions: gated and attention over the context parts.

Both replace the fixed concat-and-project with data-dependent weighting of the
context parts (global embed, channel summary, memory output), sharing the frozen
``list[[B, di]] -> [B, out]`` contract. We check the shape/out_dim contract,
finiteness, gradient flow, and that each builds into a full :class:`AgentModel`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from radio_rl.core.registry import FUSIONS
from radio_rl.features import FeatureSpec
from radio_rl.features.spec import FeatureBundle, collate
from radio_rl.models import build_agent_model

ADVANCED = ["gated", "attention"]
CLASSNAME = {"gated": "GatedFusion", "attention": "AttentionFusion"}


def _synth_bundle(n: int, spec: FeatureSpec) -> FeatureBundle:
    return FeatureBundle(
        global_feats=torch.randn(spec.global_dim),
        channel_feats=torch.randn(spec.num_channels, spec.channel_dim),
        cand_feats=torch.randn(n, spec.candidate_dim),
        cand_channel_idx=torch.zeros(n, dtype=torch.long),
        cand_mask=torch.ones(n, dtype=torch.bool),
        n_real=n,
    )


def test_advanced_fusions_registered():
    for name in ADVANCED:
        assert name in FUSIONS


@pytest.mark.parametrize("name", ADVANCED)
def test_fusion_contract_shape_and_grad(name):
    fus = FUSIONS.create(name, in_dims=[4, 6, 8], out_dim=5)
    assert fus.out_dim == 5
    parts = [torch.randn(3, 4), torch.randn(3, 6), torch.randn(3, 8)]
    out = fus(parts)
    assert out.shape == (3, 5)                     # list[[B, di]] -> [B, out]
    assert torch.isfinite(out).all()

    out.sum().backward()
    grads = [p.grad for p in fus.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", ADVANCED)
def test_fusion_handles_varied_part_widths(name):
    # parts of different widths (as the agent supplies: H, He, Hm may differ)
    fus = FUSIONS.create(name, in_dims=[3, 3, 7], out_dim=9)
    out = fus([torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 7)])
    assert out.shape == (2, 9)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ADVANCED)
def test_fusion_builds_into_agent_model(name):
    spec = FeatureSpec.default()
    model = build_agent_model({"fusion": {"type": name}}, spec)
    assert type(model.fusion).__name__ == CLASSNAME[name]

    batch = collate([_synth_bundle(3, spec), _synth_bundle(6, spec)])
    logits, value, _ = model.policy_value(batch)
    n = batch.cand_feats.shape[1]
    assert logits.shape == (2, n)
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    assert torch.isfinite(logits[batch.cand_mask]).all()
