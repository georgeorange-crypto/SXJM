"""Advanced channel encoders: Transformer / CNN / GNN across channels.

Each attends or convolves across the ``C`` channel rows but must keep the frozen
:class:`ChannelEncoder` contract ``[B, C, in] -> [B, C, out]`` exactly, so it drops
into the agent model by a one-line config swap. We check the shape/out_dim
contract, finiteness, that gradients flow, that ``out_dim`` overrides, and that
each builds into a full :class:`AgentModel` and produces valid policy/value.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from radio_rl.core.registry import CHANNEL_ENCODERS
from radio_rl.features import FeatureSpec
from radio_rl.features.spec import FeatureBundle, collate
from radio_rl.models import build_agent_model

ADVANCED = ["channel_transformer", "channel_cnn", "channel_gnn"]
CLASSNAME = {
    "channel_transformer": "ChannelTransformerEncoder",
    "channel_cnn": "ChannelCNNEncoder",
    "channel_gnn": "ChannelGNNEncoder",
}


def _synth_bundle(n: int, spec: FeatureSpec) -> FeatureBundle:
    return FeatureBundle(
        global_feats=torch.randn(spec.global_dim),
        channel_feats=torch.randn(spec.num_channels, spec.channel_dim),
        cand_feats=torch.randn(n, spec.candidate_dim),
        cand_channel_idx=torch.zeros(n, dtype=torch.long),
        cand_mask=torch.ones(n, dtype=torch.bool),
        n_real=n,
    )


def test_all_advanced_encoders_registered():
    for name in ADVANCED:
        assert name in CHANNEL_ENCODERS


@pytest.mark.parametrize("name", ADVANCED)
def test_encoder_contract_shape_and_grad(name):
    spec = FeatureSpec.default()
    enc = CHANNEL_ENCODERS.create(
        name, in_dim=spec.channel_dim, hidden_dim=32, num_channels=spec.num_channels
    )
    assert enc.out_dim == 32
    x = torch.randn(4, spec.num_channels, spec.channel_dim)
    y = enc(x)
    assert y.shape == (4, spec.num_channels, 32)          # [B, C, in] -> [B, C, out]
    assert torch.isfinite(y).all()

    y.sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads, "expected the encoder to receive gradients"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", ADVANCED)
def test_encoder_out_dim_override(name):
    spec = FeatureSpec.default()
    enc = CHANNEL_ENCODERS.create(
        name, in_dim=spec.channel_dim, hidden_dim=32, out_dim=24,
        num_channels=spec.num_channels,
    )
    assert enc.out_dim == 24
    y = enc(torch.randn(2, spec.num_channels, spec.channel_dim))
    assert y.shape == (2, spec.num_channels, 24)


@pytest.mark.parametrize("name", ADVANCED)
def test_encoder_builds_into_agent_model(name):
    spec = FeatureSpec.default()
    model = build_agent_model({"encoder": {"type": name}}, spec)
    assert type(model.channel_encoder).__name__ == CLASSNAME[name]

    batch = collate([_synth_bundle(3, spec), _synth_bundle(5, spec)])
    logits, value, enc = model.policy_value(batch)
    n = batch.cand_feats.shape[1]
    assert logits.shape == (2, n)
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    assert torch.isfinite(logits[batch.cand_mask]).all()
    assert enc.channel_embed.shape == (2, spec.num_channels, model.channel_encoder.out_dim)


def test_transformer_positional_embedding_handles_fewer_channels():
    # The learnable positional table is sized for num_channels; a shorter channel
    # axis must still work (slice), so the encoder is robust to the config value.
    enc = CHANNEL_ENCODERS.create(
        "channel_transformer", in_dim=6, hidden_dim=16, num_channels=20
    )
    y = enc(torch.randn(2, 12, 6))          # only 12 channels present
    assert y.shape == (2, 12, 16)
    assert torch.isfinite(y).all()
