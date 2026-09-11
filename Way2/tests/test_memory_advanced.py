"""Advanced temporal memories: LSTM / GRU / Mamba-style selective SSM.

Each threads an explicit state across steps and shares the frozen
``forward([B, in], state) -> ([B, out], state)`` contract, so any is a one-line
config swap. We check the shape/out_dim contract, ``state=None`` reproducibility
(the memoryless-update approximation), that a threaded state actually changes the
output, that gradients flow, and that each builds into a full :class:`AgentModel`
whose recurrent state threads through :meth:`encode`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from radio_rl.core.registry import MEMORIES
from radio_rl.features import FeatureSpec
from radio_rl.features.spec import FeatureBundle, collate
from radio_rl.models import build_agent_model

RECURRENT = ["lstm", "gru", "mamba"]
CLASSNAME = {"lstm": "LSTMMemory", "gru": "GRUMemory", "mamba": "MambaMemory"}


def _synth_bundle(n: int, spec: FeatureSpec) -> FeatureBundle:
    return FeatureBundle(
        global_feats=torch.randn(spec.global_dim),
        channel_feats=torch.randn(spec.num_channels, spec.channel_dim),
        cand_feats=torch.randn(n, spec.candidate_dim),
        cand_channel_idx=torch.zeros(n, dtype=torch.long),
        cand_mask=torch.ones(n, dtype=torch.bool),
        n_real=n,
    )


def test_all_recurrent_memories_registered():
    for name in RECURRENT:
        assert name in MEMORIES


@pytest.mark.parametrize("name", RECURRENT)
def test_memory_contract_and_state_none(name):
    mem = MEMORIES.create(name, dim=8)
    assert mem.out_dim == 8                       # out_dim defaults to dim
    x = torch.randn(3, 8)

    out, state = mem(x, None)                     # state=None starts from zeros
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()
    assert state is not None

    # state=None is deterministic (this is exactly the PPO first-state path).
    out_a, _ = mem(x, None)
    out_b, _ = mem(x, None)
    assert torch.allclose(out_a, out_b)


@pytest.mark.parametrize("name", RECURRENT)
def test_memory_threaded_state_changes_output(name):
    mem = MEMORIES.create(name, dim=8)
    x = torch.randn(2, 8)
    out0, state0 = mem(x, None)
    out1, _ = mem(x, state0)                      # same input, carried state
    assert out1.shape == (2, 8)
    assert torch.isfinite(out1).all()
    assert not torch.allclose(out0, out1), "carried state should affect the output"


@pytest.mark.parametrize("name", RECURRENT)
def test_memory_hidden_override_and_grad(name):
    mem = MEMORIES.create(name, dim=8, hidden=16)
    assert mem.out_dim == 16
    out, _ = mem(torch.randn(4, 8), None)
    assert out.shape == (4, 16)

    out.sum().backward()
    grads = [p.grad for p in mem.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", RECURRENT)
def test_memory_builds_into_agent_model_and_threads(name):
    spec = FeatureSpec.default()
    model = build_agent_model({"memory": {"type": name}}, spec)
    assert type(model.memory).__name__ == CLASSNAME[name]

    bundle = _synth_bundle(4, spec)
    b1 = collate([bundle])
    enc1 = model.encode(b1)
    assert enc1.memory_state is not None          # recurrent state produced

    # thread the state into the next step (as the pipeline's agent does)
    b2 = collate([bundle], memory_state=enc1.memory_state)
    enc2 = model.encode(b2)
    assert torch.isfinite(enc2.context).all()
    assert enc2.context.shape == enc1.context.shape
