"""AgentModel: assembles encoder + memory + fusion + pointer actor + critic.

This is the whole learnable network behind the frozen pipeline. Data flow::

    global_feats  --global MLP-------------------------> g        [B,Hg]
    channel_feats --channel encoder--> channel_embed --mean--> ch  [B,Hch]
                     (pluggable)                    (per channel)
    concat(g, ch) --memory (pluggable)--> mem_out                  [B,Hm]
    fusion([g, ch, mem_out]) (pluggable) --> context              [B,Hc]

    cand_feats  ++ gather(channel_embed, cand_channel_idx)
                --candidate MLP--> cand_embed                     [B,N,He]

    pointer actor(context, cand_embed, mask) --> logits           [B,N]
    critic(context)                          --> value            [B]

The three ``(pluggable)`` boxes are swapped by config through their registries;
every other box is fixed. Nothing here imports geometry — only tensors and the
:class:`FeatureSpec` (architecture principle B).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..core.registry import CHANNEL_ENCODERS, FUSIONS, MEMORIES
from ..features.spec import BatchedFeatures, FeatureSpec
from .actor import PointerActor
from .base import ChannelEncoder, EncoderOutput, Fusion, TemporalMemory
from .critic import Critic
from .mlp import mlp

# importing these packages registers the default plugins
from . import encoders as _encoders  # noqa: F401
from . import fusion as _fusion      # noqa: F401
from . import memory as _memory      # noqa: F401


class AgentModel(nn.Module):
    """Actor-critic over the FeatureBundle. See module docstring for the wiring."""

    def __init__(
        self,
        spec: FeatureSpec,
        hidden_dim: int,
        channel_encoder: ChannelEncoder,
        memory: TemporalMemory,
        fusion: Fusion,
    ) -> None:
        super().__init__()
        self.spec = spec
        h = int(hidden_dim)
        self.channel_encoder = channel_encoder
        self.memory = memory
        self.fusion = fusion
        he = channel_encoder.out_dim
        self._he = he

        self.global_encoder = mlp([spec.global_dim, h, h], layernorm=True)
        self.cand_encoder = mlp([spec.candidate_dim + he, h, h], layernorm=True)
        self.context_dim = fusion.out_dim
        self.actor = PointerActor(self.context_dim, h, h)
        self.critic = Critic(self.context_dim, h)

    # -- encoding ----------------------------------------------------------
    def encode(self, batch: BatchedFeatures) -> EncoderOutput:
        g = self.global_encoder(batch.global_feats)             # [B, H]
        ch = self.channel_encoder(batch.channel_feats)          # [B, C, He]
        ch_summary = ch.mean(dim=1)                             # [B, He]
        pre_ctx = torch.cat([g, ch_summary], dim=-1)            # [B, H+He]
        mem_out, state = self.memory(pre_ctx, batch.memory_state)
        context = self.fusion([g, ch_summary, mem_out])         # [B, Hc]

        idx = batch.cand_channel_idx.unsqueeze(-1).expand(-1, -1, self._he)
        ch_rows = torch.gather(ch, 1, idx)                      # [B, N, He]
        cand_in = torch.cat([batch.cand_feats, ch_rows], dim=-1)
        cand_embed = self.cand_encoder(cand_in)                 # [B, N, H]

        return EncoderOutput(
            context=context,
            cand_embed=cand_embed,
            cand_mask=batch.cand_mask,
            channel_embed=ch,
            global_embed=g,
            memory_state=state,
        )

    # -- heads -------------------------------------------------------------
    def policy_value(
        self, batch: BatchedFeatures
    ) -> tuple[torch.Tensor, torch.Tensor, EncoderOutput]:
        enc = self.encode(batch)
        logits = self.actor(enc.context, enc.cand_embed, enc.cand_mask)  # [B, N]
        value = self.critic(enc.context)                                 # [B]
        return logits, value, enc

    def evaluate_actions(
        self, batch: BatchedFeatures, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO's update: log-prob of ``actions``, entropy, and value."""
        logits, value, _ = self.policy_value(batch)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def _plugin_cfg(node: Any, default_type: str) -> tuple[str, dict]:
    """Read a ``{type: name, ...}`` model sub-node into (name, kwargs)."""
    if node is None:
        return default_type, {}
    try:
        from omegaconf import OmegaConf

        d = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)
    except Exception:
        try:
            d = dict(node)
        except Exception:
            d = {}
    name = str(d.pop("type", default_type))
    return name, {k: v for k, v in d.items() if v is not None}


def build_agent_model(model_cfg: Any, spec: FeatureSpec) -> AgentModel:
    """Build an :class:`AgentModel` from a ``model`` config node and a spec.

    ``model_cfg`` shape::

        hidden_dim: 128
        encoder: {type: channel_mlp, layers: 2}
        memory:  {type: none}
        fusion:  {type: concat}

    Missing sub-nodes fall back to the M2 defaults, so an empty node yields
    PPO+MLP. Unknown plugin names raise a clear KeyError from the registry.
    """
    def node(key: str) -> Any:
        if model_cfg is None:
            return None
        try:
            return model_cfg.get(key) if hasattr(model_cfg, "get") else model_cfg[key]
        except Exception:
            return None

    hidden = int(node("hidden_dim") or 128)

    enc_name, enc_kwargs = _plugin_cfg(node("encoder"), "channel_mlp")
    channel_encoder = CHANNEL_ENCODERS.create(
        enc_name, in_dim=spec.channel_dim, hidden_dim=hidden, **enc_kwargs
    )

    mem_name, mem_kwargs = _plugin_cfg(node("memory"), "none")
    mem_in = hidden + channel_encoder.out_dim
    memory = MEMORIES.create(mem_name, dim=mem_in, **mem_kwargs)

    fus_name, fus_kwargs = _plugin_cfg(node("fusion"), "concat")
    fusion = FUSIONS.create(
        fus_name,
        in_dims=[hidden, channel_encoder.out_dim, memory.out_dim],
        out_dim=hidden,
        **fus_kwargs,
    )

    return AgentModel(spec, hidden, channel_encoder, memory, fusion)
