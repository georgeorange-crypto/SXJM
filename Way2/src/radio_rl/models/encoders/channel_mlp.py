"""Default channel encoder: a shared per-channel MLP (``channel_mlp``).

Treats every channel's feature row independently with the same MLP — the
simplest encoder and the M2 default. It is the baseline a Channel Transformer,
CNN, or GNN (later milestones) must beat; those attend *across* channels but
expose the same ``[B, C, Dch] -> [B, C, out_dim]`` signature, so swapping them in
is a one-line config change and never touches the agent model.
"""

from __future__ import annotations

import torch

from ...core.registry import CHANNEL_ENCODERS
from ..base import ChannelEncoder
from ..mlp import mlp


@CHANNEL_ENCODERS.register("channel_mlp")
class ChannelMLPEncoder(ChannelEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int | None = None,
        layers: int = 2,
        layernorm: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        out_dim = int(out_dim or hidden_dim)
        sizes = [in_dim] + [hidden_dim] * max(0, layers - 1) + [out_dim]
        self.net = mlp(sizes, layernorm=layernorm)
        self.out_dim = out_dim

    def forward(self, channel_feats: torch.Tensor) -> torch.Tensor:
        # channel_feats: [B, C, in_dim] -> [B, C, out_dim] (MLP over last dim)
        return self.net(channel_feats)
