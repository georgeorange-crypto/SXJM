"""Channel GNN encoder (``channel_gnn``): message passing over the channel graph.

A dense GraphSAGE/GCN-style encoder on the fully-connected graph of the ``C``
channels: each layer updates a channel from its own features plus the mean of all
channels' features (the neighbourhood aggregate), with a residual connection and
LayerNorm. No natural sparse graph exists over the bands, so the graph is
complete; the message/self transforms are what let channels share evidence.

Implemented in pure torch (no ``torch-geometric`` needed), so importing it can
never fail on a missing optional dependency. Keeps the :class:`ChannelEncoder`
contract ``[B, C, in_dim] -> [B, C, out_dim]``, a one-line swap for ``channel_mlp``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import CHANNEL_ENCODERS
from ..base import ChannelEncoder


@CHANNEL_ENCODERS.register("channel_gnn")
class ChannelGNNEncoder(ChannelEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int | None = None,
        num_layers: int = 2,
        layernorm: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        out_dim = int(out_dim or hidden_dim)
        n = max(1, int(num_layers))

        self.input = nn.Linear(int(in_dim), h)
        self.self_lin = nn.ModuleList(nn.Linear(h, h) for _ in range(n))
        self.neigh_lin = nn.ModuleList(nn.Linear(h, h) for _ in range(n))
        self.norms = nn.ModuleList(
            (nn.LayerNorm(h) if layernorm else nn.Identity()) for _ in range(n)
        )
        self.output = nn.Linear(h, out_dim)
        self.out_dim = out_dim

    def forward(self, channel_feats: torch.Tensor) -> torch.Tensor:
        # channel_feats: [B, C, in_dim]
        h = self.input(channel_feats)                        # [B, C, H]
        for self_lin, neigh_lin, norm in zip(self.self_lin, self.neigh_lin, self.norms):
            msg = h.mean(dim=1, keepdim=True)                # [B, 1, H] aggregate
            upd = self_lin(h) + neigh_lin(msg)               # broadcast over channels
            h = norm(h + F.gelu(upd))                        # residual update
        return self.output(h)                                # [B, C, out_dim]
