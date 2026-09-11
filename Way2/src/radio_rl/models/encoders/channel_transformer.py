"""Channel Transformer encoder (``channel_transformer``): attends across channels.

Multi-head self-attention over the ``C`` channel rows, so every channel's
embedding is informed by every other channel (shared spectrum structure, nearby
estimates, which bands are already cleared). A learnable per-channel positional
embedding keeps the (frequency-ordered) channel identity.

It keeps the :class:`ChannelEncoder` contract ``[B, C, in_dim] -> [B, C, out_dim]``
exactly, so it swaps in for ``channel_mlp`` by one config line
(``algorithm.model.encoder.type=channel_transformer``) and never touches the
agent model. Torch-only; imported on the learnable path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...core.registry import CHANNEL_ENCODERS
from ..base import ChannelEncoder


def _pick_heads(dim: int, want: int) -> int:
    """Largest head count <= ``want`` that divides ``dim`` (>=1)."""
    want = max(1, int(want))
    for n in range(min(want, dim), 0, -1):
        if dim % n == 0:
            return n
    return 1


@CHANNEL_ENCODERS.register("channel_transformer")
class ChannelTransformerEncoder(ChannelEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_channels: int = 20,
        **_: object,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        out_dim = int(out_dim or hidden_dim)
        heads = _pick_heads(h, num_heads)

        self.input = nn.Linear(int(in_dim), h)
        self.pos = nn.Parameter(torch.zeros(1, int(num_channels), h))
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=heads,
            dim_feedforward=2 * h,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True makes the nested-tensor fast path inapplicable; disable
        # it explicitly (it is a no-op for us) to keep the output warning-free.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(num_layers), enable_nested_tensor=False
        )
        self.output = nn.Linear(h, out_dim)
        self.out_dim = out_dim

    def forward(self, channel_feats: torch.Tensor) -> torch.Tensor:
        # channel_feats: [B, C, in_dim]
        x = self.input(channel_feats)                 # [B, C, H]
        x = x + self.pos[:, : x.shape[1], :]          # channel-position bias
        x = self.encoder(x)                           # [B, C, H] (attends across C)
        return self.output(x)                         # [B, C, out_dim]
