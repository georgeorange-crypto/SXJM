"""Channel CNN encoder (``channel_cnn``): 1-D convolution across the channel axis.

Treats the channels as an ordered sequence (channel number tracks frequency) and
convolves along it, so each channel's embedding mixes in its spectral neighbours.
``same`` padding keeps the channel count fixed, so the output is still one row per
channel.

Keeps the :class:`ChannelEncoder` contract ``[B, C, in_dim] -> [B, C, out_dim]``,
so it is a one-line swap for ``channel_mlp`` and never touches the agent model.
Torch-only; imported on the learnable path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...core.registry import CHANNEL_ENCODERS
from ..base import ChannelEncoder


@CHANNEL_ENCODERS.register("channel_cnn")
class ChannelCNNEncoder(ChannelEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int | None = None,
        num_layers: int = 2,
        kernel_size: int = 3,
        **_: object,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        out_dim = int(out_dim or hidden_dim)
        k = int(kernel_size)
        pad = k // 2                                   # 'same' length for odd k

        sizes = [int(in_dim)] + [h] * max(0, int(num_layers) - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Conv1d(sizes[i], sizes[i + 1], k, padding=pad))
            if i < len(sizes) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, channel_feats: torch.Tensor) -> torch.Tensor:
        # Conv1d wants [B, C_features, L]; our sequence length is the channel axis.
        z = channel_feats.transpose(1, 2)              # [B, in_dim, C]
        z = self.net(z)                                # [B, out_dim, C]
        return z.transpose(1, 2)                       # [B, C, out_dim]
