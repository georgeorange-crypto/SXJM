"""Channel encoders (pluggable). Importing this package registers all variants.

The core PPO+MLP path depends only on :mod:`channel_mlp`; the advanced encoders
(Transformer/CNN/GNN) attend or convolve *across* channels but expose the same
``[B, C, Dch] -> [B, C, out_dim]`` signature, so they are one-line config swaps.
All are pure torch (the GNN uses no ``torch-geometric``), so importing this
package never fails on a missing optional dependency.
"""

from __future__ import annotations

from .channel_cnn import ChannelCNNEncoder
from .channel_gnn import ChannelGNNEncoder
from .channel_mlp import ChannelMLPEncoder
from .channel_transformer import ChannelTransformerEncoder

__all__ = [
    "ChannelMLPEncoder",
    "ChannelTransformerEncoder",
    "ChannelCNNEncoder",
    "ChannelGNNEncoder",
]
