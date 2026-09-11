"""Channel encoders (pluggable). Importing this package registers the defaults.

Heavy variants (Transformer/CNN are fine on torch alone; GNN needs
torch-geometric) must keep their heavy imports inside ``__init__`` so importing
this package never fails when an optional dep is missing — the core PPO+MLP path
depends only on :mod:`channel_mlp`.
"""

from __future__ import annotations

from .channel_mlp import ChannelMLPEncoder

__all__ = ["ChannelMLPEncoder"]
