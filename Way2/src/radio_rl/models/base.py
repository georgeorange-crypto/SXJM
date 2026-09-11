"""Model-layer contracts: the encoder output and the three pluggable interfaces.

Everything here is torch and imports **no geometry** (architecture principle B):
models see the tensor :class:`~radio_rl.features.spec.FeatureBundle` /
:class:`BatchedFeatures`, never the belief. The three abstract classes are the
swap points advertised by the design — a channel encoder, a temporal memory, and
a fusion — each built by name from its registry so PPO+MLP becomes
PPO+Transformer+LSTM by editing config, with no code change here.

:class:`EncoderOutput` is the single structured latent the actor and critic
consume (and that a later world-model / Dreamer can read), so adding a new head
never changes the encoder's call signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    """Structured latent produced by :meth:`AgentModel.encode`.

    ``context`` summarises the whole belief (state value + policy conditioning);
    ``cand_embed`` is one vector per candidate (what the pointer actor scores).
    ``memory_state`` is the opaque recurrent state to thread into the next step
    (``None`` for memoryless models).
    """

    context: torch.Tensor          # [B, Hc]
    cand_embed: torch.Tensor       # [B, N, He]
    cand_mask: torch.Tensor        # [B, N] bool
    channel_embed: torch.Tensor    # [B, C, Hch]
    global_embed: torch.Tensor     # [B, Hg]
    memory_state: Any = None


class ChannelEncoder(nn.Module, ABC):
    """Encodes the per-channel feature rows into per-channel embeddings.

    ``forward([B, C, Dch]) -> [B, C, out_dim]``. The default (``channel_mlp``)
    treats channels independently; a Transformer/CNN/GNN variant attends across
    them but keeps this exact signature.
    """

    out_dim: int

    @abstractmethod
    def forward(self, channel_feats: torch.Tensor) -> torch.Tensor:  # noqa: D401
        ...


class TemporalMemory(nn.Module, ABC):
    """Optional recurrence over the per-step context.

    ``forward([B, in_dim], state) -> ([B, out_dim], new_state)``. The default
    (``none``) is the identity with ``state == None``; an LSTM/GRU/Mamba variant
    carries state across steps. Keeping state explicit (not stored on the module)
    lets a single module serve many concurrent episodes.
    """

    out_dim: int

    def initial_state(self, batch_size: int, device: torch.device | str) -> Any:
        return None

    @abstractmethod
    def forward(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        ...


class Fusion(nn.Module, ABC):
    """Combines several context parts into the single context vector.

    ``forward([t0, t1, ...]) -> [B, out_dim]`` where each ``ti`` is ``[B, di]``.
    The default (``concat``) concatenates and projects; a gated/attention variant
    keeps the signature.
    """

    out_dim: int

    @abstractmethod
    def forward(self, parts: list[torch.Tensor]) -> torch.Tensor:
        ...
