"""Default fusion (``concat``): concatenate the parts and project.

Combines the global embedding, the channel summary, and the memory output into
the single context vector the actor and critic read. A gated or attention-based
fusion registers alongside with the same ``list[[B, di]] -> [B, out_dim]``
signature.
"""

from __future__ import annotations

import torch

from ...core.registry import FUSIONS
from ..base import Fusion
from ..mlp import mlp


@FUSIONS.register("concat")
class ConcatFusion(Fusion):
    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        layernorm: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.proj = mlp([int(sum(in_dims)), out_dim, out_dim], layernorm=layernorm)
        self.out_dim = int(out_dim)

    def forward(self, parts: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(parts, dim=-1))
