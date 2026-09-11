"""Attention fusion (``attention``): attend over the context parts.

Projects each context part to the output width to form a small set of tokens,
scores them against a learned query, and returns the softmax-weighted sum. Unlike
the fixed :class:`ConcatFusion`, the mixing weights are data-dependent and sum to
one across parts, so the context is a convex combination the network steers per
step.

Same contract as the default: ``list[[B, di]] -> [B, out_dim]``, a one-line swap
(``algorithm.model.fusion.type=attention``). The query is zero-initialised, so it
starts as a uniform average and specialises during training. Torch-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...core.registry import FUSIONS
from ..base import Fusion


@FUSIONS.register("attention")
class AttentionFusion(Fusion):
    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        layernorm: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        o = int(out_dim)
        self.projs = nn.ModuleList(nn.Linear(int(d), o) for d in in_dims)
        self.query = nn.Parameter(torch.zeros(o))
        self.scale = float(o) ** 0.5
        self.norm = nn.LayerNorm(o) if layernorm else nn.Identity()
        self.out_dim = o

    def forward(self, parts: list[torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([p(x) for p, x in zip(self.projs, parts)], dim=1)  # [B, P, O]
        scores = (tokens * self.query).sum(dim=-1) / self.scale                # [B, P]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)                   # [B, P, 1]
        fused = (weights * tokens).sum(dim=1)                                  # [B, O]
        return self.norm(fused)
