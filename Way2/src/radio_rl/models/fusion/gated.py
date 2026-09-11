"""Gated fusion (``gated``): a learned per-part gate instead of a fixed concat.

Projects each context part (global embedding, channel summary, memory output) to
the output width, then scales each by a sigmoid gate computed from all parts
together, and sums. This lets the network *modulate* how much each source
contributes per step (e.g. lean on the memory once a trajectory is informative),
where :class:`ConcatFusion` weights them only through a fixed linear projection.

Same contract as the default: ``list[[B, di]] -> [B, out_dim]``, a one-line swap
(``algorithm.model.fusion.type=gated``). Torch-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...core.registry import FUSIONS
from ..base import Fusion


@FUSIONS.register("gated")
class GatedFusion(Fusion):
    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        layernorm: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        o = int(out_dim)
        self._parts = len(in_dims)
        self._o = o
        self.projs = nn.ModuleList(nn.Linear(int(d), o) for d in in_dims)
        self.gate = nn.Linear(int(sum(in_dims)), self._parts * o)
        self.norm = nn.LayerNorm(o) if layernorm else nn.Identity()
        self.out_dim = o

    def forward(self, parts: list[torch.Tensor]) -> torch.Tensor:
        cat = torch.cat(parts, dim=-1)                              # [B, sum d]
        gates = torch.sigmoid(self.gate(cat))                      # [B, P*O]
        gates = gates.view(gates.shape[0], self._parts, self._o)   # [B, P, O]
        proj = torch.stack([p(x) for p, x in zip(self.projs, parts)], dim=1)  # [B, P, O]
        fused = (gates * proj).sum(dim=1)                          # [B, O]
        return self.norm(fused)
