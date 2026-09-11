"""Pointer actor: scores a variable-length, masked candidate set.

The policy never emits coordinates — it points at one of the pre-generated
candidates (the iron rule, policy side). Scoring is additive attention between a
query projected from the context and each candidate embedding, so it is
permutation-equivariant over candidates and works for any candidate count. Padded
slots are masked to ``-inf`` before the softmax, so they get exactly zero
probability regardless of their feature values.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointerActor(nn.Module):
    def __init__(self, context_dim: int, cand_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(context_dim, hidden_dim)
        self.k = nn.Linear(cand_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: torch.Tensor,      # [B, context_dim]
        cand_embed: torch.Tensor,   # [B, N, cand_dim]
        cand_mask: torch.Tensor,    # [B, N] bool (True = real)
    ) -> torch.Tensor:
        q = self.q(context).unsqueeze(1)               # [B, 1, H]
        k = self.k(cand_embed)                         # [B, N, H]
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # [B, N]
        # mask padding to -inf; keep at least the real candidates finite.
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~cand_mask, neg_inf)
        return scores
