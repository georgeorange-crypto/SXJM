"""Critic: state-value head over the fused context.

Reads only the context (a summary of the belief), not the candidate set, so it
estimates V(state) rather than a per-action value — exactly what PPO's advantage
needs. The candidate-independent value keeps the baseline stable as the candidate
set changes step to step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import mlp


class Critic(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = mlp([context_dim, hidden_dim, 1])

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context).squeeze(-1)  # [B]
