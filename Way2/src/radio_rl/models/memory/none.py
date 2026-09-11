"""Memoryless default (``none``): the identity, no recurrence.

This is the M2 default and keeps PPO a standard feed-forward actor-critic. An
LSTM/GRU/Mamba memory (later milestones) registers alongside and threads state
across steps; the agent model calls all of them through the same
``forward(x, state) -> (out, state)`` signature.
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.registry import MEMORIES
from ..base import TemporalMemory


@MEMORIES.register("none")
class NoMemory(TemporalMemory):
    def __init__(self, dim: int, **_: object) -> None:
        super().__init__()
        self.out_dim = int(dim)

    def forward(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        return x, None
