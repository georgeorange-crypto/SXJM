"""GRU memory (``gru``): a lighter recurrence over the per-step context.

Same role and contract as :class:`~radio_rl.models.memory.lstm.LSTMMemory` with a
single hidden vector as state (no cell state), so it is cheaper and often as
effective. State is explicit; ``state=None`` starts from zeros.

``forward([B, dim], state) -> ([B, out_dim], h)``. Torch-only.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ...core.registry import MEMORIES
from ..base import TemporalMemory


@MEMORIES.register("gru")
class GRUMemory(TemporalMemory):
    def __init__(self, dim: int, hidden: int | None = None, **_: object) -> None:
        super().__init__()
        self.in_dim = int(dim)
        self.out_dim = int(hidden or dim)
        self.cell = nn.GRUCell(self.in_dim, self.out_dim)

    def initial_state(self, batch_size: int, device: torch.device | str) -> Any:
        return torch.zeros(batch_size, self.out_dim, device=device)

    def forward(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        h = self.cell(x, state)
        return h, h
