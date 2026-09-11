"""LSTM memory (``lstm``): recurrence over the per-step context.

Threads an (h, c) state across the steps of an episode, so the policy can
condition on the whole trajectory of beliefs, not just the current one. State is
kept *explicit* (passed in and out, never stored on the module), so one module
serves many concurrent episodes and the trainer stays plugin-agnostic.

``forward([B, dim], state) -> ([B, out_dim], (h, c))``. ``state=None`` starts from
zeros — that is also what the PPO update passes (each transition re-encoded from a
fresh state), so a recurrent policy trains under the standard first-state
approximation while acting with the full threaded state at rollout. Torch-only.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ...core.registry import MEMORIES
from ..base import TemporalMemory


@MEMORIES.register("lstm")
class LSTMMemory(TemporalMemory):
    def __init__(self, dim: int, hidden: int | None = None, **_: object) -> None:
        super().__init__()
        self.in_dim = int(dim)
        self.out_dim = int(hidden or dim)
        self.cell = nn.LSTMCell(self.in_dim, self.out_dim)

    def initial_state(self, batch_size: int, device: torch.device | str) -> Any:
        z = torch.zeros(batch_size, self.out_dim, device=device)
        return (z, z.clone())

    def forward(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        h, c = self.cell(x, state)
        return h, (h, c)
