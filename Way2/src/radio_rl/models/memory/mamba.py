"""Mamba-style selective state-space memory (``mamba``).

A compact, pure-torch selective SSM in the Mamba / S6 style: the recurrence

    h_t = a_t ⊙ h_{t-1} + Δ_t ⊙ u_t ,   y_t = c_t ⊙ h_t + D ⊙ u_t

runs one step at a time with an explicit diagonal state ``h`` ([B, out_dim]). What
makes it *selective* (the Mamba idea) is that the input gate ``Δ_t``, the decay
``a_t = exp(-Δ_t · softplus(A))`` and the output projection ``c_t`` are all
functions of the current input, so the model can choose what to remember and what
to forget per step. A SiLU gate on the output mirrors the Mamba block.

Implemented without the CUDA ``mamba-ssm`` kernel so it always constructs and runs
on CPU (swap in that kernel for long-sequence scale). Same explicit-state contract
as the other memories: ``forward([B, dim], state) -> ([B, out_dim], h)``, with
``state=None`` starting from zeros.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MEMORIES
from ..base import TemporalMemory


@MEMORIES.register("mamba")
class MambaMemory(TemporalMemory):
    def __init__(self, dim: int, hidden: int | None = None, **_: object) -> None:
        super().__init__()
        self.in_dim = int(dim)
        self.out_dim = int(hidden or dim)
        h = self.out_dim

        self.in_proj = nn.Linear(self.in_dim, h)      # u_t
        self.dt_proj = nn.Linear(self.in_dim, h)      # Δ_t (selective step)
        self.c_proj = nn.Linear(self.in_dim, h)       # c_t (selective output)
        self.gate_proj = nn.Linear(self.in_dim, h)    # SiLU gate
        self.A_log = nn.Parameter(torch.zeros(h))     # softplus(A_log) > 0
        self.D = nn.Parameter(torch.ones(h))
        self.out_proj = nn.Linear(h, self.out_dim)

    def initial_state(self, batch_size: int, device: torch.device | str) -> Any:
        return torch.zeros(batch_size, self.out_dim, device=device)

    def forward(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        u = self.in_proj(x)                           # [B, H]
        dt = F.softplus(self.dt_proj(x))              # [B, H] > 0
        a = torch.exp(-dt * F.softplus(self.A_log))   # [B, H] in (0, 1)
        h = a * state + dt * u                        # [B, H] new SSM state
        y = self.c_proj(x) * h + self.D * u           # selective read-out
        y = y * F.silu(self.gate_proj(x))             # Mamba-style gate
        return self.out_proj(y), h
