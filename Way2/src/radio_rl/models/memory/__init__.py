"""Temporal memories (pluggable). Importing this package registers all variants.

The core default is memoryless (:mod:`none`), keeping PPO a standard feed-forward
actor-critic. The recurrent variants (LSTM/GRU and a pure-torch Mamba-style
selective SSM) thread an explicit state across the steps of an episode and share
the ``forward(x, state) -> (out, state)`` contract, so any of them is a one-line
config swap (``algorithm.model.memory.type=lstm``) with no change to the model.
"""

from __future__ import annotations

from .gru import GRUMemory
from .lstm import LSTMMemory
from .mamba import MambaMemory
from .none import NoMemory

__all__ = ["NoMemory", "LSTMMemory", "GRUMemory", "MambaMemory"]
