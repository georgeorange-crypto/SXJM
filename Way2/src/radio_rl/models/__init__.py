"""The models layer: the learnable network behind the frozen pipeline.

Torch-only, imports no geometry (architecture principle B). Importing this
package registers the default plugins (channel_mlp / none / concat) and exposes
the :class:`AgentModel` assembler. Only imported on the learnable path.
"""

from __future__ import annotations

from .actor import PointerActor
from .agent_model import AgentModel, build_agent_model
from .base import ChannelEncoder, EncoderOutput, Fusion, TemporalMemory
from .critic import Critic

__all__ = [
    "AgentModel",
    "build_agent_model",
    "EncoderOutput",
    "ChannelEncoder",
    "TemporalMemory",
    "Fusion",
    "PointerActor",
    "Critic",
]
