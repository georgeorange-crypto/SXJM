"""Small shared building block: a configurable MLP.

Kept in one place so every encoder/actor/critic builds layers the same way
(optional LayerNorm, choice of activation). Torch-only; part of the models layer.
"""

from __future__ import annotations

from typing import Optional, Sequence, Type

import torch.nn as nn


def mlp(
    sizes: Sequence[int],
    activation: Type[nn.Module] = nn.ReLU,
    out_activation: Optional[Type[nn.Module]] = None,
    layernorm: bool = False,
) -> nn.Sequential:
    """Feed-forward stack over the last dimension.

    ``sizes`` is [in, h1, ..., out]. LayerNorm (when enabled) and the activation
    are inserted after every *hidden* Linear; the output Linear gets
    ``out_activation`` (usually None). Works on any [..., in] tensor since it is
    just stacked ``nn.Linear``.
    """
    if len(sizes) < 2:
        raise ValueError("mlp needs at least [in, out]")
    layers: list[nn.Module] = []
    n = len(sizes)
    for i in range(n - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == n - 2
        if not is_last:
            if layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
        elif out_activation is not None:
            layers.append(out_activation())
    return nn.Sequential(*layers)
