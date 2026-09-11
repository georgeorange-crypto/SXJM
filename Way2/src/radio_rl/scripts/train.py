"""Train a PPO agent behind the frozen pipeline.

Usage::

    python -m radio_rl.scripts.train                 # algorithm=ppo by default
    python -m radio_rl.scripts.train problem=4 algorithm.model.hidden_dim=256
    python -m radio_rl.scripts.train algorithm.train.iterations=500 device=cuda

All arguments are Hydra-style config overrides (group selections or dotted
values), resolved by :func:`radio_rl.core.config.compose`. Torch is imported only
after config is composed, so ``--help``-style mistakes fail fast and cheap.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from ..core.config import compose


def main(argv: Optional[Sequence[str]] = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    # Default to the learnable path unless the user selected another algorithm.
    if not any(t.split("=", 1)[0].strip() == "algorithm" for t in tokens):
        tokens = ["algorithm=ppo", *tokens]

    cfg = compose(overrides=tokens)

    from ..training import train  # lazy: pulls in torch

    result = train(cfg, progress=True)
    if result.history:
        last = result.history[-1]
        print(
            "training done — "
            f"final clear_ratio={last.get('mean_clear_ratio', 0):.3f} "
            f"virtual_time={last.get('mean_virtual_time', 0):.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
