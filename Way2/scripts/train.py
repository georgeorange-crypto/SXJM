"""Train the PPO agent through the frozen pipeline.

    python scripts/train.py                          # baseline: algorithm=ppo, 200 iters
    python scripts/train.py algorithm=ppo_transformer
    python scripts/train.py algorithm.train.iterations=50 algorithm.train.eval_every=10

Every argument is a config override (see ``radio_rl.core.config.compose``): either a
group selection (``algorithm=ppo_transformer``) or a dotted value
(``algorithm.train.iterations=50``). If no ``algorithm=`` is given, the learnable
``ppo`` path is selected by default. The tensor stack is imported here only; the
mathematical core stays torch-free.
"""

from __future__ import annotations

import sys
import time

from radio_rl.core.config import compose
from radio_rl.training import train


def main(argv: list[str]) -> int:
    overrides = list(argv)
    if not any(o.split("=", 1)[0].strip() == "algorithm" for o in overrides):
        overrides = ["algorithm=ppo", *overrides]

    cfg = compose(overrides=overrides)
    print(f"overrides: {overrides}", flush=True)

    t0 = time.perf_counter()
    res = train(cfg, progress=True)
    dt = time.perf_counter() - t0

    hist = res.history
    n = len(hist)
    print("=" * 60, flush=True)
    print(f"done: {n} iterations in {dt:.1f}s ({dt / max(n, 1):.2f}s/iter)", flush=True)
    if hist:
        last = hist[-1]
        best = max(hist, key=lambda r: r.get("mean_clear_ratio", 0.0))
        print(
            f"last : return={last['mean_return']:+.3f} "
            f"clear={last['mean_clear_ratio']:.2f} "
            f"vt={last['mean_virtual_time']:.0f} len={last['mean_ep_len']:.0f}",
            flush=True,
        )
        print(
            f"best : clear={best['mean_clear_ratio']:.2f} @ iter {best['iter']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
