"""Default run path — evaluate the configured pipeline.

The whole point of the project's modularity: this one command runs the default
greedy baseline today, and the trained PPO agent tomorrow, by changing a single
config token. Examples::

    python scripts/evaluate.py                       # defaults (greedy, P3)
    python scripts/evaluate.py problem=4 seed=10
    python scripts/evaluate.py evaluation.n_episodes=50
    python scripts/evaluate.py --stress              # adversarial families
    python scripts/evaluate.py algorithm=ppo         # (once trained; later milestone)
"""

from __future__ import annotations

import sys

from radio_rl.core.config import compose
from radio_rl.evaluation import evaluate, run_stress


def main(argv: list[str]) -> None:
    stress = False
    overrides = []
    for a in argv:
        if a in ("--stress", "-s"):
            stress = True
        else:
            overrides.append(a)

    cfg = compose(overrides=overrides)
    print(f"# run: {cfg.get('run_name', 'default')}  problem={cfg.get('problem')}  "
          f"algorithm={cfg.algorithm.get('name', cfg.algorithm.get('type'))}  "
          f"env={cfg.env.type}\n")

    if stress or bool(cfg.evaluation.get("stress", False)):
        run_stress(cfg)
    else:
        evaluate(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
