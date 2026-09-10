"""Smoke test: build the default pipeline from config and run a few episodes.

Run from the repo root:  python scripts/smoke_test.py
This exercises the entire frozen path (env -> estimator -> candidates ->
greedy agent -> safety -> action) with no torch and no network.
"""

from __future__ import annotations

import argparse

from radio_rl.core.config import compose
from radio_rl.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--problem", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("overrides", nargs="*", help="extra cfg overrides, e.g. seed=1")
    args = ap.parse_args()

    cfg = compose(overrides=[f"problem={args.problem}", *args.overrides])
    pipe = Pipeline(cfg)

    print(f"{'seed':>4}  {'cleared':>9}  {'vtime_s':>10}  {'steps':>6}  "
          f"{'scans':>5}  {'clears':>6}  {'fail':>4}  {'wall_s':>7}")
    tot_cleared = tot_sources = 0
    for seed in range(args.episodes):
        stats, _ = pipe.run_episode(seed=seed, max_steps=args.max_steps)
        tot_cleared += stats.sources_cleared
        tot_sources += stats.sources_total
        print(f"{seed:>4}  {stats.sources_cleared:>4}/{stats.sources_total:<4}  "
              f"{stats.virtual_time:>10.1f}  {stats.steps:>6}  {stats.num_scans:>5}  "
              f"{stats.num_clears:>6}  {stats.failed_clears:>4}  {stats.wall_time:>7.2f}")
    ratio = (tot_cleared / tot_sources) if tot_sources else 0.0
    print(f"\ntotal cleared {tot_cleared}/{tot_sources}  ({ratio:.1%})")


if __name__ == "__main__":
    main()
