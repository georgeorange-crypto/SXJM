"""Record one or more algorithms on a case and emit a standalone HTML viewer.

The viewer shows, on the arena canvas:
  * grey dots        -- interference sources not yet found (ground truth);
  * coloured fans    -- the robot's detection traces (bearing +/-1 deg wedge);
  * an orange route  -- the robot's path;
  * X / green dots   -- sources it localised and cleared.

Examples
--------
    # default greedy baseline, one seed, problem 3
    python scripts/visualize.py

    # compare several algorithms on the SAME map (same seed => same layout)
    python scripts/visualize.py --algos heuristic,ppo --seed 7 --problem 4

    # a PPO checkpoint you trained
    python scripts/visualize.py --algos ppo \
        --override algorithm.checkpoint=runs/ppo/model.pt

    # choose the output file
    python scripts/visualize.py --out runs/view/compare.html

Open the resulting HTML by double-clicking it -- it embeds its data and needs
no server. Use the dropdown to switch algorithms and the slider / play button to
watch the search unfold step by step.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from radio_rl.core.config import compose
from radio_rl.viz import record_run, write_report


def _algo_overrides(algo: str) -> list[str]:
    """Map a short algorithm token to the config overrides that select it."""
    algo = algo.strip()
    if not algo:
        return []
    # `heuristic` and `ppo` are config-group options; anything else is assumed
    # to already be a valid `algorithm=<name>` option file.
    return [f"algorithm={algo}"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", default="heuristic",
                    help="comma-separated algorithm options to record "
                         "(e.g. 'heuristic,ppo'); each faces the same case")
    ap.add_argument("--seed", type=int, default=7,
                    help="case seed shared by every algorithm (same map)")
    ap.add_argument("--problem", type=int, default=3, choices=(3, 4),
                    help="3 = omni only; 4 = includes directional sources")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--override", action="append", default=[],
                    metavar="k=v", help="extra config override(s), repeatable")
    ap.add_argument("--out", default="runs/view/run.html",
                    help="output HTML path (relative to Way2/)")
    ap.add_argument("--no-open", action="store_true",
                    help="do not open the report in a browser afterwards")
    args = ap.parse_args(argv)

    algos = [a for a in args.algos.split(",") if a.strip()]
    if not algos:
        ap.error("no algorithms given")

    records = []
    for algo in algos:
        overrides = (
            [f"problem={args.problem}"]
            + _algo_overrides(algo)
            + list(args.override)
        )
        cfg = compose(overrides=overrides)
        print(f"# recording algo={algo!r} seed={args.seed} problem={args.problem} ...",
              flush=True)
        rec = record_run(cfg, seed=args.seed, label=algo, max_steps=args.max_steps)
        s = rec.summary
        print(f"    cleared {s['sources_cleared']}/{s['sources_total']}  "
              f"virtual_time={s['virtual_time_s']}s  steps={s['steps']}  "
              f"frames={len(rec.frames)}")
        records.append(rec)

    out = write_report(
        records, args.out,
        title=f"干扰源定位清除 · 运行可视化 (P{args.problem}, seed={args.seed})",
    )
    print(f"\n# wrote {out.resolve()}")
    if not args.no_open:
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
