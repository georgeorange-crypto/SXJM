"""Train the M10 residual scorer with REINFORCE (DESIGN.md §12).

Runs ONLY as an explicit, separate step after Way4-Math is verified stable — which
it is (P3/P4 100% full-clear, full test suite green). This script does not touch the
shipped default (``Pipeline()`` still builds a plain math planner); it trains a
scorer, validates it GREEDILY against the math baseline (禁止10 gate), and writes the
checkpoint to ``--out`` only if it does not lower full-clear.

Usage:
  python scripts/train_rl_residual.py --problem 4 --iters 20 --episodes 4 \
      --train-seeds 2000-2015 --val-seeds 2000-2011 --out runs/residual_p4.pt

The training and validation seed sets are printed and logged; validation uses the
deployment ``RLResidualPlanner`` (greedy argmin), so the reported number is exactly
what shipping the checkpoint would do.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# -- path bootstrap (mirrors compare_way3_way4.py) --
_HERE = Path(__file__).resolve()
_WAY4_SRC = _HERE.parents[1] / "src"
_SX_ROOT = _HERE.parents[2]
_WAY3 = _SX_ROOT / "Way3"
for _p in (_WAY4_SRC, _SX_ROOT, _WAY3):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from way4.rl import ResidualScorer, ResidualTrainer, TrainConfig      # noqa: E402
from way4.rl.rollout import make_way4_rollout, validate               # noqa: E402


def _parse_seeds(spec: str):
    """Accept '2000-2015' or '2000,2001,2002'."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(x) for x in spec.split(","))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the M10 residual scorer (REINFORCE)")
    ap.add_argument("--problem", type=int, default=4, choices=(3, 4))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=4, help="episodes per iteration")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=120.0)
    ap.add_argument("--residual-clip", type=float, default=120.0)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--train-seeds", default="2000-2015")
    ap.add_argument("--val-seeds", default="2000-2011")
    ap.add_argument("--field", default="smooth", choices=("smooth", "constant"))
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/residual.pt")
    args = ap.parse_args(argv)

    hidden = tuple(int(x) for x in args.hidden.split(","))
    train_seeds = _parse_seeds(args.train_seeds)
    val_seeds = _parse_seeds(args.val_seeds)

    print(f"== M10 residual training (REINFORCE) ==", flush=True)
    print(f"  problem={args.problem} field={args.field} iters={args.iters} "
          f"episodes/iter={args.episodes} lr={args.lr} T={args.temperature} "
          f"clip={args.residual_clip} hidden={hidden}", flush=True)
    print(f"  train_seeds={train_seeds[0]}..{train_seeds[-1]} ({len(train_seeds)})  "
          f"val_seeds={val_seeds[0]}..{val_seeds[-1]} ({len(val_seeds)})", flush=True)

    scorer = ResidualScorer(hidden=hidden, zero_init=True, residual_clip=args.residual_clip)
    if not scorer.available:
        print("  torch unavailable -> cannot train.", flush=True)
        return 2

    # -- baseline validation BEFORE training (sanity: zero-init == math) --
    print("\n-- pre-training validation (zero-init scorer, should equal math) --", flush=True)
    t0 = time.perf_counter()
    pre = validate(scorer, val_seeds, problem=args.problem,
                   field_kind=args.field, max_steps=args.max_steps)
    print(f"  math full-clear {pre.math_full_clear}/{pre.n}   "
          f"rl(zero) full-clear {pre.rl_full_clear}/{pre.n}   "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    rollout = make_way4_rollout(problem=args.problem, field_kind=args.field,
                                max_steps=args.max_steps)
    cfg = TrainConfig(
        iterations=args.iters, episodes_per_iter=args.episodes, lr=args.lr,
        temperature=args.temperature, seed=args.seed, train_seeds=train_seeds,
        problem=args.problem, field_kind=args.field, max_steps=args.max_steps,
    )
    trainer = ResidualTrainer(scorer, rollout, cfg)

    def _log(s):
        print(f"  iter {s['iter']:3d}  loss={s['loss']:+9.3f}  "
              f"mean_return={s['mean_return']:+8.3f}  "
              f"train_full_clear={s['full_clear_rate']*100:5.1f}%  "
              f"n_dec={s['n_decisions']}", flush=True)

    print("\n-- training --", flush=True)
    t0 = time.perf_counter()
    trainer.train(log=_log)
    print(f"  trained in {time.perf_counter()-t0:.0f}s", flush=True)

    # -- greedy validation AFTER training (the 禁止10 acceptance gate) --
    print("\n-- post-training validation (greedy RLResidualPlanner vs math) --", flush=True)
    t0 = time.perf_counter()
    post = validate(scorer, val_seeds, problem=args.problem,
                    field_kind=args.field, max_steps=args.max_steps)
    speed = post.both_cleared_speedup
    print(f"  math full-clear {post.math_full_clear}/{post.n}   "
          f"rl full-clear {post.rl_full_clear}/{post.n}", flush=True)
    if speed is not None:
        print(f"  both-cleared mean time: math {post.math_mean_time:.0f}s  "
              f"rl {post.rl_mean_time:.0f}s  ratio rl/math={speed:.3f}", flush=True)
    print(f"  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # -- accept only if it does not lower full-clear (禁止10) --
    print("\n-- acceptance gate (禁止10: rl full-clear >= math full-clear) --", flush=True)
    if post.passes:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        scorer.save(str(out))
        verdict = "ACCEPTED"
        if speed is not None and speed < 1.0:
            verdict += f" (and {(1-speed)*100:.1f}% faster on both-cleared)"
        print(f"  PASS -> checkpoint saved to {out}  [{verdict}]", flush=True)
        print(f"  NOTE: shipped default stays math (no pipeline wiring); activation "
              f"is a separate explicit step: Way4Pipeline(planner=RLResidualPlanner("
              f"scorer=ResidualScorer.load('{out}'))).", flush=True)
        return 0
    print(f"  FAIL -> rl {post.rl_full_clear} < math {post.math_full_clear}; "
          f"checkpoint DISCARDED (would lower full-clear).", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
