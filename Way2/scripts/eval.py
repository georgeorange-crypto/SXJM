"""Evaluate a trained policy (deterministic / argmax) against the heuristic.

Runs both agents on the *same held-out seeds* (default 100000+, outside the
training range) so the comparison is paired and tests generalisation, not recall.
Reports mean clear ratio, full-clear rate, task time, and episode length.

    python scripts/eval.py                                   # ppo checkpoint vs heuristic, 30 eps
    python scripts/eval.py --episodes 50
    python scripts/eval.py --checkpoint D:/.../model.pt

The PPO weights are loaded via the config's `algorithm.checkpoint` (top-level
load path); `deterministic_eval: true` makes the pipeline take argmax.
"""

from __future__ import annotations

import argparse
import statistics as stats_mod
import sys

from radio_rl.core.config import compose
from radio_rl.pipeline import Pipeline


def run_agent(overrides: list[str], seeds: list[int], max_steps: int) -> list[dict]:
    pipe = Pipeline(compose(overrides=overrides))
    out = []
    for s in seeds:
        st, _ = pipe.run_episode(seed=s, max_steps=max_steps)
        total = st.sources_total or 0
        out.append({
            "ratio": (st.sources_cleared / total) if total > 0 else 0.0,
            "cleared": st.sources_cleared,
            "total": total,
            "vt": st.virtual_time,
        })
    return out


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    ratio = sum(r["ratio"] for r in rows) / n
    full = sum(1 for r in rows if r["total"] > 0 and r["cleared"] >= r["total"]) / n
    vt = sum(r["vt"] for r in rows) / n
    cleared = sum(r["cleared"] for r in rows) / n
    total = sum(r["total"] for r in rows) / n
    ratio_sd = stats_mod.pstdev([r["ratio"] for r in rows]) if n > 1 else 0.0
    print(
        f"{name:<12s} clear={ratio:.3f}±{ratio_sd:.2f}  full_clear={full:.0%}  "
        f"cleared={cleared:.1f}/{total:.1f}  vt={vt:,.0f}s",
        flush=True,
    )
    return {"clear": ratio, "full": full, "vt": vt}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--checkpoint", type=str, default="D:/George/SX/Way2/runs/ppo/model.pt")
    args = ap.parse_args(argv)

    seeds = [args.base_seed + i for i in range(args.episodes)]
    print(f"eval on {args.episodes} held-out seeds [{seeds[0]}..{seeds[-1]}]", flush=True)
    print("=" * 72, flush=True)

    ppo = run_agent(
        ["algorithm=ppo", f"algorithm.checkpoint={args.checkpoint}"], seeds, args.max_steps
    )
    heur = run_agent(["algorithm=heuristic"], seeds, args.max_steps)

    p = summarize("PPO (argmax)", ppo)
    h = summarize("heuristic", heur)

    print("=" * 72, flush=True)
    dc = p["clear"] - h["clear"]
    print(
        f"delta: clear {dc:+.3f} ({'PPO' if dc >= 0 else 'heuristic'} better), "
        f"time {p['vt'] - h['vt']:+,.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
