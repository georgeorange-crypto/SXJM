"""M3.2 focused reward sweep: full_clear_bonus x time_coef, with validation
checkpointing, a train->val trajectory, tail-behaviour diagnostics, and tiered
model selection. Centred on the `completeness` working point; every other reward
term is frozen.

Why this shape (see the M3.1 sweep that motivated it):
  * The 200-iter headline checkpoint OVERTRAINED — same reward, same seeds, it
    scored 0.752/0% full vs the 80-iter run's 0.910/40%, while *training* clear
    rose to 0.96. So training iteration is itself a hyperparameter: we validate
    every few iters and keep the best-by-VALIDATION checkpoint, never iter=last.
  * `completeness` (exit=8, succ=20, clr=6, fcb=30, tc=1) already matched the
    heuristic (0.945/52%/8630s vs 0.944/52%/8692s). The single-knob probes showed
    the two forces that matter are ORTHOGONAL: the all-clear CLIFF (full_clear_bonus)
    forces finishing the tail; time_coef stops the agent roaming 24k s to do it.
    Linear per-source rewards (big_clear/big_success) HURT — they buy easy sources
    and abandon the hard tail — so they stay frozen. This sweep varies only the
    two orthogonal knobs to find the completeness/efficiency knee.

Seed hygiene (the M3.1 "25 held-out seeds" were reused to compare rewards + select
configs, so they are really a VALIDATION set now, not a generalisation estimate):
  * training seeds are root_seed(0)+1..(iters*8)  -> <= ~1600
  * VAL      = [100000 .. 100000+val-1]        (default 100; superset of the old 25)
  * FINAL    = [200000 .. 200000+final-1]      (default 100; NEVER touched here)
FINAL is only for the one run after the model + hyperparameters are frozen; this
script deliberately never evaluates on it.

Tail diagnostics (local ground truth, used OFFLINE in this harness only — never
fed to the Actor, so the "N must not enter the observation" invariant holds):
  * terminal remain r = sources_total - sources_cleared, and whether the episode
    ended on an explicit EXIT -> P(EXIT | remain=1/2/3), the tail-abandonment rate.
  * gaps between consecutive clear_times -> T(N-1->N), T(N-2->N-1): how long the
    last/second-last source takes once localisation is mostly done.

Selection is a full-clear TIER (within `--tier` cases of the best counts as the
same completeness tier) then clear ratio then virtual time — robust to the ~1/100
noise a strict full>clear>time ordering would chase.

    python scripts/sweep_reward.py                         # 6 profiles, 150 iters, val 100
    python scripts/sweep_reward.py --iters 120 --val 100 --val-every 5
    python scripts/sweep_reward.py --only A,E
"""

from __future__ import annotations

import argparse
import statistics as stats_mod
import sys
import time

from radio_rl.core.datatypes import ActionType
from radio_rl.core.config import compose
from radio_rl.pipeline import Pipeline
from radio_rl.training import train

CKPT_DIR = "D:/George/SX/Way2/runs/m32"

# Frozen reward context (the completeness working point minus the two swept knobs).
FROZEN = dict(exit_penalty=8.0, success_bonus=20.0, clear_success_bonus=6.0, shaping_coef=1.5)

# The 2x3 grid: only full_clear_bonus x time_coef move.
PROFILES: dict[str, dict[str, float]] = {
    "A": dict(full_clear_bonus=30.0, time_coef=1.00),   # == current completeness
    "B": dict(full_clear_bonus=30.0, time_coef=1.25),
    "C": dict(full_clear_bonus=30.0, time_coef=1.50),
    "D": dict(full_clear_bonus=50.0, time_coef=1.00),
    "E": dict(full_clear_bonus=50.0, time_coef=1.25),   # the a-priori favourite
    "F": dict(full_clear_bonus=50.0, time_coef=1.50),
}


# --- evaluation -------------------------------------------------------------
def eval_episodes(pipe: Pipeline, seeds: list[int], max_steps: int, agent=None) -> list[dict]:
    """Argmax eval on ``seeds``; per-episode clear/full/vt (+ tail fields).

    If ``agent`` is given (the live training agent) it is flipped out of recording
    mode for the duration so ``select`` takes argmax, then restored.
    """
    ag = agent if agent is not None else pipe.agent
    was_rec = getattr(ag, "_recording", False)
    if hasattr(ag, "_recording"):
        ag._recording = False
    per = []
    for s in seeds:
        st, trace = pipe.run_episode(seed=s, max_steps=max_steps, collect_trace=True)
        total = st.sources_total or 0
        remain = total - st.sources_cleared
        exited = bool(trace) and trace[-1].action.action_type == ActionType.EXIT
        ct = sorted(st.clear_times)
        per.append({
            "clear": (st.sources_cleared / total) if total > 0 else 0.0,
            "full": 1.0 if total > 0 and st.sources_cleared >= total else 0.0,
            "vt": float(st.virtual_time),
            "remain": remain,
            "exited": exited,
            # tail gaps: time to clear the last / second-last source
            "gap_last": (ct[-1] - ct[-2]) if len(ct) >= 2 else None,
            "gap_prev": (ct[-2] - ct[-3]) if len(ct) >= 3 else None,
        })
    if hasattr(ag, "_recording"):
        ag._recording = was_rec
    return per


def agg(per: list[dict]) -> dict:
    n = len(per)
    clears = [p["clear"] for p in per]
    return {
        "clear": sum(clears) / n,
        "clear_sd": stats_mod.pstdev(clears) if n > 1 else 0.0,
        "full": sum(p["full"] for p in per) / n,
        "full_n": int(sum(p["full"] for p in per)),
        "vt": sum(p["vt"] for p in per) / n,
        "n": n,
    }


def tail_diag(per: list[dict]) -> dict:
    """P(EXIT | remain=k) for k=1,2,3 and mean last/second-last clear gaps."""
    out = {}
    for k in (1, 2, 3):
        grp = [p for p in per if p["remain"] == k]
        ex = [p for p in grp if p["exited"]]
        out[f"exit_r{k}"] = (len(ex) / len(grp)) if grp else float("nan")
        out[f"n_r{k}"] = len(grp)
    gl = [p["gap_last"] for p in per if p["gap_last"] is not None]
    gp = [p["gap_prev"] for p in per if p["gap_prev"] is not None]
    out["gap_last"] = (sum(gl) / len(gl)) if gl else float("nan")
    out["gap_prev"] = (sum(gp) / len(gp)) if gp else float("nan")
    return out


# --- per-profile training with validation checkpointing ---------------------
def run_profile(name: str, knobs: dict[str, float], iters: int, max_steps: int,
                val_seeds: list[int], val_every: int, tier: int) -> dict:
    """Train one profile; validate every ``val_every`` iters; keep best-by-tier.

    Returns the best checkpoint's aggregate + tail diagnostics + the trajectory.
    Saves best.pt (tiered pick) and last.pt under runs/m32/<name>/.
    """
    import os
    import torch

    outdir = f"{CKPT_DIR}/{name}"
    os.makedirs(outdir, exist_ok=True)
    rew = {**FROZEN, **knobs}
    tokens = [
        "algorithm=ppo",
        f"algorithm.train.iterations={iters}",
        "algorithm.train.eval_every=0",           # we drive validation from the callback
        f"algorithm.train.max_steps={max_steps}",
        "algorithm.train.checkpoint=null",        # callback owns checkpointing
    ]
    tokens += [f"algorithm.reward.{k}={v}" for k, v in rew.items()]

    traj: list[dict] = []
    best = {"score": None, "state": None, "it": -1, "per": None}

    def better(cand_agg: dict, cur: dict) -> bool:
        # tiered: same completeness tier (within `tier` full-clears) -> clear -> vt
        if cur["score"] is None:
            return True
        c, b = cand_agg, cur["score"]
        if abs(c["full_n"] - b["full_n"]) > tier:
            return c["full_n"] > b["full_n"]
        if abs(c["clear"] - b["clear"]) > 1e-6:
            return c["clear"] > b["clear"]
        return c["vt"] < b["vt"]

    def cb(it: int, pipe: Pipeline, agent, rec: dict) -> None:
        if (it + 1) % val_every != 0 and it != iters - 1:
            return
        per = eval_episodes(pipe, val_seeds, max_steps, agent=agent)
        a = agg(per)
        traj.append({"it": it, "train_clear": rec["mean_clear_ratio"], **a})
        if better(a, best):
            best.update(score=a, state={k: v.clone() for k, v in agent.state_dict().items()},
                        it=it, per=per)

    t0 = time.time()
    res = train(compose(overrides=tokens), progress=False, iter_callback=cb)
    dt = time.time() - t0

    torch.save(res.agent.state_dict(), f"{outdir}/last.pt")
    if best["state"] is not None:
        torch.save(best["state"], f"{outdir}/best.pt")

    ba = best["score"] or {}
    return {
        "name": name, **knobs, **ba,
        "best_it": best["it"], "sec": dt,
        "tail": tail_diag(best["per"]) if best["per"] else {},
        "traj": traj,
        "ckpt": f"{outdir}/best.pt",
    }


def tiered_pick(rows: list[dict], tier: int) -> dict:
    best_full = max(r["full_n"] for r in rows)
    tieband = [r for r in rows if best_full - r["full_n"] <= tier]
    tieband.sort(key=lambda r: (-r["clear"], r["vt"]))
    return tieband[0]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--val", type=int, default=100, help="validation seed count")
    ap.add_argument("--val-base", type=int, default=100000)
    ap.add_argument("--final", type=int, default=100, help="final-test seed count (reserved, unused)")
    ap.add_argument("--final-base", type=int, default=200000)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--tier", type=int, default=2, help="full-clear tier band (cases)")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args(argv)

    val_seeds = [args.val_base + i for i in range(args.val)]
    names = [n.strip().upper() for n in args.only.split(",") if n.strip()] or list(PROFILES)
    bad = [n for n in names if n not in PROFILES]
    if bad:
        print(f"unknown profiles: {bad}; known: {list(PROFILES)}", flush=True)
        return 2

    print(f"M3.2 sweep | full_clear_bonus x time_coef | frozen: exit=8 succ=20 clr=6 shaping=1.5",
          flush=True)
    print(f"train<= {args.iters} iters, validate every {args.val_every} on {args.val} seeds "
          f"[{val_seeds[0]}..{val_seeds[-1]}]; final-test [{args.final_base}..{args.final_base+args.final-1}] RESERVED",
          flush=True)
    print("=" * 100, flush=True)

    # heuristic target on the SAME validation seeds
    hpipe = Pipeline(compose(overrides=["algorithm=heuristic"]))
    hper = eval_episodes(hpipe, val_seeds, args.max_steps)
    h = agg(hper)
    ht = tail_diag(hper)
    print(f"heuristic (target)  clear={h['clear']:.3f}±{h['clear_sd']:.2f}  "
          f"full={h['full']:.0%} ({h['full_n']}/{h['n']})  vt={h['vt']:,.0f}s   "
          f"P(exit|r=1/2/3)={ht['exit_r1']:.0%}/{ht['exit_r2']:.0%}/{ht['exit_r3']:.0%}",
          flush=True)
    print("-" * 100, flush=True)

    rows: list[dict] = []
    for name in names:
        try:
            r = run_profile(name, PROFILES[name], args.iters, args.max_steps,
                            val_seeds, args.val_every, args.tier)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(r)
        t = r["tail"]
        print(f"[{name}] fcb={r['full_clear_bonus']:.0f} tc={r['time_coef']:.2f}  "
              f"best@it{r['best_it']:>3d}: clear={r['clear']:.3f} full={r['full']:.0%} "
              f"({r['full_n']}/{r['n']}) vt={r['vt']:,.0f}s  "
              f"P(exit|r1/2/3)={t.get('exit_r1', float('nan')):.0%}/"
              f"{t.get('exit_r2', float('nan')):.0%}/{t.get('exit_r3', float('nan')):.0%}  "
              f"tail T(N-1->N)={t.get('gap_last', float('nan')):,.0f}s  ({r['sec']:.0f}s)",
              flush=True)
        # train->val trajectory: watch the complete->fast slide
        print(f"      trajectory (it: trainClr | val full clear vt):", flush=True)
        for e in r["traj"]:
            mark = "  <-best" if e["it"] == r["best_it"] else ""
            print(f"        it{e['it']:>3d}: {e['train_clear']:.2f} | "
                  f"full={e['full']:.0%} clear={e['clear']:.3f} vt={e['vt']:,.0f}s{mark}",
                  flush=True)

    if not rows:
        print("no successful profiles", flush=True)
        return 1

    print("=" * 100, flush=True)
    pick = tiered_pick(rows, args.tier)
    print(f"tiered pick (full-clear tier +/-{args.tier} -> clear -> vt): "
          f"[{pick['name']}] fcb={pick['full_clear_bonus']:.0f} tc={pick['time_coef']:.2f}  "
          f"full={pick['full']:.0%} ({pick['full_n']}/{pick['n']}) clear={pick['clear']:.3f} "
          f"vt={pick['vt']:,.0f}s  ckpt={pick['ckpt']}", flush=True)
    print(f"vs heuristic: full {pick['full_n']}/{pick['n']} vs {h['full_n']}/{h['n']}, "
          f"clear {pick['clear']:.3f} vs {h['clear']:.3f}, vt {pick['vt']:,.0f} vs {h['vt']:,.0f}s",
          flush=True)
    print("next: freeze reward+hyperparams, then evaluate this ckpt ONCE on the reserved final-test seeds.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
