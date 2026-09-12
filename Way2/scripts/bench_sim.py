"""Throughput benchmark: is the simulator fast enough to train on?

Replicates the frozen pipeline loop (``pipeline.run_episode``) but times each
stage separately, so we can see where wall-clock actually goes:

    gen    - candidate generation (geometry math)
    feat   - feature building (learnable path only)
    select - agent decision (NN forward for PPO; math for heuristic)
    env    - env.execute(...) == the *simulator* step itself
    other  - belief update, shield, bookkeeping

Then it times one real PPO training iteration (rollout + PPO update) to project
the full run. Run:  python scripts/bench_sim.py
"""

from __future__ import annotations

import time

import torch

from radio_rl.core.config import compose
from radio_rl.core.datatypes import ActionType
from radio_rl.pipeline import Pipeline
from radio_rl.training import train


def bench(overrides, n_eps=15, max_steps=2000, warm=1):
    cfg = compose(overrides=overrides)
    pipe = Pipeline(cfg)
    needs = bool(getattr(pipe.agent, "needs_features", False))
    for w in range(warm):
        pipe.run_episode(seed=1000 + w, max_steps=max_steps)

    acc = dict(gen=0.0, feat=0.0, select=0.0, env=0.0, other=0.0)
    steps_total = 0
    lens = []
    wall0 = time.perf_counter()
    for seed in range(n_eps):
        obs = pipe.env.reset(seed)
        belief = pipe.new_belief()
        belief.update(obs)
        pipe.agent.reset()
        s = 0
        for _ in range(max_steps):
            if pipe.env.finished:
                break
            a = time.perf_counter()
            cands = pipe.generator.generate(belief)
            b = time.perf_counter()
            feats = pipe.feature_builder.build(belief, cands) if needs else None
            c = time.perf_counter()
            idx = pipe.agent.select(cands, belief, feats)
            d = time.perf_counter()
            idx = max(0, min(idx, len(cands.candidates) - 1))
            cand = cands.candidates[idx]
            action = pipe.shield.apply(
                cand, remaining_real_s=pipe.env.remaining_real_duration_s()
            )
            e = time.perf_counter()
            if action.action_type == ActionType.EXIT:
                pipe.env.execute(action)
                f = time.perf_counter()
                acc["gen"] += b - a
                acc["feat"] += c - b
                acc["select"] += d - c
                acc["other"] += e - d
                acc["env"] += f - e
                s += 1
                break
            obs = pipe.env.execute(action)
            f = time.perf_counter()
            belief.update(obs)
            g = time.perf_counter()
            acc["gen"] += b - a
            acc["feat"] += c - b
            acc["select"] += d - c
            acc["other"] += (e - d) + (g - f)
            acc["env"] += f - e
            s += 1
        steps_total += s
        lens.append(s)
    wall = time.perf_counter() - wall0

    print("=" * 66)
    print(f"config: {overrides}   needs_features={needs}")
    print(f"episodes={n_eps}  steps_total={steps_total}  "
          f"avg_len={steps_total / n_eps:.1f}  min={min(lens)}  max={max(lens)}")
    print(f"wall={wall:.3f}s   eps/s={n_eps / wall:.1f}   "
          f"steps/s={steps_total / wall:,.0f}")
    us = {k: acc[k] / max(steps_total, 1) * 1e6 for k in acc}
    tot = sum(us.values())
    for k in ("gen", "feat", "select", "env", "other"):
        print(f"  {k:<7s} {us[k]:8.1f} us/step  ({us[k] / tot * 100:4.1f}%)")
    print(f"  {'TOTAL':<7s} {tot:8.1f} us/step")
    return steps_total / wall


def main():
    print("torch:", torch.__version__, " threads:", torch.get_num_threads())
    bench(["algorithm=heuristic"], n_eps=15)
    bench(["algorithm=ppo"], n_eps=15)

    cfg = compose(overrides=[
        "algorithm=ppo",
        "algorithm.train.iterations=1",
        "algorithm.train.episodes_per_iter=8",
        "algorithm.train.max_steps=2000",
        "algorithm.train.eval_every=0",
        "algorithm.train.checkpoint=null",
    ])
    t0 = time.perf_counter()
    res = train(cfg, progress=False)
    dt = time.perf_counter() - t0
    rec = res.history[0]
    print("=" * 66)
    print(f"train() 1 iter (8 eps + ppo update): {dt:.2f}s")
    print(f"  history[0] keys: {list(rec.keys())}")
    print(f"  -> full 200-iter run ~= {dt * 200 / 60:.1f} min "
          f"(rollout+update, excludes periodic eval)")


if __name__ == "__main__":
    main()
