"""Fixed-seed negative soundness audit for Way4 belief geometry."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import sys
HERE = Path(__file__).resolve(); ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parents[1] / "src")]
from way4.belief import ChannelBelief

ARENA = 1800.0

def source(rng):
    a = rng.random() * 2 * math.pi
    r = ARENA * math.sqrt(rng.random())
    return r * math.cos(a), r * math.sin(a)

def bearing(src, p):
    return math.degrees(math.atan2(src[1] - p[1], src[0] - p[0]))

def run(seed: int, trials: int):
    rng = random.Random(seed); no_signal = 0; clear_checks = 0
    for _ in range(trials):
        src = source(rng)
        b = ChannelBelief(1)
        for _ in range(rng.randint(1, 6)):
            while True:
                p = (rng.uniform(-ARENA, ARENA), rng.uniform(-ARENA, ARENA))
                if math.dist(src, p) > 1000.0: break
            b.record_no_signal(p)
        if not b.contains_possible_source(src):
            raise AssertionError(("NO_SIGNAL_EXCLUDED", seed, src))
        no_signal += 1
        d = rng.uniform(90.0, 240.0); base = rng.random() * 2 * math.pi
        c = ChannelBelief(7)
        for j in range(3):
            a = base + j * 2 * math.pi / 3 + rng.uniform(-0.3, 0.3)
            p = (src[0] + d * math.cos(a), src[1] + d * math.sin(a))
            c.record_bearing(p, bearing(src, p) + rng.uniform(-1.0, 1.0))
        if c.is_clearable:
            clear_checks += 1
            if math.dist(c.clear_target, src) > c.clear_threshold + 1e-4:
                raise AssertionError(("FALSE_CLEAR", seed, src))
    return {"seed": seed, "trials": trials, "no_signal_sound": no_signal,
            "clear_checks": clear_checks, "passed": True}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--seeds", default="11,23,47,71,99")
    p.add_argument("--trials", type=int, default=500); p.add_argument("--out", required=True)
    a = p.parse_args(); rows = [run(int(s), a.trials) for s in a.seeds.split(",")]
    payload = {"audit": "negative-soundness-v1", "rows": rows,
               "all_passed": all(r["passed"] for r in rows),
               "total_trials": sum(r["trials"] for r in rows),
               "total_clear_checks": sum(r["clear_checks"] for r in rows)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_passed"] else 2

if __name__ == "__main__": raise SystemExit(main())
