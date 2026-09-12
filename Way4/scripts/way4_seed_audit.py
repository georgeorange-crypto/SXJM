"""Fixed-seed end-to-end smoke/invariant audit on the bundled simulator."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve(); ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parents[1] / "src")]
from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline

def run(seed, problem, max_steps):
    case = generate_case(seed=seed, problem=problem, field_kind="smooth", mode="formal")
    eng = Engine(case); eng.enter()
    result = Way4Pipeline(Way3EngineAdapter(eng), n_channels=20,
                          problem=problem, max_steps=max_steps).run()
    resolved = result.resolved == result.n_channels
    row = {"seed": seed, "problem": problem, "truth_total": case.total,
           "truth_cleared": case.cleared_count, "resolved": result.resolved,
           "n_channels": result.n_channels, "full_clear": case.cleared_count == case.total,
           "exited": result.exited, "error": result.error, "steps": result.steps,
           "virtual_time_s": result.virtual_time_s}
    row["invariants_passed"] = bool(row["error"] is None and resolved and row["full_clear"])
    return row

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds", default="1000-1004")
    p.add_argument("--problem", type=int, default=3); p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--out", required=True); a=p.parse_args()
    if "-" in a.seeds:
        lo, hi = map(int, a.seeds.split("-")); seeds=list(range(lo, hi+1))
    else: seeds=[int(x) for x in a.seeds.split(",")]
    rows=[run(s,a.problem,a.max_steps) for s in seeds]
    payload={"audit":"way4-end-to-end-invariants-v1","problem":a.problem,"seeds":seeds,
             "rows":rows,"all_invariants_passed":all(r["invariants_passed"] for r in rows)}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2)); return 0 if payload["all_invariants_passed"] else 2
if __name__ == "__main__": raise SystemExit(main())
