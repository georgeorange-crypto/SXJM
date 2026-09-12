"""Same-seed DBSCAN ablation on the local SXJM/offline_sim.

Evidence boundary: this is an offline constraint-faithful simulator study, not
the official Way3/jammerhunt result.  It is kept separate from the Way3 gate.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve()
for p in (HERE.parents[1] / "src", HERE.parents[2]):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline

def run(seed, problem, mode, max_steps, field_kind):
    case = generate_case(seed=seed, problem=problem, field_kind=field_kind, mode="formal")
    engine = Engine(case); engine.enter()
    result = Way4Pipeline(Way3EngineAdapter(engine), problem=problem,
                          max_steps=max_steps, planner_mode="spatial",
                          spatial_clustering=mode).run()
    return {"seed": seed, "mode": mode,
            "success_ground_truth": bool(case.cleared_count == case.total),
            "cleared": int(case.cleared_count), "total": int(case.total),
            "virtual_time_s": float(result.virtual_time_s),
            "move_distance_m": float(result.move_distance_m),
            "n_measure": int(result.n_measure), "error": result.error,
            "decision_summary": result.decision_summary}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2); ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--problem", type=int, default=3, choices=(3, 4))
    ap.add_argument("--field", default="smooth")
    ap.add_argument("--max-steps", type=int, default=2000); ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rows = [run(args.seed+i, args.problem, mode, args.max_steps, args.field)
            for i in range(args.n) for mode in ("none", "dbscan")]
    result = {"evidence": "SXJM/offline_sim, not official Way3",
              "rows": rows,
              "full_clear_rate": {m: sum(r["success_ground_truth"] for r in rows if r["mode"] == m) / max(1, args.n)
                                  for m in ("none", "dbscan")}}
    text = json.dumps(result, indent=2)
    if args.out: args.out.write_text(text, encoding="utf-8")
    print(text); return 0

if __name__ == "__main__": raise SystemExit(main())
