"""Same-seed baseline gate for SpatialStop DBSCAN clustering."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve()
way3_candidates = (
    HERE.parents[2] / "Way3",
    HERE.parents[3] / "MathModelingCode" / "external" / "SXJM-way3" / "Way3",
)
way3_root = next((p for p in way3_candidates if (p / "jammerhunt").is_dir()), way3_candidates[0])
for p in (HERE.parents[1] / "src", HERE.parents[2], way3_root):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

def run(seed, problem, mode, max_steps):
    from jammerhunt import environment as env
    from way4.executor import Way3EngineAdapter
    from way4.pipeline import Way4Pipeline
    case = env.generate_case(seed=seed, problem=problem, field_kind="smooth")
    engine = env.Engine(case); engine.enter()
    result = Way4Pipeline(Way3EngineAdapter(engine), problem=problem,
                          max_steps=max_steps, planner_mode="spatial",
                          spatial_clustering=mode).run()
    return {
        "seed": seed, "mode": mode, "success": bool(case.cleared_count == case.total),
        "virtual_time_s": float(result.virtual_time_s), "distance_m": float(result.move_distance_m),
        "n_measure": int(result.n_measure), "resolved": int(result.resolved),
        "error": result.error,
        "finish_reason": getattr(engine, "finish_reason", None),
        "decision_summary": result.decision_summary,
    }

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10); ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--problem", type=int, default=3, choices=(3,4))
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    # Way3 is an optional project sibling; parse/help and static validation work
    # without it, while execution reports the exact missing dependency.
    try:
        import jammerhunt  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("Way3/jammerhunt is required to run this comparison: " + str(exc))
    rows = [run(args.seed+i, args.problem, mode, args.max_steps)
            for i in range(args.n) for mode in ("none", "dbscan")]
    result = {"rows": rows, "full_clear_rate": {
        mode: sum(r["success"] for r in rows if r["mode"] == mode) / max(1, args.n)
        for mode in ("none", "dbscan")}}
    text = json.dumps(result, indent=2)
    if args.out: args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0

if __name__ == "__main__": raise SystemExit(main())
