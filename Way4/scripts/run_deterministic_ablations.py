"""Execute the deterministic ablation cells that the current Pipeline exposes.

Unsupported cells are retained as explicit errors.  The script never fills a
missing result with a synthetic value and is therefore safe to use as an audit
artifact while the remaining planner switches are being wired.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parents[1] / "src")]

from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.evaluation_protocol import DETERMINISTIC_ABLATIONS
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline


def run_cell(name, seed, max_steps):
    supported = {
        "way4_full": dict(planner_mode="route_math"),
        "adaptive_rerank": dict(planner_mode="route_math", adaptive_scan=True),
        "adaptive_rerank_stop": dict(planner_mode="route_math", adaptive_scan=True, batch_stop=True),
    }
    if name not in supported:
        return {"seed": seed, "success": None,
                "error": "unsupported_configuration", "virtual_time_s": None}
    try:
        case = generate_case(seed=seed, problem=4, field_kind="smooth", mode="formal")
        engine = Engine(case)
        engine.enter()
        result = Way4Pipeline(
            Way3EngineAdapter(engine), n_channels=20, problem=4,
            max_steps=max_steps, **supported[name],
        ).run()
        return {
            "seed": seed,
            # Pipeline success includes clean EXIT and resolved belief; truth
            # clearance is reported separately and must not mask a stall.
            "success": bool(result.success),
            "full_clear": bool(case.cleared_count == case.total),
            "virtual_time_s": result.virtual_time_s,
            "error": result.error,
            "steps": result.steps,
        }
    except Exception as exc:  # preserve the failed cell for audit
        return {"seed": seed, "success": None,
                "error": f"{type(exc).__name__}: {exc}", "virtual_time_s": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2000-2001")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if "-" in args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-", 1))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
    results = {}
    for spec in DETERMINISTIC_ABLATIONS:
        results[spec.name] = [run_cell(spec.name, seed, args.max_steps) for seed in seeds]
    payload = {"problem": 4, "seeds": seeds, "results": results,
               "unsupported_are_explicit": True}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
