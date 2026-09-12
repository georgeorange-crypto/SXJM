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
        "routing_nearest": dict(planner_mode="route_math", routing_strategy="nearest"),
        "nbv_greedy": dict(planner_mode="route_math", nbv_objective="greedy"),
        "coverage_backbone_only": dict(planner_mode="route_math", coverage_strategy="backbone_only"),
        "minus_no_signal": dict(planner_mode="route_math", enable_no_signal=False),
        "minus_cardinality": dict(planner_mode="route_math", enable_cardinality=False),
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
            "resolved": result.resolved,
            "n_channels": result.n_channels,
            "present": result.present,
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
    parser.add_argument("--cells", default=None,
                        help="comma-separated cell names; default runs the full frozen matrix")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if "-" in args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-", 1))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
    selected = set(args.cells.split(",")) if args.cells else {s.name for s in DETERMINISTIC_ABLATIONS}
    unknown = selected - {s.name for s in DETERMINISTIC_ABLATIONS}
    if unknown:
        parser.error(f"unknown cells: {sorted(unknown)}")
    results = {}
    for spec in DETERMINISTIC_ABLATIONS:
        if spec.name not in selected:
            continue
        results[spec.name] = []
        for seed in seeds:
            results[spec.name].append(run_cell(spec.name, seed, args.max_steps))
            # Preserve every completed episode if a later cell is slow or fails.
            partial = {"problem": 4, "seeds": seeds, "results": results,
                       "selected_cells": sorted(selected),
                       "unsupported_are_explicit": True, "complete": False}
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"problem": 4, "seeds": seeds, "results": results,
               "selected_cells": sorted(selected),
               "unsupported_are_explicit": True, "complete": True}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
