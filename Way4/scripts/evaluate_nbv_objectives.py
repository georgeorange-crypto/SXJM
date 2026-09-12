"""Compare P2 NBV objectives on a reproducible synthetic single-source suite.

This is a planning benchmark, not an official simulator score.  All modes see
the same source, first bearing, and second-bearing noise for each trial.  The
reported miss rate uses the actual source and the guaranteed 1000 m detection
radius; it is deliberately kept separate from hard belief/certificate state.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Way4" / "src"))

from sxjm_core.geometry import bearing_deg, dist, norm_deg
from way4.belief import ChannelBelief
from way4.sensing import MinimaxNBV


def _belief(apex, source, noise):
    b = ChannelBelief(channel=1)
    b.record_bearing(apex, norm_deg(bearing_deg(apex, source) + noise))
    return b


def run(trials: int = 100, seed: int = 20260913):
    rng = random.Random(seed)
    modes = ("minimax", "expected", "time")
    rows = {m: {"valid": 0, "misses": 0, "diameters": [], "expected_diameters": [],
                "travel_s": []} for m in modes}
    for _ in range(int(trials)):
        rho = rng.uniform(300.0, 1400.0)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        source = (rho * math.cos(phi), rho * math.sin(phi))
        first = (0.0, 0.0)
        b1 = rng.uniform(-1.0, 1.0)
        b2 = rng.uniform(-1.0, 1.0)
        for mode in modes:
            belief = _belief(first, source, b1)
            result = MinimaxNBV(objective=mode, lambda_t=0.01).choose(belief, first)
            if result is None:
                continue
            rows[mode]["valid"] += 1
            if dist(result.point, source) > 1000.0:
                rows[mode]["misses"] += 1
                diameter = belief.diameter
            else:
                belief.record_bearing(result.point,
                                      norm_deg(bearing_deg(result.point, source) + b2))
                diameter = belief.diameter
            rows[mode]["diameters"].append(float(diameter))
            rows[mode]["expected_diameters"].append(float(result.expected_diameter))
            rows[mode]["travel_s"].append(float(dist(first, result.point) / 5.0 + 5.0))

    summary = {"seed": seed, "trials": int(trials), "benchmark": "synthetic_single_source",
               "guaranteed_detection_radius_m": 1000.0, "modes": {}}
    for mode, row in rows.items():
        n = row["valid"]
        ds = row["diameters"]
        summary["modes"][mode] = {
            "valid": n,
            "miss_rate": row["misses"] / n if n else 1.0,
            "mean_realized_diameter_m": sum(ds) / n if n else None,
            "p95_realized_diameter_m": sorted(ds)[min(n - 1, math.ceil(0.95 * n) - 1)] if n else None,
            "worst_realized_diameter_m": max(ds) if ds else None,
            "mean_predicted_diameter_m": sum(row["expected_diameters"]) / n if n else None,
            "mean_objective_score": sum(row["travel_s"]) / n if n else None,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(args.trials, args.seed)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
