"""Aggregate ``evaluate_nbv_objectives.py`` artifacts without losing seed detail."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(paths):
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not docs:
        raise ValueError("no input artifacts")
    modes = sorted(docs[0]["modes"])
    out = {"benchmark": docs[0].get("benchmark"), "n_artifacts": len(docs),
           "seeds": [d.get("seed") for d in docs], "trials_total": sum(d["trials"] for d in docs),
           "modes": {}}
    for mode in modes:
        rows = [d["modes"][mode] for d in docs]
        out["modes"][mode] = {
            "valid_total": sum(r["valid"] for r in rows),
            "miss_rate_mean_over_seeds": sum(r["miss_rate"] for r in rows) / len(rows),
            "miss_rate_max_over_seeds": max(r["miss_rate"] for r in rows),
            "mean_realized_diameter_m_mean_over_seeds": sum(r["mean_realized_diameter_m"] for r in rows) / len(rows),
            "worst_realized_diameter_m_max_over_seeds": max(r["worst_realized_diameter_m"] for r in rows),
            "mean_objective_score_mean_over_seeds": sum(r["mean_objective_score"] for r in rows) / len(rows),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
