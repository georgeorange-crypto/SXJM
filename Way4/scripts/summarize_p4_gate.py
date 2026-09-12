"""Summarize P4 gate artifacts without hiding incomplete or failed rows."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(paths):
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    rows = [r for d in docs for r in d.get("rows", [])]
    seed_counts = {}
    for r in rows:
        if r.get("seed") is not None:
            seed = int(r["seed"])
            seed_counts[seed] = seed_counts.get(seed, 0) + 1
    duplicate_seeds = sorted(seed for seed, count in seed_counts.items() if count > 1)
    successes = [float(r["time_s"]) for r in rows
                 if r.get("full_clear") and r.get("time_s") is not None]
    successes.sort()
    rho_values = [float(r["rho_T_over_LB"]) for r in rows
                  if r.get("full_clear") and r.get("rho_T_over_LB") is not None]
    def pct(p):
        if not successes:
            return None
        x = p * (len(successes) - 1)
        lo, hi = int(x), int(x) + (1 if x % 1 else 0)
        return successes[lo] if lo == hi else successes[lo] + (successes[hi] - successes[lo]) * (x - lo)
    return {
        "artifacts": [str(p) for p in paths],
        "n_artifacts": len(docs),
        "n_rows": len(rows),
        "duplicate_seeds": duplicate_seeds,
        "has_duplicate_seeds": bool(duplicate_seeds),
        "n_full_clear": len(successes),
        "full_clear_rate": len(successes) / len(rows) if rows else None,
        "failed_rows": [{"seed": r.get("seed"), "error": r.get("error"),
                         "truth_cleared": r.get("truth_cleared"),
                         "truth_total": r.get("truth_total")}
                        for r in rows if not r.get("full_clear")],
        "full_clear_time_s": {
            "n": len(successes),
            "mean": statistics.fmean(successes) if successes else None,
            "median": statistics.median(successes) if successes else None,
            "p90": pct(.90), "p95": pct(.95),
            "max": max(successes) if successes else None,
        },
        "rho_over_lb": {
            "n": len(rho_values),
            "mean": statistics.fmean(rho_values) if rho_values else None,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifacts", nargs="+", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    result = summarize(a.artifacts)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
