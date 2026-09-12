"""Summarize an ablation runner artifact without inventing missing cells."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
from way4.evaluation_protocol import DETERMINISTIC_ABLATIONS, summarize_ablation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+",
                        help="one or more real runner artifacts; rows are concatenated by cell")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.input]
    merged = {}
    for payload in payloads:
        for name, rows in payload.get("results", {}).items():
            merged.setdefault(name, []).extend(rows)
    summary = summarize_ablation_results(DETERMINISTIC_ABLATIONS, merged)
    summary.update({
        "source": [str(Path(path)) for path in args.input],
        "problem": payloads[0].get("problem") if payloads else None,
        "seeds": sorted({int(seed) for payload in payloads for seed in payload.get("seeds", [])}),
        "selected_cells": sorted(merged),
        "runner_complete": all(bool(payload.get("complete", False)) for payload in payloads),
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
