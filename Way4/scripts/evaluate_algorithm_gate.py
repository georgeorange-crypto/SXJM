"""Evaluate an ablation JSON with the full-clear non-regression gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
from way4.analysis import algorithm_gate

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--baseline", default="none")
    ap.add_argument("--variant", default="dbscan")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload)
    result = algorithm_gate(
        [r for r in rows if r.get("mode") == args.baseline],
        [r for r in rows if r.get("mode") == args.variant],
    )
    result["input"] = str(args.input)
    result["baseline"] = args.baseline
    result["variant"] = args.variant
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out: args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0

if __name__ == "__main__": raise SystemExit(main())
