"""Merge independent ablation artifacts while preserving every raw row."""
import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True, type=Path); a = p.parse_args()
    payloads = [json.loads(Path(x).read_text(encoding="utf-8")) for x in a.inputs]
    results = {}
    for payload in payloads:
        for name, rows in payload.get("results", {}).items():
            results.setdefault(name, []).extend(rows)
    seeds = sorted({int(r["seed"]) for rows in results.values() for r in rows})
    merged = {"problem": payloads[0].get("problem"), "seeds": seeds,
              "results": results, "source_artifacts": a.inputs,
              "unsupported_are_explicit": all(p.get("unsupported_are_explicit", False) for p in payloads),
              "complete": all(p.get("complete", False) for p in payloads)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(a.out), "seeds": seeds}))


if __name__ == "__main__": main()
