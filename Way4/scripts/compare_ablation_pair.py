"""Compare two same-seed ablation artifacts without hiding correctness failures."""
import argparse, json
from pathlib import Path


def compare(a, b):
    rows_a = {r["seed"]: r for rs in a.get("results", {}).values() for r in rs}
    rows_b = {r["seed"]: r for rs in b.get("results", {}).values() for r in rs}
    pairs = []
    for seed in sorted(set(rows_a) & set(rows_b)):
        x, y = rows_a[seed], rows_b[seed]
        pairs.append({"seed": seed, "a_full_clear": bool(x.get("full_clear")),
                      "b_full_clear": bool(y.get("full_clear")),
                      "a_time_s": x.get("virtual_time_s"), "b_time_s": y.get("virtual_time_s"),
                      "delta_b_minus_a_s": (float(y["virtual_time_s"]) - float(x["virtual_time_s"]))
                      if x.get("virtual_time_s") is not None and y.get("virtual_time_s") is not None else None,
                      "a_steps": x.get("steps"), "b_steps": y.get("steps")})
    return {"artifact_a": a.get("results", {}), "artifact_b": b.get("results", {}),
            "n_pairs": len(pairs), "pairs": pairs}


def main():
    p = argparse.ArgumentParser(); p.add_argument("a", type=Path); p.add_argument("b", type=Path)
    p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    result = compare(json.loads(a.a.read_text(encoding="utf-8")), json.loads(a.b.read_text(encoding="utf-8")))
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(a.out), "n_pairs": result["n_pairs"]}))


if __name__ == "__main__": main()
