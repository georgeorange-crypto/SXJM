"""Validate a Candidate-PPO checkpoint before starting an expensive evaluation."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parents[1] / "src")]
from way4.rl.candidate_ppo import MODEL_SCHEMA


def inspect_checkpoint(path: str, *, input_dim: int = 210, problem: int = 4):
    import torch

    p = Path(path)
    report = {
        "path": str(p),
        "exists": p.is_file(),
        "expected_schema": MODEL_SCHEMA,
        "expected_input_dim": input_dim,
        "expected_problem": problem,
    }
    if not p.is_file():
        report.update({"compatible": False, "reason": "missing_file"})
        return report
    blob = torch.load(p, map_location="cpu")
    report.update({
        "schema": blob.get("schema"),
        "input_dim": blob.get("input_dim"),
        "problem": blob.get("problem"),
    })
    checks = {
        "schema": blob.get("schema") == MODEL_SCHEMA,
        "input_dim": blob.get("input_dim", input_dim) == input_dim,
        "problem": blob.get("problem", problem) == problem,
        "state_dict": "state_dict" in blob,
    }
    report["checks"] = checks
    report["compatible"] = all(checks.values())
    report["reason"] = None if report["compatible"] else "schema_or_metadata_mismatch"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = inspect_checkpoint(args.checkpoint)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
