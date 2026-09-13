"""Audit AI01--AI04 gates from an ablation result artifact.

The command never treats an incomplete row as a benchmark pass.  It emits an
explicit ``insufficient_evidence`` status until every selected seed has a
valid full-clear result and the required correctness flags are present.
"""
import argparse
import json
import statistics
from pathlib import Path

from way4.analysis.performance_gate import GATES, performance_gate


def audit(payload, soundness=None):
    output = {"source_complete": bool(payload.get("complete", False)), "cells": {}}
    for name, rows in payload.get("results", {}).items():
        valid = [r for r in rows if isinstance(r, dict) and r.get("virtual_time_s") is not None]
        complete_rows = [r for r in valid if r.get("full_clear") is True]
        evidence_ok = bool(valid) and len(valid) == len(rows) and len(complete_rows) == len(rows)
        cell = {"n_rows": len(rows), "n_valid": len(valid),
                "full_clear_rate": (len(complete_rows) / len(valid) if valid else None),
                "evidence_status": "ready" if evidence_ok else "insufficient_evidence",
                "gates": {}}
        if evidence_ok:
            times = [float(r["virtual_time_s"]) / max(1, int(r.get("n_channels", 1))) for r in valid]
            mean_per_target = statistics.mean(times)
            # Current offline rows do not carry independent correctness proofs;
            # keep the gate conservative and require those fields explicitly.
            clear_correct = all(r.get("clear_correct") is True or
                                (r.get("illegal_clear") == 0 and
                                 r.get("safety_violation") == 0) for r in valid)
            certificate_sound = bool(soundness and soundness.get("all_passed") is True)
            for gate in GATES:
                cell["gates"][gate] = performance_gate(
                    mean_time_s_per_target=mean_per_target,
                    full_clear_rate=1.0,
                    clear_correct=clear_correct,
                    certificate_sound=certificate_sound,
                    gate=gate,
                )
            cell["mean_time_s_per_target"] = mean_per_target
            cell["median_time_s_per_target"] = statistics.median(times)
            cell["p90_time_s_per_target"] = max(times) if len(times) < 2 else statistics.quantiles(times, n=10, method="inclusive")[8]
            cell["max_time_s_per_target"] = max(times)
        output["cells"][name] = cell
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--soundness", type=Path,
                        help="independent negative-soundness audit JSON")
    args = parser.parse_args()
    soundness = (json.loads(args.soundness.read_text(encoding="utf-8"))
                 if args.soundness else None)
    result = audit(json.loads(args.input.read_text(encoding="utf-8")), soundness)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.out), "cells": len(result["cells"])}))


if __name__ == "__main__":
    main()
