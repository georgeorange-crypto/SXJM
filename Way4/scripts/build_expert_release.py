"""Build a reproducible deterministic-expert release manifest.

This release is intentionally the mathematical expert only.  It records the
source hashes, action vocabulary, evaluation gate and trace command so a
learner cannot silently become the algorithm definition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "way4_deterministic_expert_release.json"
FILES = [
    ROOT / "src" / "way4" / "pipeline.py",
    ROOT / "src" / "way4" / "planner" / "receding_horizon.py",
    ROOT / "src" / "way4" / "planner" / "candidates.py",
    ROOT / "src" / "way4" / "sensing" / "nbv.py",
    ROOT / "src" / "way4" / "certificate" / "manager.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release": "way4-deterministic-expert-v1",
        "algorithm": "Certificate-Guided Active Belief Planning",
        "learner_role": "Learning-Augmented Planner",
        "expert_authority": "math_planner",
        "action_vocabulary": ["SCAN", "REFINE", "CLEAR", "COVERAGE", "EXIT"],
        "full_clear_guard": "SafetyShield/EXIT guard; learner cannot bypass",
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in FILES},
        "gate_artifact": "results/way4_p4_gate_2000_2009.json",
        "trace_command": "python scripts/run_way4_traces.py runs/expert_traces 1000 1001 1002",
        "expert_trace_files": sorted(
            str(p.relative_to(ROOT)) for p in (ROOT / "runs" / "expert_traces").glob("way4_detailed_trace_seed_*.json")
        ),
        "generated_by": "scripts/build_expert_release.py",
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "files": len(FILES), "release": manifest["release"]}))


if __name__ == "__main__":
    main()
