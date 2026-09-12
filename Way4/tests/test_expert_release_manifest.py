import json
from pathlib import Path


def test_deterministic_expert_release_manifest_is_frozen_and_math_authoritative():
    p = Path(__file__).parents[1] / "results" / "way4_deterministic_expert_release.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["expert_authority"] == "math_planner"
    assert data["learner_role"] == "Learning-Augmented Planner"
    assert data["action_vocabulary"] == ["SCAN", "REFINE", "CLEAR", "COVERAGE", "EXIT"]
    assert len(data["source_sha256"]) == 5
    assert len(data["expert_trace_files"]) >= 3
