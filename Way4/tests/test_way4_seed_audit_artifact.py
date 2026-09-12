import json
from pathlib import Path


def test_seed_audit_artifact_has_all_invariants():
    p = Path(__file__).parents[1] / "results" / "way4_seed_audit_1000_1004.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["all_invariants_passed"] is True
    assert len(data["rows"]) == 5
    assert all(r["full_clear"] and r["error"] is None for r in data["rows"])
