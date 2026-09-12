import json
from pathlib import Path


def test_negative_soundness_audit_artifact_is_all_passed():
    p = Path(__file__).parents[1] / "results" / "negative_soundness_v1.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert data["total_trials"] >= 500
    assert data["total_clear_checks"] >= 500
