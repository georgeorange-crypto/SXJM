import json
from pathlib import Path

def test_smdp_smoke_artifact_is_explicitly_not_real_rollout():
    p=Path(__file__).parents[1]/"results"/"smdp_learner_smoke.json"
    d=json.loads(p.read_text(encoding="utf-8"))
    assert d["kind"] == "smdp-learner-smoke-only"
    assert d["real_rollout_evidence"] is False
    assert d["clear_beats_scan"] is True
