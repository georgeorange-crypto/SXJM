import json
from pathlib import Path

def test_ablation_audit_preserves_supported_and_unsupported_cells():
    p=Path(__file__).parents[1]/"results"/"ablation_matrix_real_2000.json"
    d=json.loads(p.read_text(encoding="utf-8"))
    assert d["unsupported_are_explicit"] is True
    assert d["results"]["way4_full"][0]["full_clear"] is True
    assert d["results"]["minus_no_signal"][0]["error"] == "unsupported_configuration"
    assert d["results"]["adaptive_rerank"][0]["full_clear"] is True
