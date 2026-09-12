import json
from pathlib import Path

def test_source_fingerprint_is_present_and_nonempty():
    p=Path(__file__).parents[1]/"results"/"way4_source_fingerprint.json"
    d=json.loads(p.read_text(encoding="utf-8"))
    assert d["working_tree_has_uncommitted_changes"] is True
    assert d["file_count"] > 100
    assert len(d["tree_sha256"]) == 64
