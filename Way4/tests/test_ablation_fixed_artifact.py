import json
from pathlib import Path

def test_fixed_ablation_artifact_has_real_successful_cells():
    p=Path(__file__).parents[1]/"results"/"ablation_fixed_supported_2000.json"
    d=json.loads(p.read_text(encoding="utf-8"))
    assert d["synthetic"] is False
    assert all(x["success"] and x["resolved"] == x["n_channels"]
               for x in d["cells"].values())
