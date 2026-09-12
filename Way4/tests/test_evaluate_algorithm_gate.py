from pathlib import Path
import json
import subprocess
import sys


def test_gate_script_writes_verdict(tmp_path):
    source = tmp_path / "rows.json"
    out = tmp_path / "gate.json"
    source.write_text(json.dumps({"rows": [
        {"seed": 1, "mode": "none", "success_ground_truth": True, "virtual_time_s": 10, "resolved": 5},
        {"seed": 1, "mode": "dbscan", "success_ground_truth": False, "virtual_time_s": 5, "resolved": 4},
    ]}), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "evaluate_algorithm_gate.py"
    subprocess.run([sys.executable, str(script), "--input", str(source), "--out", str(out)], check=True)
    assert json.loads(out.read_text(encoding="utf-8"))["promote_default"] is False
