from pathlib import Path

import torch

from scripts.validate_candidate_checkpoint import inspect_checkpoint
from way4.rl.candidate_ppo import MODEL_SCHEMA


def test_checkpoint_validation_reports_legacy_schema(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"input_dim": 210, "problem": 4, "state_dict": {}}, path)
    report = inspect_checkpoint(str(path))
    assert report["compatible"] is False
    assert report["reason"] == "schema_or_metadata_mismatch"
    assert report["checks"]["schema"] is False


def test_checkpoint_validation_accepts_current_metadata(tmp_path):
    path = tmp_path / "current.pt"
    torch.save({"schema": MODEL_SCHEMA, "input_dim": 210, "problem": 4,
                "state_dict": {}}, path)
    report = inspect_checkpoint(str(path))
    assert report["compatible"] is True


def test_checkpoint_validation_reports_missing_file(tmp_path):
    report = inspect_checkpoint(str(Path(tmp_path) / "missing.pt"))
    assert report == {
        "path": str(Path(tmp_path) / "missing.pt"),
        "exists": False,
        "expected_schema": MODEL_SCHEMA,
        "expected_input_dim": 210,
        "expected_problem": 4,
        "compatible": False,
        "reason": "missing_file",
    }
