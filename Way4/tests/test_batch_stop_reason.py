from way4.pipeline import Way4Pipeline


def test_pipeline_batch_audit_uses_explicit_replan_reason_field():
    # The field is part of the decision-audit schema even when no early stop occurs.
    pipe = Way4Pipeline.__new__(Way4Pipeline)
    pipe.decision_audit = [{"measure_count": 1, "batch_stop_reason": "replan"}]
    assert pipe.decision_audit[0]["batch_stop_reason"] == "replan"
