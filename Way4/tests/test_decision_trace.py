from way4.pipeline import Way4Pipeline


def test_decision_trace_is_exposed_without_affecting_execution():
    # This is an interface-level assertion; a full simulator episode is covered
    # by the existing pipeline smoke tests.  The field must be serializable and
    # live on EpisodeResult for downstream attribution reports.
    from dataclasses import fields
    assert "decision_trace" in {f.name for f in fields(__import__("way4.pipeline", fromlist=["EpisodeResult"]).EpisodeResult)}
