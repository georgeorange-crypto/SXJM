from dataclasses import fields
from way4.pipeline import EpisodeResult


def test_episode_result_exposes_decision_audit_field():
    # The field is part of the public result contract for every run.
    assert 'decision_audit' in {f.name for f in fields(EpisodeResult)}
