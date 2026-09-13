from way4.pipeline import EpisodeResult


def test_episode_result_exposes_batch_audit_fields():
    result = EpisodeResult(False, 0., 0., 0, 0, 0, 0, 0, 0, 0, 1, 0, False, False)
    assert result.scan_batch_count == 0
    assert result.batch_stop_reason == []
