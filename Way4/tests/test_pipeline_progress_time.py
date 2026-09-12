from dataclasses import fields
from way4.pipeline import EpisodeResult


def test_episode_result_has_real_no_progress_time_contract():
    assert 'no_progress_time_s' in {f.name for f in fields(EpisodeResult)}
