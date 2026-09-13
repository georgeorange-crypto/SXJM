import pytest

from way4.planner.commitment import FocusController


def test_focus_selects_high_completion_rate_and_retains_until_exit_reason():
    f = FocusController(threshold=0.01)
    assert f.consider(3, 0.8, 40.0)
    assert f.state.channel == 3
    assert not f.may_exit()
    assert f.may_exit(no_progress=True)
    f.clear()
    assert not f.state.active


def test_focus_rejects_invalid_evidence():
    with pytest.raises(ValueError):
        FocusController(-1)
    f = FocusController()
    with pytest.raises(ValueError):
        f.consider(1, 1.2, 10)
