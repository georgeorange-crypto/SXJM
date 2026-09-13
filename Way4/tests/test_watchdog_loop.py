from way4.rl.final_planner import ProgressWatchdog


def test_watchdog_detects_alternating_action_loop():
    w = ProgressWatchdog()
    assert not w.observe((0,), action_signature=('A', 0, 0))
    assert not w.observe((0,), action_signature=('B', 1, 0))
    assert not w.observe((0,), action_signature=('A', 0, 0))
    assert w.observe((0,), action_signature=('B', 1, 0))


def test_watchdog_reset_clears_loop_history():
    w = ProgressWatchdog()
    for a in ('A', 'B', 'A', 'B'):
        w.observe((0,), action_signature=(a,))
    w.reset()
    assert not w.observe((0,), action_signature=('A',))
