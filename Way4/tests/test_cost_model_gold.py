"""M4 acceptance — reproduce the §1.1 golden timing sample with the cost model.

DESIGN.md §1.1 table 2 is the oracle: from (0,0), initial channel 1, the command
sequence must produce the running virtual clock 105 / 111 / 194 / 199. The subtle
rules it pins down (and this test locks in):
  * ``/measure`` pays switch (1 s) only when the channel changes, and *sets* the
    measuring channel;
  * ``/clear`` pays no switch and does NOT change the measuring channel — so step 5
    (``/measure`` ch2 right after ``/clear`` ch3, with ch2 still current) costs 0
    switch.
"""

from way4.core import AnalyticalCostModel, RobotState


def test_us_rounding_is_per_component():
    cost = AnalyticalCostModel()
    # a non-integer move: 333 m / 5 = 66.6 s -> 66_600_000 us
    s = RobotState()
    u = cost.move_us((0.0, 0.0), (333.0, 0.0))
    assert u == 66_600_000


def test_gold_sample_step_costs():
    """Each step's own duration matches the §1.1 table."""
    cost = AnalyticalCostModel()
    s = RobotState()  # (0,0), channel 1, t=0

    s, dt2 = cost.apply_measure(s, (300.0, 400.0), 1)   # move 500->100, switch 0, detect 5
    assert dt2 == 105.0
    s, dt3 = cost.apply_measure(s, (300.0, 400.0), 2)   # move 0, switch 1, detect 5
    assert dt3 == 6.0
    s, dt4 = cost.apply_clear(s, (300.0, 0.0), hit=False)  # move 400->80, miss 3
    assert dt4 == 83.0
    s, dt5 = cost.apply_measure(s, (300.0, 0.0), 2)     # move 0, switch 0 (still ch2), detect 5
    assert dt5 == 5.0


def test_gold_sample_running_clock_and_channel():
    """The accumulated virtual clock and the measuring-channel state track the table."""
    cost = AnalyticalCostModel()
    s = RobotState()

    s, _ = cost.apply_measure(s, (300.0, 400.0), 1)
    assert (s.virtual_time_s, s.channel) == (105.0, 1)

    s, _ = cost.apply_measure(s, (300.0, 400.0), 2)
    assert (s.virtual_time_s, s.channel) == (111.0, 2)

    s, _ = cost.apply_clear(s, (300.0, 0.0), hit=False)
    # crux: /clear moved the robot and advanced the clock but left channel == 2
    assert (s.virtual_time_s, s.channel) == (194.0, 2)

    s, _ = cost.apply_measure(s, (300.0, 0.0), 2)
    assert (s.virtual_time_s, s.channel) == (199.0, 2)


def test_step5_switch_is_zero_because_clear_does_not_switch():
    """Isolate the headline invariant: without the clear-doesn't-switch rule, step 5
    would cost 6 s, not 5 s, and the final clock would be 200, not 199."""
    cost = AnalyticalCostModel()
    s = RobotState(x=300.0, y=400.0, channel=2, vt_us=111_000_000)
    s, _ = cost.apply_clear(s, (300.0, 0.0), hit=False)   # channel stays 2
    assert s.channel == 2
    s, dt5 = cost.apply_measure(s, (300.0, 0.0), 2)        # same channel -> no switch
    assert dt5 == 5.0
    assert s.virtual_time_s == 199.0


def test_clear_hit_costs_five():
    cost = AnalyticalCostModel()
    s = RobotState(x=300.0, y=0.0, channel=2, vt_us=0)
    _, dt = cost.apply_clear(s, (300.0, 0.0), hit=True)    # move 0, hit 5
    assert dt == 5.0


# --- batch scan (§7): one move amortised over several measurements ------------


def test_batch_scan_cost_amortises_one_move():
    """EXPLORE at a far waypoint scanning three channels pays the move once."""
    cost = AnalyticalCostModel()
    s = RobotState()  # (0,0), ch1
    # move 500->100, then ch1 detect 5 (no switch), ch2 switch1+detect5, ch3 switch1+detect5
    total = cost.batch_scan_time_s(s, (300.0, 400.0), (1, 2, 3))
    assert total == 100.0 + 5.0 + 6.0 + 6.0   # == 117.0

    # scanning each channel from scratch (separate trips) would pay the move 3x
    naive = 0.0
    for c in (1, 2, 3):
        _, dt = cost.apply_measure(RobotState(), (300.0, 400.0), c)
        naive += dt
    assert naive > total
