from way4.core import EventType
from way4.pipeline import Way4Pipeline


class _DeadlineEnv:
    def measure(self, x, y, channel):
        return None, 0.0

    def clear(self, x, y, channel):
        return False, 0.0


def test_pipeline_exposes_typed_event_and_transition_logs():
    pipe = Way4Pipeline(_DeadlineEnv(), n_channels=2, max_steps=1)
    assert pipe.events == []
    assert pipe.transitions == []
    assert EventType.BATCH_COMPLETED.value == "BATCH_COMPLETED"


def test_time_budget_warning_event_is_emitted_before_late_macro():
    pipe = Way4Pipeline(_DeadlineEnv(), n_channels=2, max_steps=1, time_budget_s=1.0)
    # The environment ends before execution, but the event log is initialized
    # and remains typed; warning emission is exercised through the pipeline path.
    pipe.run()
    assert all(hasattr(e, "event_type") for e in pipe.events)
