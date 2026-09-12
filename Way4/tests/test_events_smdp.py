from way4.core import EventBus, EventType, OptionTransition, Way4Event, transition_dataset
from way4.pipeline import Way4Pipeline


def test_event_types_and_option_transition_are_typed():
    event = Way4Event(EventType.BATCH_COMPLETED, 5.0, payload={"n": 1})
    assert event.event_type is EventType.BATCH_COMPLETED
    transition = OptionTransition(
        ((1, "UNKNOWN"),), "EXPLORE", (0.0, 0.0), (1,), 5.0, 1,
        ((1, "DETECTED"),), False, False,
    )
    assert transition.elapsed_time_s == 5.0
    assert transition.resulting_belief[0][1] == "DETECTED"


def test_event_bus_supports_global_and_typed_subscribers():
    bus = EventBus()
    seen, typed = [], []
    bus.subscribe(seen.append)
    bus.subscribe(typed.append, EventType.LOCALIZED)
    event = Way4Event(EventType.LOCALIZED, 2.0, channel=3)
    bus.publish(event)
    assert seen == [event]
    assert typed == [event]


def test_transition_dataset_is_json_ready_and_option_level():
    transition = OptionTransition(
        ((1, "UNKNOWN"),), "EXPLORE", (1.0, 2.0), (1,), 7.0, 2,
        ((1, "DETECTED"),), False, False,
    )
    record = transition_dataset([transition])[0]
    assert record["option_type"] == "EXPLORE"
    assert record["elapsed_time_s"] == 7.0
    assert "primitive" not in record


def test_pipeline_can_publish_read_only_events_to_external_bus():
    bus = EventBus(); seen = []
    bus.subscribe(seen.append, EventType.LOCALIZED)
    pipe = Way4Pipeline.__new__(Way4Pipeline)
    pipe.events = []; pipe.event_bus = bus
    event = Way4Event(EventType.LOCALIZED, 3.0, channel=4)
    pipe._emit_event(event)
    assert pipe.events == [event]
    assert seen == [event]
