from way4.core.events import OptionTransition, transition_dataset
from way4.rl import SMDPTrainConfig, TabularSMDPLearner


def _transition(**kwargs):
    base = dict(
        start_belief=((1, "UNKNOWN"),), option_type="SCAN", target=(1.0, 2.0),
        channel_batch=(1,), elapsed_time_s=2.0, primitive_count=2,
        resulting_belief=((1, "LOCALIZED"),), terminal=True, full_clear=False,
    )
    base.update(kwargs)
    return OptionTransition(**base)


def test_independent_smdp_learner_uses_time_and_terminal_reward():
    learner = TabularSMDPLearner(SMDPTrainConfig(time_scale_s=2.0, terminal_reward=5.0))
    record = transition_dataset([_transition()])[0]
    assert learner.target(record) == -2.0 + 5.0
    assert learner.fit([record])["SCAN"] == -2.0 + 5.0


def test_smdp_learner_keeps_options_separate_and_full_clear_bonus():
    learner = TabularSMDPLearner()
    records = transition_dataset([
        _transition(option_type="REFINE", elapsed_time_s=1.0, terminal=False),
        _transition(option_type="CLEAR", elapsed_time_s=1.0, full_clear=True),
    ])
    values = learner.fit(records)
    assert set(values) == {"REFINE", "CLEAR"}
    assert values["CLEAR"] > values["REFINE"]
