import inspect

import pytest

from way4.planner.candidates import CandidateGenerator


def test_formal_sensing_efficiency_gate_is_enabled_by_default():
    default = inspect.signature(CandidateGenerator).parameters["min_sensing_efficiency"].default
    assert default > 0.0
    assert CandidateGenerator(min_sensing_efficiency=0.0).min_sensing_efficiency == 0.0


def test_sensing_efficiency_gate_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        CandidateGenerator(min_sensing_efficiency=-1.0)
