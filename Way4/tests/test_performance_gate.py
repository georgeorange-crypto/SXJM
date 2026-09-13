import pytest
from way4.analysis import performance_gate


def test_final_gate_requires_safety_and_full_clear():
    assert performance_gate(mean_time_s_per_target=400, full_clear_rate=1,
                            clear_correct=True, certificate_sound=True, gate="FINAL")["passed"]
    assert not performance_gate(mean_time_s_per_target=100, full_clear_rate=1,
                                clear_correct=False, certificate_sound=True)["passed"]


def test_gate_rejects_partial_full_clear():
    result = performance_gate(mean_time_s_per_target=100, full_clear_rate=.99,
                              clear_correct=True, certificate_sound=True)
    assert not result["passed"]
