import pytest

from way4.planner import choose_refine_variant, evaluate_refine_variant


def test_refine_variants_use_joint_expected_time():
    dedicated = evaluate_refine_variant("dedicated", delta_route_s=8, measure_time_s=4,
                                        expected_remaining_s=20)
    piggyback = evaluate_refine_variant("piggyback", delta_route_s=2, measure_time_s=7,
                                        expected_remaining_s=15)
    assert dedicated.total_expected_s == 32
    assert piggyback.total_expected_s == 24
    assert choose_refine_variant((dedicated, piggyback)).candidate == "piggyback"


def test_refine_objective_rejects_negative_time():
    with pytest.raises(ValueError):
        evaluate_refine_variant("bad", delta_route_s=-1, measure_time_s=1,
                                expected_remaining_s=1)
