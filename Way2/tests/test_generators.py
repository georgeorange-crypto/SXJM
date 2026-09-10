"""Case generators must obey every hard constraint of Problem B."""

from __future__ import annotations

import math

import pytest

from radio_rl.env.generators import (
    ARENA_RADIUS_M,
    CHANNEL_MAX,
    CHANNEL_MIN,
    JAMMER_COUNT_MAX,
    JAMMER_COUNT_MIN,
    R_EFF_MAX,
    R_EFF_MIN,
    STRESS_TYPES,
    generate_case,
    generate_stress_case,
)


def _check_constraints(case):
    n = len(case.jammers)
    assert JAMMER_COUNT_MIN <= n <= JAMMER_COUNT_MAX
    channels = [j.channel for j in case.jammers]
    assert len(set(channels)) == n, "channels must be distinct (<=1 source/channel)"
    for j in case.jammers:
        assert CHANNEL_MIN <= j.channel <= CHANNEL_MAX
        assert R_EFF_MIN - 1e-6 <= j.r_eff <= R_EFF_MAX + 1e-6
        assert math.hypot(j.x, j.y) <= ARENA_RADIUS_M + 1e-6
        assert j.kind in ("omni", "dir")
        if j.kind == "dir":
            assert j.direction_deg is not None


@pytest.mark.parametrize("seed", range(8))
def test_problem3_all_omni(seed):
    case = generate_case(seed=seed, problem=3)
    _check_constraints(case)
    assert all(j.kind == "omni" for j in case.jammers)


@pytest.mark.parametrize("seed", range(8))
def test_problem4_has_both_kinds(seed):
    case = generate_case(seed=seed, problem=4)
    _check_constraints(case)
    kinds = {j.kind for j in case.jammers}
    assert "dir" in kinds and "omni" in kinds


@pytest.mark.parametrize("stype", STRESS_TYPES)
def test_stress_cases_respect_constraints(stype):
    problem = 4 if stype.startswith("dir_") else 3
    case = generate_stress_case(stype, seed=1, problem=problem)
    _check_constraints(case)


def test_reveal_structure():
    case = generate_case(seed=0, problem=4)
    rv = case.reveal()
    assert set(rv) >= {"total", "n_omni", "n_dir", "jammers"}
    assert rv["total"] == len(case.jammers)
    assert rv["n_omni"] + rv["n_dir"] == rv["total"]


def test_max_and_min_count_stress():
    assert len(generate_stress_case("max_count", seed=2, problem=3).jammers) == JAMMER_COUNT_MAX
    assert len(generate_stress_case("min_count", seed=2, problem=3).jammers) == JAMMER_COUNT_MIN
