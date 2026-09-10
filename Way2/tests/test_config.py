"""Config composition: defaults resolve, CLI overrides and group swaps apply."""

from __future__ import annotations

from radio_rl.core.config import compose, to_container


def test_defaults_compose():
    cfg = compose()
    assert int(cfg.problem) == 3
    assert str(cfg.env.type) == "local"
    assert str(cfg.algorithm.type) == "heuristic"
    assert str(cfg.algorithm.name) == "greedy_math"
    assert str(cfg.error_field.kind) == "smooth"
    assert int(cfg.candidates.k_max) == 32


def test_value_override():
    cfg = compose(overrides=["problem=4", "seed=5", "evaluation.n_episodes=7"])
    assert int(cfg.problem) == 4
    assert int(cfg.seed) == 5
    assert int(cfg.evaluation.n_episodes) == 7


def test_group_swap():
    cfg = compose(overrides=["error_field=iid"])
    assert str(cfg.error_field.kind) == "iid"
    cfg2 = compose(overrides=["algorithm=heuristic", "safety=default"])
    assert str(cfg2.algorithm.name) == "greedy_math"


def test_to_container_roundtrips():
    cfg = compose()
    d = to_container(cfg)
    assert isinstance(d, dict)
    assert d["problem"] == 3
    assert d["candidates"]["k_max"] == 32
